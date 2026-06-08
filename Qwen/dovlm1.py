import os
import sys
from loguru import logger
from pathlib import Path
import argparse
import re
import datetime
import json

import torch
import numpy as np
from models.qwen2_5_vl_vla import Qwen2_5_VL_lm_hand
from transformers import AutoProcessor, TextIteratorStreamer, Qwen2_5_VLForConditionalGeneration, AutoTokenizer
from qwen_vl_utils import process_vision_info
from PIL import ImageFile

from tools.utils import makepath, LOGGER_DEFAULT_FORMAT

# 允许加载被截断的图像，避免 OSError: image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True


# 加载VLM模型和处理器
def load_model_processor(args):
    if args.gpu_ids:
        device_map = args.device # 'auto' # args.device
        logger.info(f"Using GPU: {device_map}")
    else:
        device_map = 'cpu'

    # Check if flash-attn2 flag is enabled and load model accordingly
    if args.flash_attn2:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.checkpoint_path,
                                                   torch_dtype=torch.bfloat16, #'auto',
                                                   attn_implementation='flash_attention_2',
                                                   device_map=device_map)
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.checkpoint_path, device_map=device_map)

    processor = AutoProcessor.from_pretrained(args.checkpoint_path,
                                              use_fast=True)
    '''
    #### 🔍 作用
    - **统一处理文本和视觉信息**
    `AutoProcessor` 是 HuggingFace 提供的统一接口，会根据模型路径自动加载对应的：
    - 文本分词器（`Qwen2_5VLMTokenizer`）
    - 图像处理器（`Qwen2_5VLImageProcessor`）
    - 视频处理器（`Qwen2_5VLVideoProcessor`）

    #### 🧠 内部逻辑
    1. **自动识别模型类型**
    根据 `args.checkpoint_path`（如 `Qwen/Qwen2.5-VL-3B-Instruct`）加载对应的处理器配置。
    2. **组合多模态处理器**
    3. **核心功能**
    - 文本编码：tokenizer 将文本转换为 `input_ids`
    - 图像预处理：`image_processor` 将图像转换为 `pixel_values`
    - 视频预处理：`video_processor` 将视频转换为 `pixel_values_videos`
    '''
    # tokenizer = processor.tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, use_fast=True)

    # print(model) # 打印模型结构
    return model, processor, tokenizer


# 获取 input 的信息
def extract_info_from_text_json(text_json):
    """从 JSON 文件名中提取 sub_id 和 action 和 index"""
    filename = os.path.basename(text_json)
    match = re.match(r"sub(\d+)_(\w+)_(\d+)\.json", filename)
    if not match:
        raise ValueError(f"Invalid prompt filename: {filename}")
    sub_id, action, index = match.groups()
    return sub_id, action, index


# 获取 input 的图片
def find_image_files(sub_id, action, index):
    """查找 view1/view2 中对应的 3 张图片"""
    image_dir = Path(IMAGE_PATH) / f"sub{sub_id}" / f"{action}_{index}"
    image_crop_dir = Path(IMAGE_CROP_PATH) / f"sub{sub_id}" / f"{action}_{index}"
    
    # 修改排序逻辑，提取文件名中的数字部分
    def extract_number(path):
        # 从文件名中提取数字，例如 'cropped_00054' -> 54, '00109' -> 109
        stem = path.stem
        # 尝试匹配 'cropped_数字' 或纯数字格式
        match = re.search(r'(\d+)$', stem)
        if match:
            return int(match.group(1))
        return 0
    image_files = sorted(image_dir.glob('*.*'), key=extract_number)    
    image_crop_files = sorted(image_crop_dir.glob('*.*'), key=extract_number)
    image_files = image_files + image_crop_files

    return [str(f) for f in image_files]


# 构建 input messages
def build_messages(prompt, image_paths):
    """构建 messages，包含多个图像和文本"""
    content = []
    for image_path in image_paths:
        content.append({"type": "image", "image": f"file://{image_path}"})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def do_vlm(arg, text_json, sub_id, action, index):
    """处理输入 & 送入模型 & 得到输出"""
    import time
    logger.info(f"Processing {text_json}")
    
    # 开始计时整个处理流程
    total_start_time = time.time()

    # 读取 JSON 内容作为 text 描述
    try:
        with open(text_json, 'r', encoding='utf-8') as f:
            # text = f.read().strip()
            text_json_data = json.load(f)  # 把整个 JSON 文件解析成字典
        if not text_json_data:
            logger.warning(f"Empty JSON content in {text_json}")
            return None, None
        text = next(iter(text_json_data.values()))
        # 使用模板构建最终 prompt
        prompt = PROMPT_TEMPLATE.format(text=text)
        logger.info(f"Write prompt for text: {text}")    
    except Exception as e:
        logger.error(f"Failed to read JSON file {text_json}: {e}")
        return None, None

    # 获取 image 图片
    try:
        images_view1 = find_image_files(sub_id, action, index)
    except FileNotFoundError as e:
        logger.error(f"Directory not found: {e}")
        return None, None
    images_all = images_view1# + images_view2
    
    # 只取第二张图片 ##########################################
    # images_all = [images_view1[1]] if len(images_view1) > 1 else images_view1 # 只取第二张图片
    logger.info(f"Total images: {len(images_all)}")

    # 构建 messages
    messages = build_messages(prompt, images_all)

    # Preparation for inference
    text_inputs = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text_inputs], \
                       images=image_inputs, videos=video_inputs, \
                       padding=True, return_tensors="pt").to(model.device)

    # 提取 文本信息中对应的 {text} 片段的 embedding索引
    # 1) 获取 text 的 token ids
    # text_ids = tokenizer(text, add_special_tokens=False).input_ids
    text_ids = tokenizer(text, add_special_tokens=False, padding=True, truncation=True).input_ids #, max_length=512

    # word_embeddings = model.get_input_embeddings()
    # ec_A = word_embeddings(torch.tensor(text_ids, device=model.device))  # shape: [num_tokens, hidden_dim]
    # logger.debug(f"ec_A.shape: {ec_A.shape}")
    # logger.debug(f"ec_A: {ec_A}")

    if isinstance(text_ids[0], (list, tuple)):
        text_ids = list(text_ids[0])
    else:
        text_ids = list(text_ids)

    # 2) 获取 multimodal inputs token ids
    haystack = inputs["input_ids"][0].tolist()

    # 3) approximate sublist match
    def find_best_sublist(haystack, needle):
        best_pos = -1
        best_len = 0
        n = len(haystack)
        m = len(needle)
        for i in range(n - m + 1):
            cnt = sum([1 for k in range(m) if haystack[i+k]==needle[k]])
            if cnt > best_len:
                best_len = cnt
                best_pos = i
            # 完全匹配直接返回
            if cnt == m:
                return i, i+m
        if best_len == 0:
            return -1, -1
        return best_pos, best_pos+best_len

    start_in_inputs, end_in_inputs = find_best_sublist(haystack, text_ids)
    if start_in_inputs == -1:
        logger.error("   multimodal inputs 找到 text 的 token 子序列")
        return

    logger.debug(f"prompt: {prompt[:200]}")
    logger.debug(f"text: {text}")
    logger.debug(f"start_in_inputs: {start_in_inputs}")
    logger.debug(f"end_in_inputs: {end_in_inputs}")
    logger.debug(f"已提取embedding索引")

    # [用 model.generate 的]
    # Inference: Generation of the output
    model.eval()
    
    # 开始计时模型推理
    torch.cuda.synchronize()
    inference_start_time = time.time()
    
    with torch.no_grad():
        outputs = model(**inputs,
                        output_hidden_states=True)
    
    # 结束计时模型推理
    torch.cuda.synchronize()
    inference_end_time = time.time()
    model_inference_time = inference_end_time - inference_start_time
    
    # outputs 对象现在是一个包含多个字段的类，其中就包括你想要的 hidden_states
    # 提取中间层
    hidden_states = outputs.hidden_states[arg.select_layer]
    

    # 4) 提取 embedding
    text_embeds = hidden_states[0, start_in_inputs:end_in_inputs, :]
    # text_repr = text_embeds.mean(dim=0)

    logger.debug(f"ec_B.shape: {text_embeds.shape}")
    logger.debug(f"ec_B: {text_embeds}")


    # 保存为 .npy 文件
    base_name = os.path.splitext(os.path.basename(text_json))[0]
    output_path = os.path.join(OUTPUT_PATH, f"{base_name}_{arg.select_layer}_visual.pt")
    torch.save(hidden_states.float().cpu(), output_path)
    logger.info(f"Saved visual embedding to {output_path}")

    # 保存 {text} 的 embedding
    base_name = os.path.splitext(os.path.basename(text_json))[0]
    output_path = os.path.join(OUTPUT_PATH, f"{base_name}_{arg.select_layer}_textual.pt")
    torch.save(text_embeds.float().cpu(), output_path)
    logger.info(f"Saved textual embedding -> {output_path}")


    # 显式删除 outputs 和 inputs（避免显存累积）（可选）
    del inputs, outputs, hidden_states, text_embeds
    torch.cuda.empty_cache()  # 及时清理显存
    
    # 计算总处理时间
    total_end_time = time.time()
    total_processing_time = total_end_time - total_start_time
    
    # 返回两个时间：模型推理时间 和 总处理时间
    return model_inference_time, total_processing_time


def main(args):
    # 获取所有 JSON 文件
    all_jsons  = Path(TEXT_PATH).glob("*.json")

    # 根据模式筛选 JSON 文件
    mode = args.data_mode
    filtered_jsons = []
    for json_file in all_jsons:
        match = re.match(r"sub(\d+)_.*\.json", json_file.name)
        if match:
            num = int(match.group(1))
            if mode == 'train' and 1 <= num <= 15:
                filtered_jsons.append(json_file)
            elif mode == 'test' and 16 <= num <= 17:
                filtered_jsons.append(json_file)

    # 排序后返回
    text_jsons = sorted(filtered_jsons)
    text_jsons = [os.path.basename(json_file) for json_file in text_jsons]
    
    # 检查对应的图像文件是否存在
    valid_text_jsons = []
    for json_filename in text_jsons:
        parts = json_filename.split('_', 1)  # 只分割第一个下划线
        image_path = os.path.join(IMAGE_PATH, parts[0], parts[1].split('.')[0])
        if os.path.exists(image_path):
            valid_text_jsons.append(os.path.join(TEXT_PATH, json_filename))
        else:
            logger.debug(f"Image file not found: {image_path}")
    
    text_jsons = valid_text_jsons
    if not text_jsons:
        logger.debug("No prompt files found.")
        return

    # 初始化推理时间统计
    warmup_samples = 3  # 默认预热样本数
    max_stats_samples = -1  # -1表示统计所有样本
    
    total_model_inference_time = 0.0
    total_processing_time = 0.0
    inference_count = 0
    inference_times_per_sample = []
    stats_complete = False
    sample_counter = 0

    # 处理每个 JSON 文件
    for text_json in text_jsons:
        text_json = str(text_json)
        try:
            sub_id, action, index = extract_info_from_text_json(text_json)
        except ValueError as e:
            logger.debug(e)
            continue

        # 调用 do_vlm 并获取推理时间
        model_time, total_time = do_vlm(args, text_json, sub_id, action, index)
        
        if model_time is not None and total_time is not None:
            # 记录推理时间
            if sample_counter < warmup_samples:
                logger.info(f"[Warmup {sample_counter + 1}/{warmup_samples}] 模型推理: {model_time:.4f}s, 总处理: {total_time:.4f}s (不计入统计)")
            elif not stats_complete:
                total_model_inference_time += model_time
                total_processing_time += total_time
                inference_count += 1
                inference_times_per_sample.append({
                    'sample_idx': sample_counter,
                    'sub_id': sub_id,
                    'action': action,
                    'index': index,
                    'model_inference_time': float(model_time),
                    'total_processing_time': float(total_time),
                    'preprocessing_time': float(total_time - model_time)
                })
                logger.info(f"[样本 {sample_counter}] 模型推理: {model_time:.4f}s, 总处理: {total_time:.4f}s")
                
                # 检查是否达到统计上限
                if max_stats_samples > 0 and inference_count >= max_stats_samples:
                    stats_complete = True
                    logger.info(f"\n✓ 已收集 {inference_count} 个样本的推理时间，后续样本不再统计\n")
            else:
                logger.info(f"[样本 {sample_counter}] 模型推理: {model_time:.4f}s, 总处理: {total_time:.4f}s (已达到统计上限，不计入)")
            
            sample_counter += 1

    logger.info(f"All {mode}-Dataset items processed.")
    
    # 保存推理时间统计到JSON
    if inference_count > 0:
        avg_model_inference_time = total_model_inference_time / inference_count
        avg_processing_time = total_processing_time / inference_count
        avg_preprocessing_time = (total_processing_time - total_model_inference_time) / inference_count
        
        inference_stats = {
            'warmup_samples': warmup_samples,
            'max_stats_samples': max_stats_samples if max_stats_samples > 0 else 'all',
            'total_samples_measured': inference_count,
            'model_inference': {
                'total_time': float(total_model_inference_time),
                'average_time': float(avg_model_inference_time),
                'description': '只包含模型前向传播(获取hidden_states)的时间'
            },
            'total_processing': {
                'total_time': float(total_processing_time),
                'average_time': float(avg_processing_time),
                'description': '包含数据预处理、模型推理、后处理的完整时间'
            },
            'preprocessing': {
                'total_time': float(total_processing_time - total_model_inference_time),
                'average_time': float(avg_preprocessing_time),
                'description': '数据预处理和后处理的时间'
            },
            'per_sample_times': inference_times_per_sample
        }
        
        # 保存到JSON文件
        inference_time_json_path = os.path.join(OUTPUT_PATH, "inference_time_statistics.json")
        with open(inference_time_json_path, 'w') as f:
            json.dump(inference_stats, f, indent=4)
        
        logger.info("\n" + "="*60)
        logger.info("推理时间统计:")
        logger.info(f"  预热样本数: {warmup_samples}")
        if max_stats_samples > 0:
            logger.info(f"  目标统计样本数: {max_stats_samples}")
        else:
            logger.info(f"  目标统计样本数: 全部")
        logger.info(f"  实际统计样本数: {inference_count}")
        logger.info(f"")
        logger.info(f"  模型推理时间 (前向传播):")
        logger.info(f"    - 总时间: {total_model_inference_time:.4f} 秒")
        logger.info(f"    - 平均时间: {avg_model_inference_time:.4f} 秒")
        logger.info(f"")
        logger.info(f"  总处理时间 (含预处理+推理+后处理):")
        logger.info(f"    - 总时间: {total_processing_time:.4f} 秒")
        logger.info(f"    - 平均时间: {avg_processing_time:.4f} 秒")
        logger.info(f"")
        logger.info(f"  预处理+后处理时间:")
        logger.info(f"    - 总时间: {total_processing_time - total_model_inference_time:.4f} 秒")
        logger.info(f"    - 平均时间: {avg_preprocessing_time:.4f} 秒")
        logger.info(f"")
        logger.info(f"  统计结果已保存至: {inference_time_json_path}")
        logger.info("="*60 + "\n")


# 日志配置 loguru.logger
def config_logger(logger_path):
    logger.add(logger_path,  backtrace=True, diagnose=True)
    logger.add(lambda x:x,
                level="INFO",
                colorize=True,
                format=LOGGER_DEFAULT_FORMAT
                )


# 参数获取 cmd->args
def get_args():
    parser = argparse.ArgumentParser(description="Chatting HOI images with Qwen2.5-VL model")
    
    # 项目相关参数
    parser.add_argument('-p', '--project-name', type=str,
                       default='infer',
                       help='项目名称，用于区分不同实验')
    # 模型相关参数
    parser.add_argument('-c', '--checkpoint-path', type=str, 
                       default='Qwen/Qwen2.5-VL-3B-Instruct',
                       help='模型检查点路径 (default: %(default)s)')
    parser.add_argument('--max-new-tokens', type=int,
                       default=512,
                       help='最大生成token数 (default: %(default)s)')
    parser.add_argument('--flash-attn2',
                       action='store_true',
                       default=True,
                       help='启用 flash_attention_2 加速推理')
    parser.add_argument('--select-layer', type=int,
                       default=12,
                       help='要提取的中间层 层数')
    # 数据路径参数args.data_mode
    parser.add_argument('--data-mode', type=str,
                    default='train',
                    help='数据集分类 (default: %(default)s)')
    parser.add_argument('-i', '--image-dir', type=str,
                       default=' ',
                       help='图片文件目录 (default: %(default)s)')
    parser.add_argument('-ic', '--image-crop-dir', type=str,
                       default=' ',
                       help='图片文件目录 (default: %(default)s)')            
    parser.add_argument('-v', '--video-dir', type=str,
                       default='/data/videos',
                       help='视频文件目录 (default: %(default)s)')
    parser.add_argument('-t', '--text-dir', type=str,
                       default='./processed_data/omomo_text_anno_json_data',
                       help='提示文件路径 (default: %(default)s)')
    parser.add_argument('-o', '--output-dir', type=str,
                       default='./processed_data/',
                       help='输出保存目录 (default: %(default)s)')  
    # 环境变量
    parser.add_argument('--hf-home', type=str,
                       default='./Qwen/Qwen2.5-VL/Qwen_model',
                       help='设置 HF_HOME 环境变量 (default: %(default)s)')
    parser.add_argument('--gpu-ids', type=str,
                       default='0',
                       help='指定使用的GPU ID列表，逗号分隔（如 \'0,1,2\'）')
    args = parser.parse_args()

    # 设置环境变量
    os.environ["HF_HOME"] = args.hf_home
    
    if args.gpu_ids:
        try:
            gpu_ids_list = [int(x) for x in args.gpu_ids.split(',')] # 验证 GPU ID 格式是否正确
            assert all(gpu_id >= 0 for gpu_id in gpu_ids_list), "GPU ID 必须为非负整数" # 验证 GPU ID 是否为非负数
            # 如果通过验证，则设置 GPU 环境变量和设备
            # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
            args.device = f"cuda:{args.gpu_ids}"
        except (ValueError, AssertionError) as e:
            logger.debug(f"Invalid GPU IDs: {e}")
            raise ValueError(f"Invalid GPU IDs: {e}")
    else:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


if __name__ == "__main__":
    # 配置 args & cfgs
    args = get_args()
    CKPT_PATH = args.checkpoint_path
    TEXT_PATH = args.text_dir
    IMAGE_PATH = args.image_dir
    IMAGE_CROP_PATH = args.image_crop_dir
    VIDEO_PATH = args.video_dir

    # 生成当前时间戳作为项目名称
    timestamp = datetime.datetime.now().strftime("%y%m%d")
    PROJECT_NAME = f"{timestamp}-{args.project_name}"

    OUTPUT_PATH = os.path.join(args.output_dir, PROJECT_NAME)
    os.makedirs(OUTPUT_PATH, exist_ok=True) # 确保输出目录存在
    logger_path = makepath(os.path.join(OUTPUT_PATH, '%s.log' % (PROJECT_NAME)), isfile=True)
    config_logger(logger_path)

    PROMPT_TEMPLATE="""
    We are conducting the text-to-HOI motion generation task and the given textual description is: {text}. \
    We want to extract motion priors from the following reference images to \
    facilitate Human-Object-Interaction motion generation. \
    These priors include the human pose, the shape and size of the object, \
    and the contact region on the object when interaction happens, etc. \
    The initial position of the object is in front of the person.
    """

    # 加载 模型和处理器
    model, processor, tokenizer = load_model_processor(args)

    # 运行 vlm
    main(args)