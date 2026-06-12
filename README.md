# ViHOI: Human-Object Interaction Synthesis with Visual Priors

This repository contains the implementation for the **CVPR 2026** paper
**"ViHOI: Human-Object Interaction Synthesis with Visual Priors"**.


## Environment Setup

`Diffusion Environment` is used for ViHOI training and inference. `VLM Environment` is only needed if
you need to regenerate Qwen2.5-VL visual/textual embeddings.

### Diffusion Environment

For a fresh `diffusion environment`, follow the CHOIS setup:

```bash
conda create -n chois_env python=3.8
conda activate chois_env
```

Install PyTorch:

```bash
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

Install PyTorch3D:

```bash
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
```

Install `human_body_prior`, BPS, and other dependencies:

```bash
pip install tqdm dotmap PyYAML omegaconf loguru
cd human_body_prior
python setup.py develop
cd ..

pip install git+https://github.com/otaheri/chamfer_distance
pip install git+https://github.com/otaheri/bps_torch
pip install git+https://github.com/openai/CLIP.git
pip install -r requirements.txt
```

### VLM Environment

The Qwen environment is used by `Qwen/dovlm1.py` to extract visual/textual
hidden-state embeddings from Qwen2.5-VL.

```bash
conda env create -n qwen -f Qwen/environment.yml
conda activate qwen
```

If you prefer to install it manually, the key packages are `transformers`,
`qwen-vl-utils`, `accelerate`, `flash-attn`, `torch`, `torchvision`, `pillow`,
`loguru`, and their CUDA-compatible dependencies.

## Prerequisites

Please download SMPL-X/SMPL-H models from the official websites. The current
code expects the body models under:

```text
data/smpl_all_models/
processed_data/smpl_all_models/
```

The training and inference scripts use `./processed_data` as the data root. The
expected data include the processed motion data, object geometry, contact labels,
text annotations, and precomputed ViHOI VLM embeddings. The default scripts
refer to these VLM embedding folders:

```text
processed_data/250929-dual_vlm_hidden_3_padded
processed_data/250925-dual_vlm_hidden_12_padded
processed_data/251006-dual_gen_vlm_hidden_3_padded_test
processed_data/250930-dual_gen_vlm_hidden_12_padded_test
```


## Visual Prior Preprocessing

To save training resources, ViHOI does not run Qwen2.5-VL inside the diffusion
training loop. We first render reference HOI images, extract Qwen2.5-VL
visual/textual embeddings offline, pad the variable-length embeddings, and then
run the diffusion training/inference scripts with the precomputed embeddings.

### 1. Render Reference Images

Use the following script to render reference HOI images:

```text
tools/render_image.py
```

Run:

```bash
python tools/render_image.py
```

The current script reads processed motion/object data from `./processed_data`
and renders selected frames to:

```text
image_render/dataset/gt_image/train/image/view1
```

If you want to render another split, camera view, or output folder, edit the
hard-coded paths in `tools/render_image.py`. If rendering fails, also check the
Blender path/configuration in `manip/vis/blender_vis_mesh_motion.py`. The
rendered images are used as the image input for Qwen2.5-VL embedding extraction.

### 2. Extract Qwen2.5-VL Embeddings

Use the following script to extract visual/textual embeddings:

```bash
conda activate qwen
cd Qwen

python dovlm1.py \
  --project-name dual_vlm_hidden_3 \
  --data-mode train \
  --image-dir /path/to/train/rendered_images \
  --text-dir ../processed_data/omomo_text_anno_json_data \
  --output-dir ../processed_data \
  --select-layer 3 \
  --checkpoint-path Qwen/Qwen2.5-VL-3B-Instruct
```

After extracting raw `.pt` embeddings, pad them with:

```bash
python tools/padding.py 
```

## Training

ViHOI follows the CHOIS training pipeline, but additionally uses visual/textual
VLM priors.

```text
conda activate chois_env
scripts/train_vihoi.sh
```

## Inference

ViHOI inference follows the CHOIS single-window testing setting and generates
single-window human-object interaction sequences.

```text
conda activate chois_env
scripts/test_vihoi.sh
```

## Acknowledgements

We gratefully acknowledge the foundational work of these open-source projects
that enabled our research: [OMOMO](https://github.com/lijiaman/omomo_release?tab=readme-ov-file),
[CHOIS](https://github.com/lijiaman/chois_release?tab=readme-ov-file),
[MDM](https://github.com/GuyTevet/motion-diffusion-model) and
[Qwen](https://github.com/QwenLM/Qwen2.5-VL).


## Citation

If you find this repository useful, please cite our ViHOI paper.

```bibtex
@InProceedings{Cai_2026_CVPR,
    author    = {Cai, Songjin and Zhong, Linjie and Guo, Ling and Ding, Changxing},
    title     = {ViHOI: Human-Object Interaction Synthesis with Visual Priors},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2026},
    pages     = {30686-30695}
}
```
