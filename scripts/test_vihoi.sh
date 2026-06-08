CUDA_VISIBLE_DEVICES=1 \
OMP_NUM_THREADS=8 \
python trainer_vihoi_dual.py \
--window=120 \
--batch_size=32 \
--data_root_folder="./processed_data" \
--pretrained_model="./Vision_HOI/visual_3_textual_12/weights/model-34.pt" \
--save_res_folder="./vision_hoi_single_window_results/visual_3_textual_12" \
--input_first_human_pose \
--use_object_keypoints \
--add_semantic_contact_labels \
--use_random_frame_bps \
--loss_w_feet=1 \
--loss_w_fk=0.5 \
--loss_w_obj_pts=1 \
--test_sample_res \
--use_guidance_in_denoising \
--use_vlm_condition \
--vlm_embedding_dir="./processed_data/251006-dual_gen_vlm_hidden_3_padded" \
--vlm_embedding_text_dir="./processed_data/250930-dual_gen_vlm_hidden_12_padded" \
--vlm_projection_type="transformer" \
--compute_metrics
# \
