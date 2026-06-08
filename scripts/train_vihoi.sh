OMP_NUM_THREADS=8 \
CUDA_VISIBLE_DEVICES=7 \
WANDB_MODE="offline" \
python trainer_vihoi_dual.py \
--window=120 \
--batch_size=32 \
\
--data_root_folder="./processed_data" \
--project="./Vision_HOI" \
--exp_name="visual_3_textual_12" \
--wandb_pj_name="whole_model" \
--entity="songjin" \
\
--input_first_human_pose \
--use_random_frame_bps \
--use_object_keypoints \
--loss_w_feet=1 \
--loss_w_fk=0.5 \
--loss_w_obj_pts=1 \
\
--use_vlm_condition \
--vlm_embedding_dir="./processed_data/250929-dual_vlm_hidden_3_padded" \
--vlm_embedding_text_dir="./processed_data/250925-dual_vlm_hidden_12_padded" \
--vlm_projection_type="transformer" \
\
--learning_rate=1e-5 \
