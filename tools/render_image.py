import numpy as np
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trimesh
import torch
import joblib
import json
from human_body_prior.src.human_body_prior.body_model.body_model import BodyModel
from manip.vis.blender_vis_mesh_motion import run_blender_rendering_and_save2video, save_verts_faces_to_mesh_file_w_object
from manip.data.cano_traj_dataset import get_smpl_parents
from manip.lafan1.utils import rotate_at_frame_w_obj
import pytorch3d.transforms as p_transforms
import shutil

def run_smplx_model(root_trans, aa_rot_rep, betas, gender, bm_dict, return_joints24=True):
    # root_trans: BS X T X 3
    # aa_rot_rep: BS X T X 22 X 3 
    # betas: BS X 16
    # gender: BS 
    bs, num_steps, num_joints, _ = aa_rot_rep.shape
    if num_joints != 52:
        padding_zeros_hand = torch.zeros(bs, num_steps, 30, 3).to(aa_rot_rep.device) # BS X T X 30 X 3 
        aa_rot_rep = torch.cat((aa_rot_rep, padding_zeros_hand), dim=2) # BS X T X 52 X 3 

    aa_rot_rep = aa_rot_rep.reshape(bs*num_steps, -1, 3) # (BS*T) X n_joints X 3 
    betas = betas[:, None, :].repeat(1, num_steps, 1).reshape(bs*num_steps, -1) # (BS*T) X 16 
    gender = np.asarray(gender)[:, np.newaxis].repeat(num_steps, axis=1)
    gender = gender.reshape(-1).tolist() # (BS*T)

    smpl_trans = root_trans.reshape(-1, 3) # (BS*T) X 3  
    smpl_betas = betas # (BS*T) X 16
    smpl_root_orient = aa_rot_rep[:, 0, :] # (BS*T) X 3 
    smpl_pose_body = aa_rot_rep[:, 1:22, :].reshape(-1, 63) # (BS*T) X 63
    smpl_pose_hand = aa_rot_rep[:, 22:, :].reshape(-1, 90) # (BS*T) X 90 

    B = smpl_trans.shape[0] # (BS*T) 

    smpl_vals = [smpl_trans, smpl_root_orient, smpl_betas, smpl_pose_body, smpl_pose_hand]
    # batch may be a mix of genders, so need to carefully use the corresponding SMPL body model
    gender_names = ['male', 'female', "neutral"]
    pred_joints = []
    pred_verts = []
    prev_nbidx = 0
    cat_idx_map = np.ones((B), dtype=np.int64)*-1
    for gender_name in gender_names:
        gender_idx = np.array(gender) == gender_name
        nbidx = np.sum(gender_idx)

        cat_idx_map[gender_idx] = np.arange(prev_nbidx, prev_nbidx + nbidx, dtype=np.int64)
        prev_nbidx += nbidx

        gender_smpl_vals = [val[gender_idx] for val in smpl_vals]

        if nbidx == 0:
            # skip if no frames for this gender
            continue
        
        # reconstruct SMPL
        cur_pred_trans, cur_pred_orient, cur_betas, cur_pred_pose, cur_pred_pose_hand = gender_smpl_vals
        bm = bm_dict[gender_name]

        pred_body = bm(pose_body=cur_pred_pose, pose_hand=cur_pred_pose_hand, \
                betas=cur_betas, root_orient=cur_pred_orient, trans=cur_pred_trans)
        
        pred_joints.append(pred_body.Jtr)
        pred_verts.append(pred_body.v)

    # cat all genders and reorder to original batch ordering
    if return_joints24:
        x_pred_smpl_joints_all = torch.cat(pred_joints, axis=0) # () X 52 X 3 
        lmiddle_index= 28 
        rmiddle_index = 43 
        x_pred_smpl_joints = torch.cat((x_pred_smpl_joints_all[:, :22, :], \
            x_pred_smpl_joints_all[:, lmiddle_index:lmiddle_index+1, :], \
            x_pred_smpl_joints_all[:, rmiddle_index:rmiddle_index+1, :]), dim=1) 
    else:
        x_pred_smpl_joints = torch.cat(pred_joints, axis=0)[:, :num_joints, :]
        
    x_pred_smpl_joints = x_pred_smpl_joints[cat_idx_map] # (BS*T) X 22 X 3 

    x_pred_smpl_verts = torch.cat(pred_verts, axis=0)
    x_pred_smpl_verts = x_pred_smpl_verts[cat_idx_map] # (BS*T) X 6890 X 3 

    
    x_pred_smpl_joints = x_pred_smpl_joints.reshape(bs, num_steps, -1, 3) # BS X T X 22 X 3/BS X T X 24 X 3  
    x_pred_smpl_verts = x_pred_smpl_verts.reshape(bs, num_steps, -1, 3) # BS X T X 6890 X 3 

    mesh_faces = pred_body.f 
    
    return x_pred_smpl_joints, x_pred_smpl_verts, mesh_faces 


def save_verts_faces_to_mesh_file_w_object(mesh_verts, mesh_faces, obj_verts, obj_faces, save_mesh_folder):
    # mesh_verts: T X Nv X 3
    # mesh_faces: Nf X 3
    if not os.path.exists(save_mesh_folder):
        os.makedirs(save_mesh_folder)

    num_meshes = mesh_verts.shape[0]
    for idx in range(num_meshes):
        mesh = trimesh.Trimesh(vertices=mesh_verts[idx],
                        faces=mesh_faces)
        curr_mesh_path = os.path.join(save_mesh_folder, "%05d"%(idx)+".ply")
        mesh.export(curr_mesh_path)

        obj_mesh = trimesh.Trimesh(vertices=obj_verts[idx],
                        faces=obj_faces)
        curr_obj_mesh_path = os.path.join(save_mesh_folder, "%05d"%(idx)+"_object.ply")
        obj_mesh.export(curr_obj_mesh_path)


def load_rest_pose_object_geometry(object_name):
    rest_obj_path = os.path.join('./processed_data/rest_object_geo', object_name+".ply")
    
    mesh = trimesh.load_mesh(rest_obj_path)
    rest_verts = np.asarray(mesh.vertices) # Nv X 3
    obj_mesh_faces = np.asarray(mesh.faces) # Nf X 3

    return rest_verts, obj_mesh_faces 

def load_object_geometry_w_rest_geo(obj_rot, obj_com_pos, rest_verts):
    # obj_scale: T, obj_rot: T X 3 X 3, obj_com_pos: T X 3, rest_verts: Nv X 3 
    rest_verts = rest_verts.to(obj_rot.device)
    rest_verts = rest_verts[None].repeat(obj_rot.shape[0], 1, 1)
    transformed_obj_verts = obj_rot.bmm(rest_verts.transpose(1, 2)) + obj_com_pos[:, :, None]
    transformed_obj_verts = transformed_obj_verts.transpose(1, 2) # T X Nv X 3 

    return transformed_obj_verts 

if __name__ == "__main__":
        
    data_root_folder = "./processed_data" 
    surface_model_male_fname = os.path.join('./processed_data/smpl_all_models/smplx', "SMPLX_MALE.npz")
    surface_model_female_fname = os.path.join('./processed_data/smpl_all_models/smplx', "SMPLX_FEMALE.npz")
            
    male_bm = BodyModel(bm_fname=surface_model_male_fname,
                    num_betas=16,
                    num_expressions=None,
                    num_dmpls=None,
                    dmpl_fname=None).cuda()
    female_bm = BodyModel(bm_fname=surface_model_female_fname,
                    num_betas=16,
                    num_expressions=None,
                    num_dmpls=None,
                    dmpl_fname=None).cuda()
    bm_dict = {'male' : male_bm, 'female' : female_bm}           

    # 构建文件路径
    processed_data_path = os.path.join(data_root_folder,
    "test_diffusion_manip_seq_joints24.p")  
    data_dict = joblib.load(processed_data_path)
    for i in range(len(data_dict)):    
        # Prepare list used for evaluation. 
        human_jnts_list = []
        human_verts_list = [] 
        obj_verts_list = [] 
        trans_list = []
        human_mesh_faces_list = []
        obj_mesh_faces_list = [] 


        # 1. Load all raw data
        seq_name = data_dict[i]['seq_name']
        object_name = seq_name.split('_')[1]
        if object_name in ["vacuum", "mop"]:
            continue
        
        seq_root_trans = data_dict[i]['trans'] # T X 3
        seq_root_orient = data_dict[i]['root_orient'] # T X 3
        seq_pose_body = data_dict[i]['pose_body'].reshape(-1, 21, 3) # T X 21 X 3
        
        obj_com_pos = data_dict[i]['obj_com_pos'] # T X 3
        ori_obj_rot = data_dict[i]['obj_rot'] # T X 3 X 3
        
        trans2joint = data_dict[i]['trans2joint'] # 3
        rest_human_offsets = data_dict[i]['rest_offsets'] # 24 X 3, assuming joints24
        betas_np = data_dict[i]['betas']
        gender = str(data_dict[i]['gender'])

        # 2. Replicate cano_traj_dataset.py:cal_normalize_data_input
        # Human motion to quaternions
        joint_aa_rep = torch.cat((torch.from_numpy(seq_root_orient).float()[:, None, :], \
            torch.from_numpy(seq_pose_body).float()), dim=1) # T X 22 X 3
        local_rot_mat = p_transforms.axis_angle_to_matrix(joint_aa_rep) # T X 22 X 3 X 3
        Q = p_transforms.matrix_to_quaternion(local_rot_mat).detach().cpu().numpy() # T X 22 X 4

        # Object motion to quaternions (using canonicalized rotation)
        rest_obj_json_path = os.path.join(data_root_folder, "rest_object_geo", object_name + ".json")
        json_data = json.load(open(rest_obj_json_path, 'r'))
        rest_pose_rot_mat = np.asarray(json_data['rest_pose_ori_obj_rot']) # 3 X 3
        ori_obj_rot_tensor = torch.from_numpy(ori_obj_rot).float()
        rest_pose_rot_mat_tensor = torch.from_numpy(rest_pose_rot_mat).float()[None, :, :]
        obj_rot_mat = torch.matmul(
            ori_obj_rot_tensor, 
            rest_pose_rot_mat_tensor.repeat(ori_obj_rot_tensor.shape[0], 1, 1).transpose(1, 2)
        ) # T X 3 X 3
        obj_q = p_transforms.matrix_to_quaternion(obj_rot_mat).detach().cpu().numpy() # T X 4

        # Human positions X
        X = torch.from_numpy(rest_human_offsets).float()[:22][None].repeat(joint_aa_rep.shape[0], 1, 1).detach().cpu().numpy() # T X 22 X 3
        X[:, 0, :] = seq_root_trans
        
        # Get smpl parents
        parents = get_smpl_parents(use_joints24=False)

        # Call canonicalization function
        cano_X, cano_Q, cano_obj_x, cano_obj_q = rotate_at_frame_w_obj(
            X[np.newaxis], Q[np.newaxis], obj_com_pos[np.newaxis], obj_q[np.newaxis],
            trans2joint[np.newaxis], parents, n_past=1, floor_z=True
        )

        # 3. Replicate cano_traj_dataset.py:process_window_data
        # Get canonicalized human root position and calculate offset
        cano_human_root_trans = torch.from_numpy(cano_X[0, :, 0, :]).float() # T X 3
        move_to_zero_trans = cano_human_root_trans[0:1, :].clone() # 1 X 3
        move_to_zero_trans[:, 2] = 0

        # Apply offset to get final human and object translations
        root_trans = (cano_human_root_trans - move_to_zero_trans).cuda()
        curr_gt_obj_com_pos = (torch.from_numpy(cano_obj_x[0]).float() - move_to_zero_trans).cuda()

        # Get final rotations
        final_local_rot_mat = p_transforms.quaternion_to_matrix(torch.from_numpy(cano_Q[0]).float()) # T X 22 X 3 X 3
        curr_local_rot_aa_rep = p_transforms.matrix_to_axis_angle(final_local_rot_mat).cuda() # T X 22 X 3
        curr_gt_obj_rot_mat = p_transforms.quaternion_to_matrix(torch.from_numpy(cano_obj_q[0]).float()).cuda() # T X 3 X 3
        
        # Prepare other data
        bs = 1
        betas = torch.from_numpy(betas_np).float().cuda()
        
        # Get human verts 
        mesh_jnts, mesh_verts, mesh_faces = \
            run_smplx_model(root_trans[None], curr_local_rot_aa_rep[None], \
            betas, [gender], bm_dict, return_joints24=True)
        # 可视化参数 root_trans  curr_local_rot_aa_rep


        # Get object verts 
        obj_rest_verts, obj_mesh_faces = load_rest_pose_object_geometry(object_name)
        obj_rest_verts = torch.from_numpy(obj_rest_verts)

        gt_obj_mesh_verts = load_object_geometry_w_rest_geo(curr_gt_obj_rot_mat, \
                    curr_gt_obj_com_pos, obj_rest_verts.float())
        # 可视化参数 curr_gt_obj_rot_mat  curr_gt_obj_com_pos


        actual_len = seq_root_trans.shape[0]

        # Frame selection: 2 uniform, 1 from last contact change
        if actual_len < 3:
            contact_frames = list(range(actual_len))
        else:
            # 1. Get two uniformly sampled frames (0-indexed)
            uniform_frames = [
                # 0 #取第一帧
                int(round(actual_len / 3.0)) - 1,
                int(round(actual_len * 2 / 3.0)) - 1
            ]

            # 2. Get the last contact change frame
            last_contact_frame = -1
            contact_label_path = os.path.join("./processed_data/contact_labels_w_semantics_npy_files", seq_name + ".npy")
            if os.path.exists(contact_label_path):
                contact_labels = np.load(contact_label_path)
                # Iterate backwards from the second to last frame
                for frame_idx in range(len(contact_labels) - 1, 0, -1):
                    if not np.array_equal(contact_labels[frame_idx], contact_labels[frame_idx - 1]):
                        last_contact_frame = frame_idx
                        break
            
            # If no change was found or file doesn't exist, use the last frame of the sequence.
            if last_contact_frame == -1:
                last_contact_frame = actual_len - 1
            
            # 3. Combine, sort, and remove duplicates
            final_frames = uniform_frames + [last_contact_frame]  #取第一帧
            contact_frames = sorted(list(set(final_frames)))
        
        human_jnts_list.append(mesh_jnts[0])
        human_verts_list.append(mesh_verts[0]) 
        obj_verts_list.append(gt_obj_mesh_verts)
        trans_list.append(root_trans) 

        human_mesh_faces_list.append(mesh_faces)
        obj_mesh_faces_list.append(obj_mesh_faces) 


        dest_mesh_vis_folder = './render'

        # 解析seq_name，获取子目录和文件名
        seq_name = data_dict[i]['seq_name']  # 例如 'sub10_clothesstand_000'
        sub_dir, file_name = seq_name.split('_', 1)  # sub10, clothesstand_000

        mesh_save_folder = os.path.join('./render/dataset/mesh', sub_dir, file_name)
        out_rendered_img_folder = os.path.join('./render/dataset/test/gt_image/view2', sub_dir, file_name)
        
        # Save only the contact frames to mesh files
        if not os.path.exists(mesh_save_folder):
            os.makedirs(mesh_save_folder)
            
        for frame_idx in contact_frames:
            if frame_idx < actual_len:
                # Save human mesh
                human_mesh = trimesh.Trimesh(vertices=mesh_verts.detach().cpu().numpy()[0][frame_idx], faces=mesh_faces.detach().cpu().numpy())
                human_mesh_path = os.path.join(mesh_save_folder, f"{frame_idx:05d}.ply")
                human_mesh.export(human_mesh_path)
                
                # Save object mesh
                obj_mesh = trimesh.Trimesh(vertices=gt_obj_mesh_verts.detach().cpu().numpy()[frame_idx], faces=obj_mesh_faces)
                obj_mesh_path = os.path.join(mesh_save_folder, f"{frame_idx:05d}_object.ply")
                obj_mesh.export(obj_mesh_path)

        #view original
        # floor_blend_path = './processed_data/blender_files/floor_colorful_mat.blend'
        #view 1
        floor_blend_path = './processed_data/floor_colorful_mat_view2.blend'

        camera_pos = (5, -5, 3)
        camera_rot = (75, 0, 45)
        run_blender_rendering_and_save2video(mesh_save_folder, out_rendered_img_folder, out_vid_path=None, \
            vis_object=True, vis_condition=False, scene_blend_path=floor_blend_path, contact_frames=contact_frames)

        # Clean up the temporary mesh files
        if os.path.exists(mesh_save_folder):
            shutil.rmtree(mesh_save_folder)