import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.bool), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1) # b x ls x ls
    
    return subsequent_mask


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v

        self.w_q = nn.Linear(d_model, n_head*d_k)
        self.w_k = nn.Linear(d_model, n_head*d_k)
        self.w_v = nn.Linear(d_model, n_head*d_v)
        nn.init.normal_(self.w_q.weight, mean=0, std=np.sqrt(2.0/(d_model+d_k)))
        nn.init.normal_(self.w_k.weight, mean=0, std=np.sqrt(2.0/(d_model+d_k)))
        nn.init.normal_(self.w_v.weight, mean=0, std=np.sqrt(2.0/(d_model+d_v)))

        self.temperature = np.power(d_k, 0.5)
        self.attn_dropout = nn.Dropout(0.1)

        self.fc = nn.Linear(n_head*d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.layer_norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, q, k, v, mask=None):
        # q: BS X T X D, k: BS X T X D, v: BS X T X D, mask: BS X T X T 
        bs, n_q, _ = q.shape
        bs, n_k, _ = k.shape
        bs, n_v, _ = v.shape

        assert n_k == n_v

        residual = q

        q = self.w_q(q).view(bs, n_q, self.n_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_q, self.d_k)
        k = self.w_k(k).view(bs, n_k, self.n_head, self.d_k).permute(2, 0, 1, 3).contiguous().view(-1, n_k, self.d_k)
        v = self.w_v(v).view(bs, n_v, self.n_head, self.d_v).permute(2, 0, 1, 3).contiguous().view(-1, n_v, self.d_v)

        attn = torch.bmm(q, k.transpose(1, 2)) # (n_head*bs) X n_q X n_k
        attn = attn / self.temperature

        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1) # (n_head*bs) x n_q x n_k 
            attn = attn.masked_fill(mask, -np.inf)

        attn = F.softmax(attn, dim=2) # (n_head*bs) X n_q X n_k
        
        attn = self.attn_dropout(attn)
        output = torch.bmm(attn, v) # (n_head*bs) X n_q X d_v

        output = output.view(self.n_head, bs, n_q, self.d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(bs, n_q, -1)
        # BS X n_q X (n_head*D)

        # output = self.fc(output) # BS X n_q X D
        output = self.dropout(self.fc(output)) # BS X n_q X D
        output = self.layer_norm(output + residual) # BS X n_q X D

        return output, attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_in, d_hid):
        super(PositionwiseFeedForward, self).__init__()

        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: BS X N X D
        residual = x
        output = x.transpose(1, 2) # BS X D X N
        output = self.w_2(F.relu(self.w_1(output))) # BS X D X N
        output = output.transpose(1, 2) # BS X N X D
        output = self.dropout(output)
        output = self.layer_norm(output + residual) # BS X N X D

        return output


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v):
        super(DecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_model)

    def forward(self, decoder_input, self_attn_time_mask, self_attn_padding_mask):
        # decode_input: BS X T X D
        # time_mask: BS X T X T (padding postion are ones)
        # padding_mask: BS X T (padding position are zeros, diff usage from above)
        bs, dec_len, dec_hidden = decoder_input.shape
        
        decoder_out, dec_self_attn = self.self_attn(decoder_input, decoder_input, decoder_input, \
                                mask=self_attn_time_mask)
        # BS X T X D, BS X T X T
        decoder_out *= self_attn_padding_mask.unsqueeze(-1).float()
        # BS X T X D

        decoder_out = self.pos_ffn(decoder_out) # BS X T X D
        decoder_out *= self_attn_padding_mask.unsqueeze(-1).float()

        return decoder_out, dec_self_attn
        # BS X T X D, BS X T X T


class Decoder(nn.Module):
    def __init__(
            self,
            d_feats, d_model,
            n_layers, n_head, d_k, d_v, max_timesteps, use_full_attention=False):
        super(Decoder, self).__init__()

        self.start_conv = nn.Conv1d(d_feats, d_model, 1) # (input: 17*3)
        self.position_vec = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(max_timesteps + 2, d_model, padding_idx=0), # Increased size to avoid out-of-bounds for extra tokens
            freeze=True)
        self.layer_stack = nn.ModuleList([DecoderLayer(d_model, n_head, d_k, d_v)
            for _ in range(n_layers)])

        self.use_full_attention = use_full_attention 

    def forward(self, decoder_input, padding_mask, decoder_pos_vec, obj_embedding=None, obj_embedding_text=None, motion_sequence_mask=None):
        # decoder_input: BS X D X T 
        # padding_mask: BS X 1 X T
        # decoder_pos_vec: BS X 1 X T
        # obj_embedding: BS X 1 X D
        # vlm_embedding: BS X 1 X D

        dec_self_attn_list = []

        padding_mask = padding_mask.squeeze(1) # BS X T
        decoder_pos_vec = decoder_pos_vec.squeeze(1) # BS X T

        input_embedding = self.start_conv(decoder_input)  # BS X d_model X T
        input_embedding = input_embedding.transpose(1, 2) # BS X T X d_model
        
        all_embeddings = []
        if obj_embedding is not None:
            # all_embeddings.append(obj_embedding_text) # only textual
            all_embeddings.append(obj_embedding)
        if obj_embedding_text is not None:
            all_embeddings.append(obj_embedding_text)
        all_embeddings.append(input_embedding)

        new_input_embedding = torch.cat(all_embeddings, dim=1)

        # self.position_vec = self.position_vec.cuda()
        pos_embedding = self.position_vec(decoder_pos_vec) # BS X T+num_extra_tokens X D
       
        # Time mask is same for all blocks, while padding mask differ according to the position of block
        if self.use_full_attention:
            time_mask = None
        else:
            time_mask = get_subsequent_mask(decoder_pos_vec) 
        # BS X T X T (Prev steps are 0, later 1)
       
        dec_output = new_input_embedding + pos_embedding # BS X T+1 X D
        for dec_layer in self.layer_stack:
            dec_output, dec_self_attn = dec_layer(
                dec_output, # BS X T X D
                self_attn_time_mask=time_mask, # BS X T X T
                self_attn_padding_mask=padding_mask) # BS X T

            dec_self_attn_list += [dec_self_attn]

        return dec_output, dec_self_attn_list
        # BS X T X D, list



        # dec_self_attn_list = []
        # vlm_embedding_list = [obj_embedding_text, obj_embedding]

        # motion_sequence_mask = motion_sequence_mask.squeeze(1) # BS X T

        # padding_mask = padding_mask.squeeze(1) # BS X T
        # decoder_pos_vec = decoder_pos_vec.squeeze(1) # BS X T

        # input_embedding = self.start_conv(decoder_input)  # BS X d_model X T
        # input_embedding = input_embedding.transpose(1, 2) # BS X T X d_model
        
        # # Use staged processing if vlm_embedding_list is provided, otherwise fallback to original method
        # if vlm_embedding_list is not None:
        #     # New staged processing approach
            
        #     # Stage 1: 前两层 - 动作序列 + 文本条件
        #     stage1_embeddings = []
            
        #     # Add text condition for first 2 layers
        #     if len(vlm_embedding_list) > 0 and vlm_embedding_list[0] is not None and vlm_embedding_list[1] is not None:
        #         stage1_embeddings.append(vlm_embedding_list[0])  # text condition: BS X 1 X D
        #         stage1_embeddings.append(vlm_embedding_list[1])  # image condition: BS X 1 X D
        #     stage1_embeddings.append(input_embedding)  # motion sequence: BS X T X D  

        #     stage1_input = torch.cat(stage1_embeddings, dim=1)  # BS X (T+num_conditions) X D  BS X (T+2) X D  
            
        #     # Position encoding for stage 1
        #     stage1_seq_len = stage1_input.shape[1] # T+2
        #     stage1_pos_vec = torch.arange(stage1_seq_len, device=decoder_pos_vec.device) + 1  # +1 like original
        #     stage1_pos_vec = stage1_pos_vec[None, None, :].repeat(input_embedding.shape[0], 1, 1)  # BS X 1 X stage1_seq_len
        #     stage1_pos_vec = stage1_pos_vec.squeeze(1)  # BS X stage1_seq_len
        #     stage1_pos_embedding = self.position_vec(stage1_pos_vec)  # BS X (T+num_conditions) X D  BS X (T+2) X D  
            
        #     # Padding mask for stage 1 (add padding for condition tokens)
        #     num_conditions = stage1_seq_len - input_embedding.shape[1] # 2
        #     stage1_condition_mask = torch.ones(padding_mask.shape[0], num_conditions, device=padding_mask.device, dtype=padding_mask.dtype) # BS X 2
        #     stage1_padding_mask = torch.cat([stage1_condition_mask, motion_sequence_mask], dim=1) # BS X (T+2)

        #     # Time mask for stage 1
        #     if self.use_full_attention:
        #         stage1_time_mask = None
        #     else:
        #         stage1_time_mask = get_subsequent_mask(stage1_pos_vec)
        #     # Create custom mask to block the second token (index 1) from influencing motion tokens
        #     # Create a mask that only blocks the second token (index 1)
        #     stage1_condition_mask = torch.zeros(stage1_pos_vec.shape[0], stage1_seq_len, stage1_seq_len, 
        #                                  device=stage1_pos_vec.device, dtype=torch.bool)
        #     # Mask out the second token (index 1) by setting its column to True
        #     # This prevents all tokens from attending to the second token
        #     stage1_condition_mask[:, :, 1] = True
        #     # Stage 1 processing: first 2 layers
        #     stage1_output = stage1_input + stage1_pos_embedding
        #     for layer_idx in range(min(2, len(self.layer_stack))):
        #         dec_layer = self.layer_stack[layer_idx]
        #         stage1_output, dec_self_attn = dec_layer(
        #             stage1_output,
        #             # self_attn_time_mask=stage1_time_mask, # BS X T+2 X T+2
        #             self_attn_time_mask=stage1_condition_mask,
        #             self_attn_padding_mask=stage1_padding_mask)
        #         dec_self_attn_list.append(dec_self_attn)
            
        #     # Stage 2: 后两层 - 前2层输出 + 图像条件
        #     if len(self.layer_stack) > 2:
        #         stage2_embeddings = []
                
        #         # Add image condition for last 2 layers  
        #         # if len(vlm_embedding_list) > 1 and vlm_embedding_list[1] is not None:
        #         #     pos_embedding_for_image = stage1_pos_embedding[:, :num_conditions, :] # 提取stage1对条件的位置信息
        #         #     vlm_embedding_list[1] += pos_embedding_for_image
        #         #     stage2_embeddings.append(vlm_embedding_list[1])  # image condition: BS X 1 X D
        #         # stage1_output_motion = stage1_output[:, num_conditions:, :]
        #         # stage2_embeddings.append(stage1_output_motion)  # stage1 output: BS X (T+stage1_conditions) X D
                
        #         # stage2_input = torch.cat(stage2_embeddings, dim=1)  # BS X (T+stage2_conditions) X D 
                
        #         # # Position encoding for stage 2
        #         # # stage2_seq_len = stage2_input.shape[1] 
        #         # # stage2_pos_vec = torch.arange(stage2_seq_len, device=decoder_pos_vec.device) + 1  # +1 like original
        #         # # stage2_pos_vec = stage2_pos_vec[None, None, :].repeat(input_embedding.shape[0], 1, 1)  # BS X 1 X stage2_seq_len
        #         # # stage2_pos_vec = stage2_pos_vec.squeeze(1)  # BS X stage2_seq_len
        #         # # stage2_pos_embedding = self.position_vec(stage2_pos_vec)  # BS X (T+all_conditions) X D
                
        #         # # # Padding mask for stage 2 (add padding for new condition tokens)
        #         # # # num_stage2_conditions = stage2_seq_len - stage1_output.shape[1]
        #         # stage2_condition_mask = torch.ones(padding_mask.shape[0], num_conditions, device=padding_mask.device, dtype=padding_mask.dtype)
        #         # # stage2_condition_mask[:, 0] = 0
        #         # # stage2_padding_mask = torch.cat([stage2_condition_mask, stage1_padding_mask], dim=1)
        #         # stage2_padding_mask = torch.cat([stage2_condition_mask, motion_sequence_mask], dim=1)

        #         # # # Time mask for stage 2
        #         # # if self.use_full_attention:
        #         # #     stage2_time_mask = None
        #         # # else:
        #         # #     stage2_time_mask = get_subsequent_mask(stage2_pos_vec)
                    
        #         # # Stage 2 processing: last 2 layers
        #         stage2_output = stage1_output
        #         for layer_idx in range(2, len(self.layer_stack)):
        #             dec_layer = self.layer_stack[layer_idx]
        #             stage2_output, dec_self_attn = dec_layer(
        #                 stage2_output,
        #                 self_attn_time_mask=stage1_time_mask,
        #                 self_attn_padding_mask=stage1_padding_mask)
        #             dec_self_attn_list.append(dec_self_attn)
                    
        #         final_output = stage2_output
        #         return final_output, dec_self_attn_list
        #     else:
        #         final_output = stage1_output
        #         return final_output, dec_self_attn_list