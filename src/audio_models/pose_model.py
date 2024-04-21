import os
import math
import torch
import torch.nn as nn
from transformers import Wav2Vec2Config

from .torch_utils import get_mask_from_lengths
from .wav2vec2 import Wav2Vec2Model


def init_biased_mask(n_head, max_seq_len, period):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)                   
        else:                                                 
            closest_power_of_2 = 2**math.floor(math.log2(n)) 
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
    slopes = torch.Tensor(get_slopes(n_head))
    bias = torch.arange(start=0, end=max_seq_len, step=period).unsqueeze(1).repeat(1,period).view(-1)//(period)
    bias = - torch.flip(bias,dims=[0])
    alibi = torch.zeros(max_seq_len, max_seq_len)
    for i in range(max_seq_len):
        alibi[i, :i+1] = bias[-(i+1):]
    alibi = slopes.unsqueeze(1).unsqueeze(1) * alibi.unsqueeze(0)
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0) + alibi
    return mask


def enc_dec_mask(device, T, S):
    mask = torch.ones(T, S)
    for i in range(T):
        mask[i, i] = 0
    return (mask==1).to(device=device)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=600):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class Audio2PoseModel(nn.Module):
    def __init__(
        self,
        config
    ):  
        
        super().__init__()
        
        latent_dim = config['latent_dim']
        model_path = config['model_path']
        only_last_fetures = config['only_last_fetures']
        from_pretrained = config['from_pretrained']
        out_dim = config['out_dim']

        self.out_dim = out_dim

        self._only_last_features = only_last_fetures

        self.audio_encoder_config = Wav2Vec2Config.from_pretrained(model_path, local_files_only=True)
        if from_pretrained:
            self.audio_encoder = Wav2Vec2Model.from_pretrained(model_path, local_files_only=True)
        else:
            self.audio_encoder = Wav2Vec2Model(self.audio_encoder_config)
        self.audio_encoder.feature_extractor._freeze_parameters()

        hidden_size = self.audio_encoder_config.hidden_size

        self.pose_map = nn.Linear(out_dim, latent_dim)
        self.in_fn = nn.Linear(hidden_size, latent_dim)

        self.PPE = PositionalEncoding(latent_dim)
        self.biased_mask = init_biased_mask(n_head = 8, max_seq_len = 600, period=1)
        decoder_layer = nn.TransformerDecoderLayer(d_model=latent_dim, nhead=8, dim_feedforward=2*latent_dim, batch_first=True)        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=8)
        self.pose_map_r = nn.Linear(latent_dim, out_dim)

        self.id_embed = nn.Embedding(100, latent_dim) # 100 ids


    def infer(self, input_value, seq_len, id_seed=None):
        embeddings = self.audio_encoder(input_value, seq_len=seq_len, output_hidden_states=True)

        if self._only_last_features:
            hidden_states = embeddings.last_hidden_state
        else:
            hidden_states = sum(embeddings.hidden_states) / len(embeddings.hidden_states)

        hidden_states = self.in_fn(hidden_states)

        id_embedding = self.id_embed(id_seed).unsqueeze(1)
        
        init_pose = torch.zeros([hidden_states.shape[0], 1, self.out_dim]).to(hidden_states.device)
        for i in range(seq_len):
            if i==0:
                pose_emb = self.pose_map(init_pose)
                pose_input = self.PPE(pose_emb)
            else:
                pose_input = self.PPE(pose_emb)

            pose_input = pose_input + id_embedding
            tgt_mask = self.biased_mask[:, :pose_input.shape[1], :pose_input.shape[1]].clone().detach().to(hidden_states.device)
            memory_mask = enc_dec_mask(hidden_states.device,  pose_input.shape[1], hidden_states.shape[1])
            pose_out = self.transformer_decoder(pose_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            pose_out = self.pose_map_r(pose_out)
            new_output = self.pose_map(pose_out[:,-1,:]).unsqueeze(1)
            pose_emb = torch.cat((pose_emb, new_output), 1)
        return pose_out
    
