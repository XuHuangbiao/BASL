import torch
import torch.nn as nn
import numpy as np
from models.transformer import TransformerEncoderLayer, TransformerEncoder
from .i3d import I3D


class SingleI3DExtractor(nn.Module):
    """
    独立的I3D特征提取器，所有网络共享
    为了显存优化，I3D作为独立模块
    """

    def __init__(self, pretrained_weight=None, dataset='MTL-AQA'):
        super().__init__()
        self.backbone = I3D(num_classes=400, modality='rgb', dropout_prob=0.5)
        self.dataset = dataset
        if pretrained_weight is not None:
            checkpoint = torch.load(pretrained_weight)
            self.backbone.load_state_dict(checkpoint)
            print(f"✓ I3D预训练权重加载成功: {pretrained_weight}")

    def forward(self, video):
        if self.dataset == 'MTL-AQA':
            start_idx = [0, 10, 20, 30, 40, 50, 60, 70, 80, 86]
        elif self.dataset == 'JIG':
            start_idx = list(range(0, 160, 16))
        else:
            start_idx = list(range(0, 90, 10))
        video_pack = torch.cat([video[:, :, i: i + 16] for i in start_idx])
        feat = self.backbone(video_pack)
        feat = feat.reshape(len(start_idx), len(video), -1).transpose(0, 1)
        return feat


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0., max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


def temporal_random_mask(x, mask_ratio=0.15):
    """弱增强：随机时间步 mask"""
    B, T, D = x.shape
    num_mask = max(1, int(T * mask_ratio))
    out = x.clone()
    for b in range(B):
        mask_pos = np.random.choice(T, num_mask, replace=False)
        out[b, mask_pos, :] = 0
    return out


def temporal_gaussian_noise(x, std=0.05):
    """强增强：高斯噪声"""
    return x + torch.randn_like(x) * std


class BASLCore(nn.Module):
    """
    DASL 核心网络（Transformer + 回归头）
    """

    def __init__(self, in_dim, hidden_dim, n_head, n_encoder, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.pos_enc = PositionalEncoding(hidden_dim, dropout=dropout)

        enc_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_head,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.encoder = TransformerEncoder(enc_layer, num_layers=n_encoder)

        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, feat):
        x = self.in_proj(feat)          # (B, T, hidden_dim)
        x = self.pos_enc(x)             # 位置编码
        enc_feat = self.encoder(x)      # (B, T, hidden_dim)

        pooled = self.temporal_pool(enc_feat.transpose(1, 2)).squeeze(-1)  # (B, hidden_dim)
        score = self.score_head(pooled).squeeze(-1)                        # (B,)

        return score, pooled


class BASL(nn.Module):
    """
    单一网络：Transformer + 投影头 + 回归头
    支持原始/强弱增强三路输出：
      - output / score_original：原始视图（用于监督与伪标签回归）
      - score_weak/score_strong：一致性学习
    """

    def __init__(self, in_dim, hidden_dim, n_head, n_encoder, dropout, config):
        super().__init__()
        self.config = config

        self.use_i3d = getattr(config, 'dataset', 'RG') in ['MTL-AQA', 'FineDiving', 'JIG']
        if self.use_i3d:
            self.i3d_extractor = SingleI3DExtractor(
                pretrained_weight=getattr(config, 'pretrained_i3d_weight', None),
                dataset=config.dataset
            )
            in_dim = 1024
        else:
            self.i3d_extractor = None

        self.core = BASLCore(in_dim, hidden_dim, n_head, n_encoder, dropout)

        self.proj_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2)
        )

    def extract_features(self, video):
        if self.use_i3d:
            return self.i3d_extractor(video)
        return video

    def forward(self, video, mode='train', return_aug=False):
        feat = self.extract_features(video)

        if return_aug:
            mask_ratio = getattr(self.config, 'mask_ratio', 0.15)
            noise_std = getattr(self.config, 'eps', 0.05)

            # 原始 + 弱增强 + 强增强
            x_orig = feat
            x_weak = temporal_random_mask(feat, mask_ratio=mask_ratio)
            x_strong = temporal_gaussian_noise(feat, std=noise_std)

            score_o, pooled_o = self.core(x_orig)
            score_w, pooled_w = self.core(x_weak)
            score_s, pooled_s = self.core(x_strong)

            proj_w = self.proj_head(pooled_w)
            proj_s = self.proj_head(pooled_s)

            return {
                "output": score_o,             # 用原始视图作为主输出（监督/伪标签）
                "score_original": score_o,
                "score_weak": score_w,
                "score_strong": score_s,
                "feat_weak": proj_w,
                "feat_strong": proj_s
            }

        # 标准推理：原始视图
        score, _ = self.core(feat)
        return {"output": score}

    def get_trainable_parameters(self):
        return list(self.parameters())