#!/usr/bin/env python3
# -*- coding:utf-8 -*-


"""
@Project :gorge_walk
@File    :model.py
@Author  :kaiwu
@Date    :2022/11/15 20:57

"""

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from kaiwu_agent.utils.common_func import attached


class SharedBase(nn.Module):
    def __init__(
        self, hidden_dim, embedding_dim=8, maze_size=(64, 64), num_treasures=10
    ):
        super(SharedBase, self).__init__()

        # 嵌入层
        self.x_embedding = nn.Embedding(maze_size[0], embedding_dim)
        self.y_embedding = nn.Embedding(maze_size[1], embedding_dim)

        # 图像处理层 (5x5 输入)
        self.cnn_layer = nn.Sequential(
            nn.Conv2d(
                4, hidden_dim, kernel_size=3, padding=1
            ),  # 4通道输入, 输出 hidden_dim 通道
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 缩小特征图
            nn.Flatten(start_dim=-3),
        )
        # 图像特征处理后的 LayerNorm
        self.layernorm_cnn = nn.LayerNorm(
            hidden_dim * 2 * 2
        )  # 2x2 是卷积后特征图的大小

        # 处理其他特征
        self.fc_dist = nn.Linear(11, hidden_dim)  # end_dist (1) + treasure_dist (10)
        self.fc_treasure_status = nn.Linear(num_treasures, hidden_dim)

        # 其他特征处理后的 LayerNorm
        self.layernorm_dist = nn.LayerNorm(hidden_dim)
        self.layernorm_treasure_status = nn.LayerNorm(hidden_dim)

    def forward(self, input):
        (x, y, end_dist, treasure_dist, images, treasure_status) = input
        # 嵌入位置坐标
        x_emb = self.x_embedding(x).flatten(-2)
        y_emb = self.y_embedding(y).flatten(-2)
        position_emb = torch.cat((x_emb, y_emb), dim=-1)

        # 图像特征处理
        features = self.cnn_layer(images.float())
        features = self.layernorm_cnn(features)  # 应用 LayerNorm

        # 处理距离信息
        dist_info = torch.cat((end_dist, treasure_dist), dim=-1).float()
        dist_processed = F.relu(self.fc_dist(dist_info))
        dist_processed = self.layernorm_dist(dist_processed)  # 应用 LayerNorm

        # 处理宝箱状态
        treasure_status_processed = F.relu(self.fc_treasure_status(treasure_status))
        treasure_status_processed = self.layernorm_treasure_status(
            treasure_status_processed
        )  # 应用 LayerNorm

        # 融合特征
        combined_features = torch.cat(
            (position_emb, features, dist_processed, treasure_status_processed), dim=-1
        )

        return combined_features


class PPOFeatureExtractor(nn.Module):
    def __init__(
        self,
        hidden_dim_base=16,
        hidden_dim_ac=128,
        embedding_dim=8,
        action_dim=4,
        maze_size=(64, 64),
        num_treasures=10,
    ):
        super(PPOFeatureExtractor, self).__init__()  # 调用父类的__init__方法
        self.shared_base = SharedBase(
            hidden_dim=hidden_dim_base,
            embedding_dim=8,
            maze_size=maze_size,
            num_treasures=num_treasures,
        )

        self.actor = nn.Sequential(
            nn.Linear(6 * hidden_dim_base + 2 * embedding_dim, hidden_dim_ac),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dim_ac),  # 添加LayerNorm
            nn.Linear(hidden_dim_ac, action_dim),
        )
        self.critic = nn.Sequential(
            nn.Linear(6 * hidden_dim_base + 2 * embedding_dim, hidden_dim_ac),
            nn.LeakyReLU(),
            nn.LayerNorm(hidden_dim_ac),  # 添加LayerNorm
            nn.Linear(hidden_dim_ac, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="leaky_relu")
                nn.init.constant_(m.bias, 0)

    def forward_actor(self, features_dict):
        features = features_dict["features"]
        masks = features_dict["mask"]
        latent = self.shared_base(features)
        probs = self.actor(latent)
        if torch.isnan(probs).any():
            print(features, latent, probs)
        probs = probs.masked_fill(masks, -1e9)  # 使用mask
        probs = F.softmax(probs, dim=-1)
        return probs
        pass

    def forward(self, features_dict):
        features = features_dict["features"]
        masks = features_dict["mask"]
        latent = self.shared_base(features)
        probs = self.actor(latent)
        probs = probs.masked_fill(masks, -1e9)  # 使用mask
        probs = F.softmax(probs, dim=-1)
        values = self.critic(latent)
        return probs, values
