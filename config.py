#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :gorge_walk
@File    :config.py
@Author  :kaiwu
@Date    :2022/12/15 22:50

"""


# Configuration of dimensions
# 关于维度的配置
class Config:
    channels = 32
    # state_dim = 151
    hidden_dim = 151
    action_dim = 4
    learning_rate = 1e-4
    weight_decay = 1e-4
    vf_coef = 1
    ent_coef = 2
    target_kl = 0.05
    max_grad_norm = 0.5
    clip_range = 0.4
    n_epochs = 10
    batch_size = 256
    debug = False
    normalize_advantage = False
    # workflow Config
    max_step = 800
    treasure_num = 5
    EPISODES = 10000
    buffer_size = 12000
    gamma = 0.95
    gae_lambda = 0.95
    repeat_prob = 0.5
    # dimensionality of the sample
    # 样本维度
    SAMPLE_DIM = 5
    # Dimension of movement action direction
    # 移动动作方向的维度
    OBSERVATION_SHAPE = 214

    # The following configurations can be ignored
    # 以下是可以忽略的配置
    LEGAL_ACTION_SHAPE = 0
    SUB_ACTION_MASK_SHAPE = 0
    LSTM_HIDDEN_SHAPE = 0
    LSTM_CELL_SHAPE = 0

    DIM_OF_ACTION = 4
