#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :gorge_walk
@File    :definition.py
@Author  :kaiwu
@Date    :2022/12/15 22:50

"""
import numpy as np
import torch

from kaiwu_agent.utils.common_func import create_cls, attached


SampleData = create_cls(
    "SampleData", state=None, action=None, reward=None, next_state=None
)

ObsData = create_cls("ObsData", feature=None)
#
ActData = create_cls("ActData", action=None, value=None, log_prob=None)


def generate_mask_vectorized(data):
    # 提取所有行的obstacle_flat并重塑为 (n, 5, 5)
    obstacle_map = data[:, 140:165].reshape(-1, 5, 5)
    # 选择特定位置的值并生成掩码
    mask = torch.stack(
        [
            obstacle_map[:, 2, 3] == 1,
            obstacle_map[:, 2, 1] == 1,
            obstacle_map[:, 1, 2] == 1,
            obstacle_map[:, 3, 2] == 1,
        ],
        dim=1,
    )
    return mask


def processFeature(obs: torch.tensor, device="cuda"):
    obs = obs.reshape(-1, 250).to(device)
    pos_x = obs[:, 0:1].int() // 64
    pos_y = obs[:, 0:1].int() % 64
    end_dist = obs[:, 129:130].float()
    treasure_dist = obs[:, 130:140].float()
    treasure_status = obs[:, -10:].float()
    feature_map = obs[:, 140:240].reshape(-1, 4, 5, 5).float()
    mask = generate_mask_vectorized(obs).to(device)
    return {
        "features": (
            pos_x,
            pos_y,
            end_dist,
            treasure_dist,
            feature_map,
            treasure_status,
        ),
        "mask": mask,
    }


@attached
def observation_process(raw_obs):
    return ObsData(feature=np.array(raw_obs))


@attached
def action_process(act_data):
    return act_data.action


def extractAVL(act_data):
    return act_data.action, act_data.value, act_data.log_prob


@attached
def sample_process(list_game_data):
    return [SampleData(**i.__dict__) for i in list_game_data]


def get_center_memory_average(memory_flat):
    # 将一维数组转换为5x5二维数组
    memory_grid = np.array(memory_flat).reshape((5, 5))
    # 获取中间3x3区域
    center_grid = memory_grid[1:4, 1:4]
    # 计算3x3区域的平均值
    center_average = np.mean(center_grid)
    return center_average


def reward_shaping(
    frame_no,
    score,
    terminated,
    truncated,
    obs,
    _obs,
    treasure_intervel,
    max_step,
    treasure_num,
    formerPos,
):
    reward = 0
    early_truncated = False
    memory_grid = np.array(_obs[215:240]).reshape((5, 5))

    mem_center_average = np.mean(memory_grid[1:4, 1:4])
    curPos = _obs[0]
    if mem_center_average > 0.5:
        early_truncated = True

    if terminated:
        # The reward for winning
        # 奖励1. 获胜的奖励
        # reward += score
        uncollected_treasures = np.sum(_obs[240:250])
        reward += max(50, 150 - uncollected_treasures * 80)
        if uncollected_treasures < 1e-5:
            reward += max(0, (max_step / (treasure_num + 1) - treasure_intervel) * 0.2)
    else:
        # The reward for being close to the finish line
        # 奖励2. 靠近终点的奖励:
        reward += (obs[129] > _obs[129]) * (_obs[129] / 6)
        # 走到之前进入过的格子时，负奖励
        reward -= (_obs[227] > 0.2) * min((_obs[227] - 0.2), 0.3)
        # # 走到新的格子中，正奖励，鼓励探索
        reward += (_obs[227] == 0.1) * 0.1
        # 视野内出现宝箱
        if np.sum(np.array(_obs[165:190])) > np.sum(np.array(obs[165:190])):
            reward += 1
        if not truncated:
            # The reward for obtaining a treasure chest
            # 奖励3. 获得宝箱的奖励
            if score > 0:
                reward += score + max(
                    0, (max_step / (treasure_num + 1) - treasure_intervel) * 0.2
                )
            else:
                # The reward for being close to the treasure chest (considering only the nearest one)
                # 奖励4. 靠近宝箱的奖励(只考虑最近的那个宝箱)
                treasure_dist_old = np.array(obs[130:140])
                treasure_dist = np.array(_obs[130:140])
                min_dist = np.min(treasure_dist_old)
                # 找到所有最小值的下标
                min_dist_index = np.where(treasure_dist_old == min_dist)[
                    0
                ]  # 加上130以获取在obs中的实际位置
                # print(min_dist_index,type(min_dist_index))
                # 比较newobs中所有这些下标位置的元素与obs中相应位置元素的大小
                if np.any(
                    treasure_dist_old[min_dist_index] < treasure_dist[min_dist_index]
                ):
                    reward += min_dist / 6 * 0.2
            if early_truncated:
                reward -= 15
            if curPos == formerPos:
                reward -= 2
    return reward, early_truncated
