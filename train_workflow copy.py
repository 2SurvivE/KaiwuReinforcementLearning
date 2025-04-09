#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :gorge_walk
@File    :train_workflow.py
@Author  :kaiwu
@Date    :2022/12/15 22:50

"""

from kaiwu_agent.utils.common_func import Frame
from kaiwu_agent.utils.common_func import attached
import time
import math
import numpy as np
import torch
from diy.rl_utils import RolloutBuffer
from diy.feature.definition import (
    observation_process,
    action_process,
    sample_process,
    reward_shaping,
    extractAVL,
    generate_mask_vectorized,
    processFeature,
    processFeature
)
import os
from diy.config import Config

@attached
def workflow(envs, agents, logger=None, monitor=None):
    env, agent = envs[0], agents[0]
    max_step = Config.max_step
    treasure_num = Config.treasure_num
    EPISODES = Config.EPISODES
    # 测试 RolloutBuffer 类
    buffer_size = Config.buffer_size
    obs_shape = (250,)  # 例如，观测维度为10
    action_shape = (1,)  # 例如，动作维度为2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gae_lambda = Config.gae_lambda
    gamma = Config.gamma
    # 创建 RolloutBuffer 实例
    buffer = RolloutBuffer(buffer_size, obs_shape, action_shape, device, gae_lambda, gamma)
    
    # agent.load_modelTmp('ckpt',id=str(17))
    # agent.save_model(id=8016)
    # return 
    # Initializing monitoring data
    # 监控数据初始化
    monitor_data = {
        "reward": 0,
        "entropy_loss": 0,
        "policy_gradient_loss": 0,
        "value_loss": 0,
        "approx_kl": 0,
        "clip_fraction": 0,
        "loss":0,
        "explained_variance":0,
    }
    agent.set_monitor(monitor_data)
    last_report_monitor_time = time.time()

    logger.info("Start Training ...")
    start_t = time.time()

    total_rew, win_cnt = (
        0,
        0,
    )
    eps_count = 0

    for episode in range(1,EPISODES):
    # for episode in range(100):
        if episode% 501==0:
            agent.save_model()
            agent.save_modelTmp('ckpt',id=str(episode))
            pass
        # User-defined environment launch configuration
        # 用户自定义的环境启动配置
        usr_conf = {
            "diy": {
                "start": [29, 9],
                "end": [11, 55],
                "treasure_id": [0, 1, 2, 3, 4],
                # "treasure_num": 5,
                # "treasure_random": 1,
                "max_step": max_step,
            }
        }
        # Reset the environment and obtain the initial state
        # 重置环境, 并获取初始状态
        
        obs = env.reset(usr_conf=usr_conf)

        # Disaster recovery
        # 容灾
        if obs is None:
            continue

        # First frame processing
        # 首帧处理
        obs_data = observation_process(obs)

        # Task loop
        # 任务循环
        terminated = False
        truncated = False
        last_episode_start = True
        
        treasure_intervel=0
        formerPos = -1 # 记录前2个的位置
        while (not terminated) and (not truncated) and (not buffer.full):
            # Agent performs inference to obtain the predicted action for the next frame
            # Agent 进行推理, 获取下一帧的预测动作
            with torch.no_grad():
                act_data = agent.predict(list_obs_data=[obs_data])[0]
            # Unpacking ActData into actions
            # ActData 解包成动作
            action, value, log_prob = extractAVL(act_data)
            # Interact with the environment, perform actions, and obtain the next state
            # 与环境交互, 执行动作, 获取下一步的状态
            frame_no, new_obs, score, terminated, truncated, env_info = env.step(action.flatten().item())
            treasure_intervel+=1
            if new_obs is None:
                break
            # Feature processing
            # 特征处理
            new_obs_data = observation_process(new_obs)
            # Compute reward
            # 计算 reward
            # shaped_reward = reward_shaping(frame_no, score, terminated, truncated, obs, new_obs)
            shaped_reward,truncated = reward_shaping(frame_no, score, terminated, truncated, obs, new_obs, treasure_intervel, max_step, treasure_num, formerPos)
            formerPos = int(obs[0])
            reward = torch.tensor(shaped_reward)
            buffer.add(np.array(obs), action,reward,value,log_prob,last_episode_start)
            obs_data = new_obs_data  # type: ignore[assignment]
            obs = new_obs
            last_episode_start = terminated
            # 更新总奖励和状态
            total_rew += shaped_reward
            if score==0 and not terminated:
                treasure_intervel = 0
        
        win_cnt += terminated
        eps_count += 1
        if not buffer.full:
            continue
        with torch.no_grad():
            # Compute value for the last timestep
            act_data = agent.predict(list_obs_data=[obs_data])[0]
            _, last_value, _ = extractAVL(act_data)
        buffer.compute_returns_and_advantage(last_values=last_value, dones= terminated)

        """
        Update policy using the currently gathered rollout buffer.
        """
        agent.learn([buffer])
        buffer.reset()
        # Reporting training progress
        # 上报训练进度
        now = time.time()
        if now - last_report_monitor_time > 20:
            logger.info(f"Episode: {episode + 1}, Reward: {total_rew/eps_count}")
            # logger.info(f"Training Win Rate: {win_cnt / (episode + 1)}")
            logger.info(f"terminated Rate: {win_cnt / eps_count}")
            monitor_data["reward"] = total_rew
            if monitor:
                monitor.put_data({os.getpid(): monitor_data})
            total_rew = 0
            eps_count = 0
            win_cnt = 0
            last_report_monitor_time = now

        # The model has converged, training is complete, and reporting monitoring metric
        # 模型收敛, 结束训练, 上报监控指标
        # if win_cnt / (episode + 1) > 0.98 and episode > 100:
        #     logger.info(f"Training Converged at Episode: {episode + 1}")
        #     monitor_data["reward"] = total_rew
        #     if monitor:
        #         monitor.put_data({os.getpid(): monitor_data})
        #     break
        

    end_t = time.time()
    logger.info(f"Training Time for {episode + 1} episodes: {end_t - start_t} s")
    agent.episodes = episode + 1

    # model saving
    # 保存模型
    agent.save_model()

    return

# agent.load_modelTmp('ckpt','246-backup')
    # buffer_size = 100
    # obs_shape = (250,)  # 例如，观测维度为10
    # action_shape = (1,)  # 例如，动作维度为2
    # device = 'cuda'
    # gae_lambda = 0.95
    # gamma = 0.99
    # # 创建 RolloutBuffer 实例
    # buffer = RolloutBuffer(buffer_size, obs_shape, action_shape, device, gae_lambda, gamma)
    # usr_conf = {
    #         "diy": {
    #             "start": [29, 9],
    #             "end": [11, 55],
    #             "treasure_id": [0, 1, 2, 3, 4],
    #             # "treasure_num": 5,
    #             # "treasure_random": 1,
    #             "max_step": 800,
    #         }
    #     }
    # obs = env.reset(usr_conf=usr_conf)
    # obs_data = observation_process(obs)

    # # Task loop
    # # 任务循环
    # terminated = False
    # truncated = False
    # last_episode_start = True
    # while (not terminated) and (not truncated):
    #     # Agent performs inference to obtain the predicted action for the next frame
    #     # Agent 进行推理, 获取下一帧的预测动作
    #     with torch.no_grad():
    #         act_data = agent.predict(list_obs_data=[obs_data])[0]
    #     # Unpacking ActData into actions
    #     # ActData 解包成动作
    #     action, value, log_prob = extractAVL(act_data)
    #     # Interact with the environment, perform actions, and obtain the next state
    #     # 与环境交互, 执行动作, 获取下一步的状态
    #     frame_no, new_obs, score, terminated, truncated, env_info = env.step(action.flatten().item())

    #     if new_obs is None:
    #         break
    #     # Feature processing
    #     # 特征处理
    #     new_obs_data = observation_process(new_obs)
    #     # Compute reward
    #     # 计算 reward
    #     shaped_reward = reward_shaping(frame_no, score, terminated, truncated, obs, new_obs)
    #     reward = torch.tensor(shaped_reward)
    #     buffer.add(np.array(obs), action,reward,value,log_prob,last_episode_start)
    #     obs_data = new_obs_data  # type: ignore[assignment]
    #     obs = new_obs
    #     last_episode_start = terminated
    #     # 更新总奖励和状态
    #     # total_rew += shaped_reward
    # buffer.save('ckpt/buffer.npz')
    # buffer.load('ckpt/buffer.npz')