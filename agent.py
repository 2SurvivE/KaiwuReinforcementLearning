#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
@Project :gorge_walk
@File    :agent.py
@Author  :kaiwu
@Date    :2022/12/15 22:50

"""

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from kaiwu_agent.agent.base_agent import (
    predict_wrapper,
    exploit_wrapper,
    learn_wrapper,
    save_model_wrapper,
    load_model_wrapper,
)
from kaiwu_agent.utils.common_func import attached
from kaiwu_agent.agent.base_agent import BaseAgent
from diy.feature.definition import ActData, processFeature
from diy.config import Config

from typing import (
    Optional,
    Tuple,
)
from diy.rl_utils import explained_variance
from diy.model.model import PPOFeatureExtractor


@attached
class Agent(BaseAgent):
    """PPO算法,采用截断方式"""

    def __init__(
        self, agent_type="player", device=None, logger=None, monitor=None
    ) -> None:
        self.logger = logger
        # Initialize parameters
        # 参数初始化
        # self.feature_extractor = PPOFeatureExtractor(channels=Config.channels, hidden_dim=Config.hidden_dim,action_dim= Config.action_dim).to(device)
        self.feature_extractor = PPOFeatureExtractor(
            hidden_dim_base=16, hidden_dim_ac=128, embedding_dim=8, action_dim=4
        ).to(device)
        self.paramslist = list(self.feature_extractor.parameters())
        self.optimizer = optim.Adam(
            self.paramslist, lr=Config.learning_rate, weight_decay=Config.weight_decay
        )
        self.n_epochs = Config.n_epochs  # 一条序列的数据用来训练轮数
        self.clip_range = Config.clip_range  # PPO中截断范围的参数
        self.clip_range_vf = None
        self.debug = Config.debug
        self.device = device
        self.vf_coef = Config.vf_coef
        self.ent_coef = Config.ent_coef
        self.target_kl = Config.target_kl
        self.max_grad_norm = Config.max_grad_norm
        self.batch_size = Config.batch_size
        self.normalize_advantage = Config.normalize_advantage
        self.monitor_data = None
        super().__init__(agent_type, device, logger, monitor)

    def set_monitor(self, monitor_data):
        self.monitor_data = monitor_data

    @predict_wrapper
    def predict(self, list_obs_data):
        """
        The input is list_obs_data, and the output is list_act_data.
        """
        """
        输入是 list_obs_data, 输出是 list_act_data
        """
        # Evaluate the values for the given observations
        # features = processFeature(torch.tensor(list_obs_data[0].feature),device=self.device)
        features = processFeature(
            torch.tensor(list_obs_data[0].feature), device=self.device
        )
        probs, values = self.feature_extractor.forward(features)
        distribution = torch.distributions.Categorical(probs)
        actions = distribution.sample()
        actions = actions.reshape((-1, 1))
        log_probs = distribution.log_prob(actions)
        return [ActData(action=actions, value=values, log_prob=log_probs)]

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: actions
        :return: estimated values, log likelihood of taking those actions
            and entropy of the actions distribution.
        """
        # Preprocess the observation if needed
        features = processFeature(obs, device=self.device)
        probs, values = self.feature_extractor.forward(features)
        distribution = torch.distributions.Categorical(probs)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_probs, entropy

    @exploit_wrapper
    def exploit(self, list_obs_data):
        features = processFeature(
            torch.tensor(list_obs_data[0].feature), device=self.device
        )
        probs = self.feature_extractor.forward_actor(features)
        actions = torch.argmax(probs, dim=-1)
        return [ActData(action=actions.flatten().item())]

    @learn_wrapper
    def learn(self, list_sample_data):
        buffer = list_sample_data[0]
        continue_training = True
        # train for n_epochs epochs
        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer
            for rollout_data in buffer.get(self.batch_size):
                actions = rollout_data.actions.flatten()
                values, log_probs, entropy = self.evaluate_actions(
                    rollout_data.observations, actions
                )
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (
                        advantages.std() + 1e-8
                    )

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_probs - rollout_data.log_probs)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(
                    ratio, 1 - self.clip_range, 1 + self.clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean(
                    (torch.abs(ratio - 1) > self.clip_range).float()
                ).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new values
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.values + torch.clamp(
                        values - rollout_data.values,
                        -self.clip_range_vf,
                        self.clip_range_vf,
                    )

                # values loss using the TD(gae_lambda) target
                value_loss = F.smooth_l1_loss(rollout_data.returns, values_pred)
                # value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_probs)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = (
                    policy_loss
                    + self.ent_coef * entropy_loss
                    + self.vf_coef * value_loss
                )

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with torch.no_grad():
                    log_ratio = log_probs - rollout_data.log_probs
                    approx_kl_div = (
                        torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    )
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    print(
                        f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                    )
                    break

                # Optimization step
                self.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.paramslist, self.max_grad_norm)
                self.optimizer.step()
            if not continue_training:
                break

        explained_var = explained_variance(
            buffer.values.flatten(), buffer.returns.flatten()
        )

        self.monitor_data["entropy_loss"] = np.mean(entropy_losses)
        self.monitor_data["policy_gradient_loss"] = np.mean(pg_losses)
        self.monitor_data["value_loss"] = np.mean(value_losses)
        self.monitor_data["approx_kl"] = np.mean(approx_kl_divs)
        self.monitor_data["clip_fraction"] = np.mean(clip_fractions)
        self.monitor_data["loss"] = loss.item()
        self.monitor_data["explained_variance"] = explained_var
        # Logs
        self.logger.info(
            f"train/entropy_loss:{self.monitor_data['entropy_loss']}",
        )
        self.logger.info(
            f"train/policy_gradient_loss:{self.monitor_data['policy_gradient_loss']}"
        )
        self.logger.info(f"train/value_loss:{self.monitor_data['value_loss']}")
        self.logger.info(f"train/approx_kl:{self.monitor_data['approx_kl']}")
        self.logger.info(f"train/clip_fraction:{self.monitor_data['clip_fraction']}")
        self.logger.info(f"train/loss:{self.monitor_data['loss']}")
        self.logger.info(f"train/explained_variance:{explained_var}")
        # if hasattr(self.policy, "log_std"):
        #     self.logger.info("train/std", th.exp(self.policy.log_std).mean().item())
        # self.logger.info("train/n_updates", self._n_updates, exclude="tensorboard")
        # self.logger.info("train/clip_range", clip_range)
        # if self.clip_range_vf is not None:
        #     self.logger.info("train/clip_range_vf", clip_range_vf)
        pass

    @save_model_wrapper
    def save_model(self, path=None, id="1"):
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pth"
        # np.save(model_file_path, self.Q)
        torch.save(self.feature_extractor.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    @load_model_wrapper
    def load_model(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pth"
        try:
            self.feature_extractor.load_state_dict(
                torch.load(model_file_path, map_location=self.device)
            )
            self.logger.info(f"load model {model_file_path} successfully")
        except FileNotFoundError:
            self.logger.info(f"File {model_file_path} not found")
            exit(1)

    def save_modelTmp(self, path=None, id="1"):
        # To save the model, it can consist of multiple files,
        # and it is important to ensure that each filename includes the "model.ckpt-id" field.
        # 保存模型, 可以是多个文件, 需要确保每个文件名里包括了model.ckpt-id字段
        model_file_path = f"{path}/model.ckpt-{str(id)}.pth"
        # np.save(model_file_path, self.Q)
        torch.save(self.feature_extractor.state_dict(), model_file_path)
        self.logger.info(f"save model {model_file_path} successfully")

    def load_modelTmp(self, path=None, id="1"):
        # When loading the model, you can load multiple files,
        # and it is important to ensure that each filename matches the one used during the save_model process.
        # 加载模型, 可以加载多个文件, 注意每个文件名需要和save_model时保持一致
        model_file_path = f"{path}/model.ckpt-{str(id)}.pth"
        try:
            self.feature_extractor.load_state_dict(
                torch.load(model_file_path, map_location=self.device)
            )
            self.logger.info(f"load model {model_file_path} successfully")
        except FileNotFoundError:
            self.logger.info(f"File {model_file_path} not found")
