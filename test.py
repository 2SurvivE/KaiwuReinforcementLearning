import numpy as np
from diy.rl_utils import RolloutBuffer
buffer_size = 100
obs_shape = (250,)  # 例如，观测维度为10
action_shape = (1,)  # 例如，动作维度为2
device = 'cuda'
gae_lambda = 0.95
gamma = 0.99
buffer = RolloutBuffer(buffer_size, obs_shape, action_shape, device, gae_lambda, gamma)
buffer.load('ckpt/buffer2.npz')
# print(buffer.actions[:100])
# print(buffer.observations[:10,0]//64,buffer.observations[:10,0]%64)
# print('treasure_dist',buffer.observations[:10,130: 140])
# print('obstacle',buffer.observations[:10,140: 165].reshape(-1,5,5))
# print('memory',buffer.observations[:10,215: 240].reshape(-1,5,5))
# print('end dist',buffer.observations[:10,129] )
print(buffer.pos)
size = buffer.pos
print(buffer.advantages[:size])
print(buffer.values[:size])
print(buffer.rewards[:size])

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np

# # 定义FCNN模型
# class FCNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size):
#         super(FCNN, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_size, output_size)

#     def forward(self, x):
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         return out

# # 设置参数
# one_hot_length = 10  # 假设每个坐标用长度为10的one-hot编码表示
# input_size = 2 * one_hot_length
# hidden_size = 128
# output_size = 2  # 例如回归任务中的两个输出坐标
# batch_size = 16
# num_epochs = 20
# learning_rate = 0.001

# # 生成示例数据
# def generate_data(num_samples):
#     x_coords = np.random.randint(0, one_hot_length, num_samples)
#     y_coords = np.random.randint(0, one_hot_length, num_samples)
#     x_one_hot = np.eye(one_hot_length)[x_coords]
#     y_one_hot = np.eye(one_hot_length)[y_coords]
#     inputs = np.hstack((x_one_hot, y_one_hot))
#     targets = np.stack((x_coords, y_coords), axis=1)
#     return torch.tensor(inputs, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)

# # 生成训练数据
# train_inputs, train_targets = generate_data(1000)
# train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)

# # 初始化模型、损失函数和优化器
# model = FCNN(input_size, hidden_size, output_size)
# criterion = nn.MSELoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# # 训练模型
# for epoch in range(num_epochs):
#     for inputs, targets in train_loader:
#         outputs = model(inputs)
#         print(inputs.shape,outputs)
#         break
#         loss = criterion(outputs, targets)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#     break
#     print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# print("Training complete.")
# print(train_inputs)

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class MazeAgentWithEmbedding(nn.Module):
#     def __init__(self, maze_size, embedding_dim, output_dim):
#         super(MazeAgentWithEmbedding, self).__init__()
#         self.x_embedding = nn.Embedding(maze_size[0], embedding_dim)
#         self.y_embedding = nn.Embedding(maze_size[1], embedding_dim)
#         self.fc1 = nn.Linear(embedding_dim * 2, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, output_dim)

#     def forward(self, x, y):
#         x_emb = self.x_embedding(x)
#         y_emb = self.y_embedding(y)
#         position_emb = torch.cat((x_emb, y_emb), dim=-1)
#         x = F.relu(self.fc1(position_emb))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# # 迷宫的大小，例如64x64
# maze_size = (64, 64)
# embedding_dim = 8  # 嵌入维度
# output_dim = 4  # 动作的数量，例如上、下、左、右

# # 当前坐标(x, y)
# current_position = (12, 45)
# x_input = torch.tensor([current_position[0]], dtype=torch.long)
# y_input = torch.tensor([current_position[1]], dtype=torch.long)

# # 定义神经网络
# agent = MazeAgentWithEmbedding(maze_size, embedding_dim, output_dim)

# # 前向传播
# output = agent(x_input, y_input)
# print(output)
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class MazeAgentWithCNN(nn.Module):
#     def __init__(self, maze_size, embedding_dim, num_treasures,hideen_dim1,hideen_dim2, output_dim):
#         super(MazeAgentWithCNN, self).__init__()
        
#         # 嵌入层
#         self.x_embedding = nn.Embedding(maze_size[0], embedding_dim)
#         self.y_embedding = nn.Embedding(maze_size[1], embedding_dim)
        
#         # 图像处理层 (5x5 输入)
#         self.cnn_layer = nn.Sequential(
#             nn.Conv2d(4, hideen_dim1, kernel_size=3, padding=1),  # 5x5 输入, 输出 16x5x5
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),       # 16x5x5 -> 16x2x2
#             nn.Flatten(start_dim=1)
#         )
        
#         # 处理其他特征
#         self.fc_dist = nn.Linear(11, hideen_dim1)  # end_dist (1) + treasure_dist (10)
#         self.fc_treasure_status = nn.Linear(num_treasures, hideen_dim1)
        
#         # 融合层
#         self.fc1 = nn.Linear(hideen_dim1 * 6 + 2*embedding_dim, hideen_dim2)  # 16*4 (图像特征) + 16 (距离信息) + 16 (宝箱状态) + 16(位置信息)
#         self.fc2 = nn.Linear(hideen_dim2, 128)
#         self.fc3 = nn.Linear(128, output_dim)

#     def forward(self, x, y, end_dist, treasure_dist, images, treasure_status):
#         # 嵌入位置坐标
#         x_emb = self.x_embedding(x).flatten(-2)
#         y_emb = self.y_embedding(y).flatten(-2)
#         position_emb = torch.cat((x_emb, y_emb), dim=-1)
#         # 图像特征处理
#         features = self.cnn_layer(images.float())
#         # 处理距离信息
#         dist_info = torch.cat((end_dist, treasure_dist), dim=-1).float()
#         dist_processed = F.relu(self.fc_dist(dist_info))
#         # 处理宝箱状态
#         # print(treasure_status.shape)
#         treasure_status_processed = F.relu(self.fc_treasure_status(treasure_status))
#         print('----')
#         for ten in [position_emb, features, dist_processed, treasure_status_processed]:
#             print(ten.shape)
#         # 融合特征
#         combined_features = torch.cat((position_emb, features, dist_processed, treasure_status_processed), dim=-1)
#         # print(combined_features.shape)
#         # 通过全连接层处理
#         x = F.relu(self.fc1(combined_features))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


# class SharedBase(nn.Module):
#     def __init__(self, hidden_dim, embedding_dim=8, maze_size=(64,64), num_treasures=10):
#         super(SharedBase, self).__init__()
        
#         # 嵌入层
#         self.x_embedding = nn.Embedding(maze_size[0], embedding_dim)
#         self.y_embedding = nn.Embedding(maze_size[1], embedding_dim)
        
#         # 图像处理层 (5x5 输入)
#         self.cnn_layer = nn.Sequential(
#             nn.Conv2d(4, hidden_dim, kernel_size=3, padding=1),  # 4通道输入, 输出 hidden_dim 通道
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),       # 缩小特征图
#             nn.Flatten(start_dim=-3)
#         )
#         # 图像特征处理后的 LayerNorm
#         self.layernorm_cnn = nn.LayerNorm(hidden_dim * 2 * 2)  # 2x2 是卷积后特征图的大小

#         # 处理其他特征
#         self.fc_dist = nn.Linear(11, hidden_dim)  # end_dist (1) + treasure_dist (10)
#         self.fc_treasure_status = nn.Linear(num_treasures, hidden_dim)
        
#         # 其他特征处理后的 LayerNorm
#         self.layernorm_dist = nn.LayerNorm(hidden_dim)
#         self.layernorm_treasure_status = nn.LayerNorm(hidden_dim)

#     def forward(self, input):
#         (x, y, end_dist, treasure_dist, images, treasure_status) = input
#         # 嵌入位置坐标
#         x_emb = self.x_embedding(x).flatten(-2)
#         y_emb = self.y_embedding(y).flatten(-2)
#         position_emb = torch.cat((x_emb, y_emb), dim=-1)
        
#         # 图像特征处理
#         features = self.cnn_layer(images.float())
#         features = self.layernorm_cnn(features)  # 应用 LayerNorm
        
#         # 处理距离信息
#         dist_info = torch.cat((end_dist, treasure_dist), dim=-1).float()
#         dist_processed = F.relu(self.fc_dist(dist_info))
#         dist_processed = self.layernorm_dist(dist_processed)  # 应用 LayerNorm
        
#         # 处理宝箱状态
#         treasure_status_processed = F.relu(self.fc_treasure_status(treasure_status))
#         treasure_status_processed = self.layernorm_treasure_status(treasure_status_processed)  # 应用 LayerNorm
        
#         # 融合特征
#         combined_features = torch.cat((position_emb, features, dist_processed, treasure_status_processed), dim=-1)

#         return combined_features



# class PPOFeatureExtractor(nn.Module):
#     def __init__(self,hidden_dim_base=16, hidden_dim_ac =128 , embedding_dim=8, action_dim=4, maze_size=(64,64), num_treasures=10):
#         super(PPOFeatureExtractor, self).__init__()  # 调用父类的__init__方法
#         self.shared_base = SharedBase(hidden_dim=hidden_dim_base, embedding_dim=8, maze_size=maze_size, num_treasures=num_treasures)

#         self.actor = nn.Sequential(
#             nn.Linear(6*hidden_dim_base+2*embedding_dim, hidden_dim_ac),
#             nn.LeakyReLU(),
#             nn.LayerNorm(hidden_dim_ac),  # 添加LayerNorm
#             nn.Linear(hidden_dim_ac, action_dim),
#         )
#         self.critic = nn.Sequential(
#             nn.Linear(6*hidden_dim_base+2*embedding_dim, hidden_dim_ac),
#             nn.LeakyReLU(),
#             nn.LayerNorm(hidden_dim_ac),  # 添加LayerNorm
#             nn.Linear(hidden_dim_ac, 1),
#         )
#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
#                 nn.init.constant_(m.bias, 0)

#     def forward_actor(self, features_dict):
#         features = features_dict['features']
#         masks = features_dict['mask']
#         latent = self.shared_base(features)
#         probs = self.actor(latent)
#         if torch.isnan(probs).any():
#             print(features, latent, probs)
#         probs = probs.masked_fill(masks, -1e9)  # 使用mask
#         probs = F.softmax(probs, dim=-1)
#         return probs
#         pass
    
#     def forward(self,features_dict):
#         features = features_dict['features']
#         masks = features_dict['mask']
#         latent = self.shared_base(features)
#         probs = self.actor(latent)
#         if torch.isnan(probs).any():
#             print(feature_maps, latent, probs)
#         probs = probs.masked_fill(masks, -1e9)  # 使用mask
#         probs = F.softmax(probs, dim=-1)
#         values = self.critic(latent)
#         return probs,values


# def processFeature(obs:torch.tensor, device ='cuda'):
#     obs = obs.reshape(-1,250).to(device)
#     pos_x = obs[:,0:1].int()//64
#     pos_y = obs[:,0:1].int()% 64
#     end_dist = obs[:,129:130]
#     treasure_dist = obs[:,130:140]
#     treasure_status = obs[:,-10:]
#     feature_map = obs[:,140:240].reshape(-1,4,5,5).float()
#     # feature_x = torch.concatenate([pos_x,pos_y,end_dist,treasure_dist,treasure_status],dim=1).float().to(device)
#     mask = generate_mask_vectorized(obs).to(device)
#     return {"features":(pos_x,pos_y,end_dist,treasure_dist,feature_map,treasure_status),"mask":mask}
# # x, y, end_dist, treasure_dist, images, treasure_status
# def generate_mask_vectorized(data):
#     # 提取所有行的obstacle_flat并重塑为 (n, 5, 5)
#     obstacle_map = data[:, 140:165].reshape(-1, 5, 5)
    
#     # 选择特定位置的值并生成掩码
#     mask = torch.stack([
#         obstacle_map[:, 2, 3] == 1,
#         obstacle_map[:, 2, 1] == 1,
#         obstacle_map[:, 1, 2] == 1,
#         obstacle_map[:, 3, 2] == 1
#     ], dim=1)
#     return mask


# # 迷宫的大小，例如64x64
# maze_size = (64, 64)
# embedding_dim = 8  # 嵌入维度
# num_treasures = 10  # 宝箱数量
# output_dim = 4  # 动作的数量，例如上、下、左、右

# # 当前坐标(x, y)和其他特征
# # x_input = torch.tensor([12], dtype=torch.long).reshape(-1,1)
# # y_input = torch.tensor([45], dtype=torch.long).reshape(-1,1)
# # end_dist = torch.tensor([2], dtype=torch.long).reshape(-1,1)
# # treasure_dist = torch.tensor([1] * 10, dtype=torch.long).reshape(-1,10)  # 示例宝箱距离
# # img = torch.tensor([[0] * 100], dtype=torch.float32).reshape(-1,4, 5, 5) 
# # treasure_status = torch.tensor([1] * 10, dtype=torch.float32).reshape(-1,10)  # 示例宝箱状态

# # 定义神经网络
# # agent = MazeAgentWithCNN(maze_size, embedding_dim, num_treasures, 16,128,output_dim)
# # agent = SharedBase(16)
# agent = PPOFeatureExtractor()
# feature_dict = processFeature(torch.rand((250,)),device='cpu')
# # print(feature_dict)
# output = agent.forward(feature_dict)
# # 前向传播
# # output = agent(x_input, y_input, end_dist, treasure_dist, img, treasure_status)
# print(output)
