from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from common import *
import torch.nn.functional as F
from torch.distributions import Normal
HIDDEN_SIZE = 256


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents, args, output_activation='tanh'):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agents = num_agents
        self.args = args

        self.fc1 = torch.nn.Linear(obs_dim, HIDDEN_SIZE)
        self.fc_mu = torch.nn.Linear(HIDDEN_SIZE, act_dim)
        self.fc_std = torch.nn.Linear(HIDDEN_SIZE, act_dim)
        self.activate = nn.Softmax(dim=-1)
        
    def forward(self, x):

        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        std = F.softplus(self.fc_std(x))
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  # rsample()是重参数化采样
        log_prob = dist.log_prob(normal_sample)
        action = self.activate(normal_sample)
        # 计算tanh_normal分布的对数概率密度
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents, args):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agents = num_agents

        self.args = args

        sizes_prev = [obs_dim + act_dim, HIDDEN_SIZE]

# 如果args.algo参数为"bicnet"，则使用LSTMNet作为通信网络，否则使用普通的多层感知机（MLP）。
# 因此，bicnet算法相对于ddpg算法来说，更注重多智能体之间的通信和协作，而ddpg算法更注重单智能体的决策和行动。
        
        if self.args.algo == "bicnet":
            self.comm_net = LSTMNet(HIDDEN_SIZE, HIDDEN_SIZE)
            sizes_post = [HIDDEN_SIZE << 1, HIDDEN_SIZE, 1]
        elif self.args.algo == "ddpg":
            sizes_post = [HIDDEN_SIZE, HIDDEN_SIZE, 1]
        elif self.args.algo == "sac":
            sizes_post = [HIDDEN_SIZE, HIDDEN_SIZE, 1]

        self.prev_dense = mlp(sizes_prev)
        self.post_dense = mlp(sizes_post)

    def forward(self, obs_batch, action_batch):
        out = torch.cat((obs_batch, action_batch), dim=-1)
        out = self.prev_dense(out)

        if self.args.algo == "bicnet":
            out = self.comm_net(out)

        out = self.post_dense(out)
        return out


class LSTMNet(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 batch_first=True,
                 bidirectional=True):
        super(LSTMNet, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
            bidirectional=bidirectional
        )

    def forward(self, data, ):
        output, (_, _) = self.lstm(data)
        return output

class RNNNet(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 batch_first=True,
                 bidirectional=True):
        super(RNNNet, self).__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
            bidirectional=bidirectional
        )

    def forward(self, data):
        output, _ = self.rnn(data)
        return output