from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from common import *

HIDDEN_SIZE = 256


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents, args, output_activation='tanh'):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agents = num_agents

        self.args = args

        sizes_prev = [obs_dim, HIDDEN_SIZE]
        middle_prev = [HIDDEN_SIZE, HIDDEN_SIZE]
        sizes_post = [HIDDEN_SIZE << 1, HIDDEN_SIZE, act_dim]

        self.prev_dense = mlp(sizes_prev)

        sizes_post = [HIDDEN_SIZE, HIDDEN_SIZE, act_dim]

        self.prev_dense = mlp(sizes_prev)
        self.post_dense = mlp(sizes_post, output_activation=output_activation)

    def forward(self, obs_batch):
        out = self.prev_dense(obs_batch)
        out = self.post_dense(out)
        return out


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents, args):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agents = num_agents
        self.agent_critic_list = []

        self.args = args

        for i in range(num_agents):
            self.agent_critic_list.append(Critic_single(obs_dim, act_dim, num_agents, i, args))

        self.agent_critic_list = torch.nn.ModuleList(self.agent_critic_list)

        self.mixq_net = torch.nn.Linear(num_agents, 1)

    def forward(self, obs_batch, action_batch):
        q_list = []
        for i in range(self.num_agents):
            q = self.agent_critic_list[i](obs_batch, action_batch)
            q_list.append(q)
        q_list = torch.stack(q_list)
        q_list = q_list.permute(1,2,3,0)
        q = self.mixq_net(q_list).squeeze(-1)
        return q

class Critic_single(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents, agent_index, args):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.agent_index = agent_index
        self.num_agents = num_agents

        self.args = args

        sizes_prev = [2*obs_dim + act_dim, HIDDEN_SIZE]

        sizes_post = [HIDDEN_SIZE, HIDDEN_SIZE, 1]

        self.prev_dense = mlp(sizes_prev)
        self.post_dense = mlp(sizes_post)

    def forward(self, obs_batch, action_batch):
        out = torch.cat((obs_batch[:, self.agent_index, : ].unsqueeze(1).repeat(1,self.num_agents,1), obs_batch, action_batch), dim=-1)
        out = self.prev_dense(out)
        out = self.post_dense(out)
        return out

if __name__ == '__main__':
    model = Critic(10,10,5,None,'cuda')
    print(len(list(model.parameters())))