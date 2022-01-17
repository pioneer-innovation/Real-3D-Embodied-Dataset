import os

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import time
import numpy as np
import gym
import tianshou as ts
from myenv.R3AD_V1 import R3AD_V1
from tianshou_agent.VQL_V2 import VoteQN2
from tianshou_agent.RVQL_V1 import RNNVoteQN2
import torch
import json
from tianshou.utils import BasicLogger
from torch.utils.tensorboard import SummaryWriter

def load_dict(filename):
    '''load dict from json file'''
    with open(filename,"r") as json_file:
        dic = json.load(json_file)
    return dic
def save_dict(filename, dic):
    '''save dict into json file'''
    with open(filename,'w') as json_file:
        json.dump(dic, json_file)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn',force=True)

    train_envs = ts.env.SubprocVectorEnv([lambda: gym.make('R3AD-v3') for i in range(1)])
    test_envs = ts.env.SubprocVectorEnv([lambda: gym.make('R3AD-v3') for i in range(1)])

    np.random.seed(0)
    torch.manual_seed(0)
    train_envs.seed(0)
    test_envs.seed(0)

    net = VoteQN2(6).cuda()
    # net = RNNVoteQN2(6).cuda()
    optim = torch.optim.Adam(net.neck.parameters(), lr=0)
    policy = ts.policy.DQNPolicy(net,
                                 optim,
                                 discount_factor=0.99,
                                 estimation_step=3,
                                 target_update_freq=1000)

    log_path = './log/VQL_V2'
    writer = SummaryWriter(log_path)
    logger = BasicLogger(writer)

    def train_fn(epoch, env_step):
        # nature DQN setting, linear decay in the first 1M steps
        if env_step <= 1e4:
            eps = 1 - env_step / 1e4 * (1 - 0.1)
        else:
            eps = 0.1
        policy.set_eps(eps)
        logger.write('train/eps', env_step, eps)

    def test_fn(epoch, env_step):
        policy.set_eps(1)

    def load_fn(policy):
        policy.load_state_dict(torch.load(os.path.join(log_path, 'policy_step_final.pth')))

    load_fn(policy)

    train_collector = ts.data.Collector(policy, train_envs,ts.data.VectorReplayBuffer(total_size=2000, buffer_num=2))
    test_collector = ts.data.Collector(policy, test_envs, exploration_noise=False)

    test_collector.collect(n_step=1000)

    # result = ts.trainer.offpolicy_trainer(
    #     policy, train_collector, test_collector,
    #     max_epoch=100, step_per_epoch=10, step_per_collect=10,
    #     episode_per_test=1, batch_size=64,update_per_step=0,
    #     train_fn=train_fn,test_fn=test_fn,logger=logger,
    #     stop_fn=lambda mean_rewards: mean_rewards >= 35000,
    #     test_in_train=False,test_freq=1)


    # print(f'Finished training! Use {result["duration"]}')

    print('finish')
