import os

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
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
from tianshou.data import VectorReplayBuffer


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

    # global_path = './global_value.json'
    # global_dict = {'gpu_id':0}
    # save_dict(global_path,global_dict)
    # global_dict = load_dict(global_path)

    train_envs = ts.env.SubprocVectorEnv([lambda: gym.make('R3AD-v3') for i in range(2)])
    test_envs = ts.env.SubprocVectorEnv([lambda: gym.make('R3AD-v3') for i in range(1)])

    np.random.seed(0)
    torch.manual_seed(0)
    train_envs.seed(0)
    test_envs.seed(0)

    # remeber to set stack num while change the model
    net = VoteQN2(6).cuda()
    # net = RNNVoteQN2(6).cuda()
    optim = torch.optim.Adam(net.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=8, gamma=0.95)
    policy = ts.policy.DQNPolicy(net,
                                 optim,
                                 discount_factor=0.9,
                                 estimation_step=3,
                                 target_update_freq=2000)

    log_path = './log/VQL_V2'
    writer = SummaryWriter(log_path)
    logger = BasicLogger(writer)

    def train_fn(epoch, env_step):
        # nature DQN setting, linear decay in the first 1M steps
        if env_step <= 1e5:
            eps = 1 - env_step / 1e5 * (1 - 0.333)
        else:
            eps = 0.333
        if env_step>5:
            scheduler.step()
        policy.set_eps(eps)
        logger.write('train/eps', env_step, eps)

    def test_fn(epoch, env_step):
        policy.set_eps(0)

    def save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy_step_best.pth'))

    def final_save_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, 'policy_step_final.pth'))

    def load_fn(policy):
        policy.load_state_dict(torch.load(os.path.join(log_path, 'policy.pth')))

    buffer = VectorReplayBuffer(
        total_size=4000, buffer_num=2, ignore_obs_next=True)
    train_collector = ts.data.Collector(policy, train_envs,buffer,exploration_noise=True)
    test_collector = ts.data.Collector(policy, test_envs)
    st = time.time()
    policy.set_eps(1)
    train_collector.collect(n_step=2000)
    et = time.time()
    t = et - st
    print('pre collect time:', t)

    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector,
        max_epoch=100, step_per_epoch=1000, step_per_collect=100,
        episode_per_test=10, batch_size=56,update_per_step=0.1,
        train_fn=train_fn,test_fn=test_fn,logger=logger,
        stop_fn=lambda mean_rewards: mean_rewards >= 3500,
        save_fn=save_fn,test_in_train=False,test_freq=10)

    final_save_fn(policy)

    print(f'Finished training! Use {result["duration"]}')

