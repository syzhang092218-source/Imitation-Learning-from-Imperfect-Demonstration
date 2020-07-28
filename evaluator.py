import gym
import argparse
import numpy as np
import torch
from torch.autograd import Variable


env_name = 'BipedalWalker-v3'
epi = 4500
addition_arg = '_optimal'
render = True
n_eval_episodes = 5

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', default=env_name, help='env to run')
parser.add_argument('--epi', default=epi, help='training episodes of the network')
parser.add_argument('--addition_arg', default=addition_arg, help='additional args')
parser.add_argument('-n', '--n_eval_episodes', default=n_eval_episodes, help='number of episodes for evaluation')
parser.add_argument('--render', action='store_true', default=render, help='render or not')
args = parser.parse_args()

env_name = args.env_name
epi = args.epi
addition_arg = args.addition_arg
n_eval_episodes = args.n_eval_episodes
render = args.render

model_path = 'networks_{}{}/policy_net_{}_epi{}.pkl'.format(env_name, addition_arg, env_name, epi)
policy_net = torch.load(model_path)
env = gym.make(env_name)
state = env.reset()
episode_reward = 0.0
episode_rewards, episode_lengths = [], []
ep_len = 0
i_episode = 0

while i_episode < n_eval_episodes:
    for _ in range(50000):     # prevent infinity loop
        state = torch.from_numpy(state).unsqueeze(0)
        action, _, _ = policy_net(Variable(state))
        action = action.data[0].numpy()
        next_state, reward, done, _ = env.step(action)
        episode_reward += reward
        ep_len += 1

        if render:
            env.render()

        if done:
            print("Episode Reward: {:.2f}".format(episode_reward))
            print("Episode Length", ep_len)
            episode_rewards.append(episode_reward)
            episode_lengths.append(ep_len)
            episode_reward = 0.0
            ep_len = 0
            state = env.reset()
            i_episode += 1
            break

        state = next_state

if render:
    env.close()

print("Mean reward: {:.2f} +/- {:.2f}".format(np.mean(episode_rewards), np.std(episode_rewards)))
print("Mean episode length: {:.2f} +/- {:.2f}".format(np.mean(episode_lengths), np.std(episode_lengths)))

