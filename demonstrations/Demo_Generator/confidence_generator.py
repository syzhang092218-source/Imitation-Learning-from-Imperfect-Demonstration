import numpy as np
import argparse


parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--env-name', default="Humanoid-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--truth', action='store_true', default=False)
args = parser.parse_args()

args.env_name = 'BipedalWalker-v3'

rewards = np.load('./Demos/{}_rewards.npy'.format(args.env_name))

if args.truth:
    confidence = np.ones_like(rewards)
else:
    rewards = np.clip(rewards, a_min=-1, a_max=1)
    rewards += 1
    max_reward = np.max(rewards)
    confidence = rewards / max_reward

np.save('./Demos/{}_mixture_conf.npy'.format(args.env_name), confidence.T)
