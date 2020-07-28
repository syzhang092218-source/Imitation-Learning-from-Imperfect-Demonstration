import numpy as np
import argparse


env_name = 'BipedalWalkerHardcore-v3'
truth = False

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=env_name, metavar='G',
                    help='name of the environment to run')
parser.add_argument('--truth', action='store_true', default=truth)
args = parser.parse_args()

env_name = args.env_name
truth = args.truth

rewards = np.load('./Demos/{}_rewards.npy'.format(args.env_name))

if args.truth:
    confidence = np.ones_like(rewards)
else:
    rewards = np.clip(rewards, a_min=-1, a_max=1)
    rewards += 1
    max_reward = np.max(rewards)
    confidence = rewards / max_reward

np.save('./Demos/{}_mixture_conf.npy'.format(args.env_name), confidence.T)
