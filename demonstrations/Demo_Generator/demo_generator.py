import gym
import argparse
from stable_baselines import TRPO, PPO2, SAC, A2C, ACER, ACKTR, DQN, DDPG, HER, TD3
from stable_baselines.common.evaluation import evaluate_policy
import os
import numpy as np

ALGOS = {
    'a2c': A2C,
    'acer': ACER,
    'acktr': ACKTR,
    'dqn': DQN,
    'ddpg': DDPG,
    'her': HER,
    'sac': SAC,
    'ppo2': PPO2,
    'trpo': TRPO,
    'td3': TD3
}

env_name = "BipedalWalker-v3"
alg = 'sac'

parser = argparse.ArgumentParser()
parser.add_argument('--render', action='store_true', default=False)
parser.add_argument('-n', '--n-demos', help='number of demonstrations to generate', default=15000, type=int)
parser.add_argument('--evaluate', action='store_true', default=False, help='evaluate the network or not')
parser.add_argument('--env_name', default=env_name, help='env to run the model on')
parser.add_argument('--alg', default=alg, help='algorithm to run')
args = parser.parse_args()

env_name = args.env_name
alg = args.alg
model_path = "./Networks/{}_{}.pkl".format(alg, env_name)
found = os.path.isfile(model_path)
if not found:
    ValueError("No model found for in path: {}".format(model_path))
log_dir = 'log'

# create the env and load the model
env = gym.make(env_name)
model = ALGOS[alg].load(model_path, env=env)
print("Successfully load the network")

# evaluate the network
if args.evaluate:
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10, render=False)
    print("mean reward: {}\tstd reward: {}".format(mean_reward, std_reward))

# record the demos
i_demos = 0
num_demos = args.n_demos
num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
demos = np.zeros((num_demos, num_inputs + num_actions))
demo_rewards = np.zeros((1, num_demos))

# record the rewards
obs = env.reset()
episode_reward = 0.0
episode_rewards, episode_lengths = [], []
ep_len = 0
state = None

for i_demos in range(args.n_demos):
    # record the observation
    demos[i_demos, :num_inputs] = obs

    # select an action and update the env
    action, state = model.predict(obs, state=state, deterministic=False)
    if isinstance(env.action_space, gym.spaces.Box):
        action = np.clip(action, env.action_space.low, env.action_space.high)
    obs, reward, done, infos = env.step(action)

    # record the demonstration
    demos[i_demos, num_inputs:] = action
    demo_rewards[0, i_demos] = reward
    i_demos += 1

    if args.render:
        env.render('human')

    episode_reward += reward
    ep_len += 1

    if done:
        print("Episode Reward: {:.2f}".format(episode_reward))
        print("Episode Length", ep_len)
        print("Current Demos: ", i_demos)
        state = None
        episode_rewards.append(episode_reward)
        episode_lengths.append(ep_len)
        episode_reward = 0.0
        ep_len = 0
        env.reset()

if args.render:
    env.close()
print("Mean reward: {:.2f} +/- {:.2f}".format(np.mean(episode_rewards), np.std(episode_rewards)))
print("Mean episode length: {:.2f} +/- {:.2f}".format(np.mean(episode_lengths), np.std(episode_lengths)))

# save the demonstration
np.save('../Demo_Generator/Demos/{}_mixture.npy'.format(env_name), demos)
np.save('../Demo_Generator/Demos/{}_rewards.npy'.format(env_name), demo_rewards)
print("Demonstration saved")
