import argparse
import os
import datetime
import gymnasium as gym
import numpy as np
import itertools
import torch
from sac import SAC
import pickle
from replay_memory import ReplayMemory

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy every 50 episodes (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--max_episodes', type=int, default=2000, metavar='N',
                    help='number of episodes (default: 2000)')
parser.add_argument('--max_timesteps', type=int, default=2000, metavar='N',
                    help='max timesteps in one episode (default: 2000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

if not os.path.exists('stats/'):
    os.makedirs('stats/')

# Environment
# env = NormalizedActions(gym.make(args.env_name))
if args.env_name == "LunarLander-v2":
    env = gym.make(args.env_name, continuous=True)
else:
    env = gym.make(args.env_name)
# env.seed(args.seed)
# env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

# logging variables
critic_1_loss_log = []
critic_2_loss_log = []
policy_loss_log = []
entropy_loss_log = []
alpha_log = []
rewards_log = []
average_rewards_log = []
lengths_log = []


def save_statistics():
    with open(f"./stats/SAC_{args.env_name}-gamma{args.gamma}-tau{args.tau}-lr{args.lr}-alpha{args.alpha}-autotune{args.automatic_entropy_tuning}-seed{args.seed}-stat.pkl", 'wb') as f:
        pickle.dump({'critic_1_loss_log': critic_1_loss_log,
                     'critic_2_loss_log': critic_2_loss_log,
                     'policy_loss_log': policy_loss_log,
                     'entropy_loss_log': entropy_loss_log,
                     'alpha_log': alpha_log,
                     'rewards_log': rewards_log,
                     'average_rewards_log': average_rewards_log,
                     'lengths_log': lengths_log}, f)


# Memory
memory = ReplayMemory(args.replay_size, args.seed)

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in range(1, args.max_episodes+1):
    print(i_episode)
    episode_reward = 0
    episode_steps = 0
    critic_1_losses = []
    critic_2_losses = []
    policy_losses = []
    entropy_losses = []
    done = False
    state, _ = env.reset()

    for t in range(args.max_timesteps):
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(
                    memory, args.batch_size, updates)

                critic_1_losses.append(critic_1_loss)
                critic_2_losses.append(critic_2_loss)
                policy_losses.append(policy_loss)
                entropy_losses.append(ent_loss)
                updates += 1

        next_state, reward, done, trunc, _ = env.step(action)  # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(
            not done)

        memory.push(state, action, reward, next_state,
                    mask)  # Append transition to memory

        state = next_state
        if done or trunc:
            break

    rewards_log.append(episode_reward)
    lengths_log.append(t)
    critic_1_loss_log.append(np.mean(critic_1_losses))
    critic_2_loss_log.append(np.mean(critic_2_losses))
    policy_loss_log.append(np.mean(policy_losses))
    entropy_loss_log.append(np.mean(entropy_losses))
    try:
        alpha_log.append(alpha)
    except:
        raise RuntimeError(
            "Batch size is larger than episode length, try a smaller one!")

    if i_episode % 50 == 0 and args.eval is True:
        avg_reward = 0.
        episodes = 10
        for _ in range(episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            for _ in range(args.max_timesteps):
                action = agent.select_action(state, evaluate=True)

                next_state, reward, done, trunc, _ = env.step(action)
                episode_reward += reward

                state = next_state
                if done or trunc:
                    break
            avg_reward += episode_reward
        avg_reward /= episodes

        average_rewards_log.append((avg_reward, i_episode))

        print("----------------------------------------")
        print(
            f"Avg. Reward last 50 episodes: {round(np.mean(rewards_log[-50:]), 2)}")
        print("Test Episodes: {}, Avg. Reward: {}".format(
            episodes, round(avg_reward, 2)))
        print("----------------------------------------")

        # save every 250 episodes
        if i_episode % 250 == 0:
            print("########## Saving a checkpoint... ##########")
            agent.save_checkpoint(
                args.env_name, "", f"checkpoints/SAC_{args.env_name}_{i_episode}-gamma{args.gamma}-tau{args.tau}-lr{args.lr}-alpha{args.alpha}-autotune{args.automatic_entropy_tuning}-seed{args.seed}.pth")
            save_statistics()

save_statistics()
env.close()
