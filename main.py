import argparse
import os
import numpy as np
import torch
from sac import SAC
import pickle
from per import PrioritizedReplayBuffer
import hockey.hockey_env as h_env

OPP_CHOICES = ["basic_weak", "basic_strong", "shooting", "defending"]
CHECKPOINT_INTERVAL = 500
EVAL_INTERVAL = 100
EVAL_EPS = 10

# self-play
MIXED_FREQ = 250
ADVANCED_SELF_TRAIN = 100
ADVANCED_BASIC = 50

parser = argparse.ArgumentParser(
    description='PyTorch Soft Actor-Critic Laserhockey Args')
parser.add_argument('--env_name', default="Hockey",
                    help='Gym environment (default: Hockey) -> code works only for hockey')
parser.add_argument('--hockey_train_mode', default="basic_strong", choices=OPP_CHOICES,
                    help='For hockey env, determine the train mode (default: basic_strong)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help=f'Evaluates a policy every {EVAL_INTERVAL} episodes (default: True)')
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
parser.add_argument('--max_episodes', type=int, default=10000, metavar='N',
                    help='number of episodes (default: 10000)')
parser.add_argument('--self_play', type=bool, default=True, metavar='G',
                    help='Train on opponent and random previous agent version (default: True)')
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
                    help='size of replay buffer (default: 1000000)')
parser.add_argument('--replay_alpha', type=float, default=0.1, metavar='N',
                    help='determines how much prioritization is used, α = 0 corresponding to the uniform case (default: 0.1)')
parser.add_argument('--replay_beta', type=float, default=0.1, metavar='N',
                    help='determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities (default: 0.1)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
args = parser.parse_args()

if not os.path.exists('stats/'):
    os.makedirs('stats/')

# Environment
if args.hockey_train_mode == "shooting":
    env = h_env.HockeyEnv(mode=h_env.Mode.TRAIN_SHOOTING)
elif args.hockey_train_mode == "defending":
    env = h_env.HockeyEnv(mode=h_env.Mode.TRAIN_DEFENSE)
else:
    env = h_env.HockeyEnv()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)


def set_opponent(i_episode, basic=True, decay=0.7):
    """
    basic: Whether basic opponent should be loaded or previous trained SAC agent.
    prob: Probability to choose the most recent agent checkpoint.
    """
    if basic:
        print(
            f"Loading {'strong' if args.hockey_train_mode == 'basic_strong' else 'weak'} basic opponent...")
        if args.hockey_train_mode == "basic_weak":
            return h_env.BasicOpponent(weak=True)
        elif args.hockey_train_mode == "basic_strong":
            return h_env.BasicOpponent(weak=False)
        else:
            raise RuntimeError("Invalid mode argument")

    # weighted checkpoint selection (more recent ones more likely)
    num_checkpoints = (i_episode - 1) // CHECKPOINT_INTERVAL
    weights = np.exp(np.linspace(-num_checkpoints * decay, 0, num_checkpoints))
    probabilities = weights / weights.sum()
    selected_index = np.random.choice(
        range(num_checkpoints), p=probabilities) + 1
    selected_checkpoint_ep = selected_index * CHECKPOINT_INTERVAL
    print(
        f"Loading previous agent after {selected_checkpoint_ep} episodes as opponent...")
    op = SAC(env.observation_space.shape[0], env.action_space, args)
    op.load_checkpoint(
        f"checkpoints/SAC_{args.env_name}_{selected_checkpoint_ep}-gamma{args.gamma}-tau{args.tau}-lr{args.lr}-alpha{args.alpha}-autotune{args.automatic_entropy_tuning}-pera{args.replay_alpha}-perb{args.replay_beta}-seed{args.seed}.pth", evaluate=True)


# Hockey has no max episode steps
env_has_max_steps = True
try:
    env._max_episode_steps
except:
    env_has_max_steps = False

# Self-play strategy:
episodes_mixed = int(args.max_episodes * 0.4)
episodes_advanced = int(args.max_episodes * 0.3)
if args.self_play:
    episodes_against_basic = int(args.max_episodes * 0.3)
else:
    # mixed and advanced never come
    episodes_against_basic = args.max_episodes

num_actions = env.action_space.shape[0] // 2

# logging variables
critic_1_loss_log = []
critic_2_loss_log = []
policy_loss_log = []
entropy_loss_log = []
alpha_log = []
rewards_log = []
winner_log = []
lengths_log = []
eval_avg_log = []
eval_winner_log = []
eval_lengths_log = []


def save_statistics():
    with open(f"./stats/SAC_{args.env_name}-gamma{args.gamma}-tau{args.tau}-lr{args.lr}-alpha{args.alpha}-autotune{args.automatic_entropy_tuning}-pera{args.replay_alpha}-perb{args.replay_beta}-seed{args.seed}-stat.pkl", 'wb') as f:
        pickle.dump({'critic_1_loss_log': critic_1_loss_log,
                     'critic_2_loss_log': critic_2_loss_log,
                     'policy_loss_log': policy_loss_log,
                     'entropy_loss_log': entropy_loss_log,
                     'alpha_log': alpha_log,
                     'rewards_log': rewards_log,
                     'winner_log': winner_log,
                     'lengths_log': lengths_log,
                     'eval_avg_log': eval_avg_log,
                     'eval_winner_log': eval_winner_log,
                     'eval_lengths_log': eval_lengths_log}, f)


# Memory
memory = PrioritizedReplayBuffer(
    state_size=env.observation_space.shape[0], action_size=num_actions, buffer_size=args.replay_size, eps=1e-2, alpha=args.replay_alpha, beta=args.replay_beta)

# Training Loop
total_numsteps = 0
updates = 0

opponent = set_opponent(0, basic=True)

for i_episode in range(1, args.max_episodes+1):
    if i_episode <= episodes_against_basic:
        pass
    elif i_episode <= episodes_mixed:
        # switch every CHECKPOINT_INTERVAL steps
        if (i_episode - 1 - episodes_against_basic) % (MIXED_FREQ * 2) == 0:
            opponent = set_opponent(i_episode, basic=False)
        elif (i_episode - 1 - episodes_against_basic) % (MIXED_FREQ * 2) == MIXED_FREQ:
            opponent = set_opponent(i_episode, basic=True)
    else:
        if (i_episode - 1 - episodes_mixed - episodes_against_basic) % (ADVANCED_SELF_TRAIN + ADVANCED_BASIC) == 0:
            opponent = set_opponent(i_episode, basic=False)
        elif (i_episode - 1 - episodes_mixed - episodes_against_basic) % (ADVANCED_SELF_TRAIN + ADVANCED_BASIC) == ADVANCED_SELF_TRAIN:
            opponent = set_opponent(i_episode, basic=True)

    episode_reward = 0
    episode_steps = 0
    critic_1_losses = []
    critic_2_losses = []
    policy_losses = []
    entropy_losses = []
    done = False
    obs, info = env.reset()
    obs_opponent = env.obs_agent_two()

    for t in range(args.max_timesteps):
        if args.start_steps > total_numsteps:
            agent_action = env.action_space.sample(
            )[:num_actions]  # Sample random action
        else:
            agent_action = agent.select_action(
                obs)  # Sample action from policy
        opponent_action = opponent.act(obs_opponent)

        if memory.real_size > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha, td_errors, tree_idxs = agent.update_parameters(
                    memory, args.batch_size, updates)

                critic_1_losses.append(critic_1_loss)
                critic_2_losses.append(critic_2_loss)
                policy_losses.append(policy_loss)
                entropy_losses.append(ent_loss)
                updates += 1
                memory.update_priorities(tree_idxs, td_errors)

        next_obs, reward, done, trunc, info = env.step(
            np.hstack([agent_action, opponent_action]))  # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if (env_has_max_steps and episode_steps == env._max_episode_steps) else float(
            not done)

        memory.add(obs, agent_action, reward, next_obs,
                   mask)  # Append transition to memory

        obs = next_obs
        obs_opponent = env.obs_agent_two()

        if done or trunc:
            break

    rewards_log.append(episode_reward)
    winner = info["winner"]
    winner_log.append(winner)
    lengths_log.append(t+1)
    try:
        alpha_log.append(alpha)
        critic_1_loss_log.append(np.mean(critic_1_losses))
        critic_2_loss_log.append(np.mean(critic_2_losses))
        policy_loss_log.append(np.mean(policy_losses))
        entropy_loss_log.append(np.mean(entropy_losses))
    except:
        print("First episode: No updates, only transition collection.")

    if winner == 1:
        w = "W"
    elif winner == 0:
        w = "D"
    else:
        w = "L"

    print(f"{i_episode} ({w})")

    if i_episode % EVAL_INTERVAL == 0 and args.eval is True:
        avg_reward = 0.
        for _ in range(EVAL_EPS):
            obs, info = env.reset()
            obs_opponent = env.obs_agent_two()
            episode_reward = 0
            done = False
            for t in range(args.max_timesteps):
                agent_action = agent.select_action(obs, evaluate=True)
                opponent_action = opponent.act(obs_opponent)
                next_obs, reward, done, trunc, info = env.step(
                    np.hstack([agent_action, opponent_action]))
                episode_reward += reward

                obs = next_obs
                obs_opponent = env.obs_agent_two()

                if done or trunc:
                    break
            avg_reward += episode_reward
            eval_winner_log.append(info["winner"])
            eval_lengths_log.append(t+1)
        avg_reward /= EVAL_EPS

        eval_avg_log.append((avg_reward, i_episode))

        print("----------------------------------------")
        print(f"Last {EVAL_INTERVAL} Episodes:")
        print(
            f"Avg. Reward: {round(np.mean(rewards_log[-EVAL_INTERVAL:]), 2)}")
        print(
            f"Wins: {sum(1 for x in winner_log[-EVAL_INTERVAL:] if x == 1)}, Draws: {sum(1 for x in winner_log[-EVAL_INTERVAL:] if x == 0)}, Losses: {sum(1 for x in winner_log[-EVAL_INTERVAL:] if x == -1)} ")
        print(
            f"Avg. Episode Length: {round(np.mean(lengths_log[-EVAL_INTERVAL:]), 2)}, Min: {np.min(lengths_log[-EVAL_INTERVAL:])}, Max: {np.max(lengths_log[-EVAL_INTERVAL:])}")
        print("--------")
        print(f"{EVAL_EPS} Test Episodes:")
        print(f"Avg. Reward: {round(avg_reward, 2)}")
        print(
            f"Wins: {sum(1 for x in eval_winner_log[-EVAL_EPS:] if x == 1)}, Draws: {sum(1 for x in eval_winner_log[-EVAL_EPS:] if x == 0)}, Losses: {sum(1 for x in eval_winner_log[-EVAL_EPS:] if x == -1)} ")
        print(
            f"Avg. Episode Length: {round(np.mean(eval_lengths_log[-EVAL_EPS:]), 2)}, Min: {np.min(eval_lengths_log[-EVAL_EPS:])}, Max: {np.max(eval_lengths_log[-EVAL_EPS:])}")
        print("----------------------------------------")

        # save every {CHECKPOINT_INTERVAL} episodes
        if i_episode % CHECKPOINT_INTERVAL == 0:
            print("########## Saving a checkpoint... ##########")
            agent.save_checkpoint(
                args.env_name, "", f"checkpoints/SAC_{args.env_name}_{i_episode}-gamma{args.gamma}-tau{args.tau}-lr{args.lr}-alpha{args.alpha}-autotune{args.automatic_entropy_tuning}-pera{args.replay_alpha}-perb{args.replay_beta}-seed{args.seed}.pth")
            save_statistics()

save_statistics()
env.close()
