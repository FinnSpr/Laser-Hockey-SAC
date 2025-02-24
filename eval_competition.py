import os
import argparse
from sac import SAC
import hockey.hockey_env as h_env
import numpy as np
import json

NUM_EPS_BASIC_WEAK = 520
NUM_EPS_BASIC_STRONG = 520
NUM_EPS_VS_OTHER_AGENT = 20

MAX_TIMESTEPS = 2000


if not os.path.exists('stats/'):
    os.makedirs('stats/')

args = argparse.Namespace(
    env_name="Hockey",
    policy="Gaussian",
    gamma=0.99,
    tau=0.005,
    lr=0.0003,
    alpha=0.2,
    automatic_entropy_tuning=False,
    seed=123456,
    batch_size=128,
    max_episodes=10000,
    self_play=True,
    max_timesteps=2000,
    hidden_size=256,
    updates_per_step=1,
    start_steps=10000,
    target_update_interval=1,
    replay_size=1000000,
    replay_alpha=0.1,
    replay_beta=0.1,
    cuda=True
)

env = h_env.HockeyEnv()

basic_weak = h_env.BasicOpponent(weak=True)
basic_strong = h_env.BasicOpponent(weak=False)

# alpha-0.0
alpha_0_0_4000 = SAC(env.observation_space.shape[0], env.action_space, args)
alpha_0_0_4000.load_checkpoint(
    f"checkpoints/SAC_Hockey_4000-gamma0.99-tau0.005-lr0.0003-alpha0.0-autotuneFalse-pera0.1-perb0.1-seed280.pth", evaluate=True)
alpha_0_0_11000 = SAC(env.observation_space.shape[0], env.action_space, args)
alpha_0_0_11000.load_checkpoint(
    f"checkpoints/SAC_Hockey_11000-gamma0.99-tau0.005-lr0.0003-alpha0.0-autotuneFalse-pera0.1-perb0.1-seed280.pth", evaluate=True)
alpha_0_0_15000 = SAC(env.observation_space.shape[0], env.action_space, args)
alpha_0_0_15000.load_checkpoint(
    f"checkpoints/SAC_Hockey_15000-gamma0.99-tau0.005-lr0.0003-alpha0.0-autotuneFalse-pera0.1-perb0.1-seed280.pth", evaluate=True)

# alpha-0.05
alpha_0_05_4000 = SAC(env.observation_space.shape[0], env.action_space, args)
alpha_0_05_4000.load_checkpoint(
    f"checkpoints/SAC_Hockey_4000-gamma0.99-tau0.005-lr0.0003-alpha0.05-autotuneFalse-pera0.1-perb0.1-seed132.pth", evaluate=True)
alpha_0_05_11000 = SAC(env.observation_space.shape[0], env.action_space, args)
alpha_0_05_11000.load_checkpoint(
    f"checkpoints/SAC_Hockey_11000-gamma0.99-tau0.005-lr0.0003-alpha0.05-autotuneFalse-pera0.1-perb0.1-seed132.pth", evaluate=True)
alpha_0_05_15000 = SAC(env.observation_space.shape[0], env.action_space, args)
alpha_0_05_15000.load_checkpoint(
    f"checkpoints/SAC_Hockey_15000-gamma0.99-tau0.005-lr0.0003-alpha0.05-autotuneFalse-pera0.1-perb0.1-seed132.pth", evaluate=True)

# alpha-0.2
alpha_0_2_4000 = SAC(env.observation_space.shape[0], env.action_space, args)
alpha_0_2_4000.load_checkpoint(
    f"checkpoints/SAC_Hockey_4000-gamma0.99-tau0.005-lr0.0003-alpha0.2-autotuneFalse-pera0.1-perb0.1-seed781.pth", evaluate=True)
alpha_0_2_11000 = SAC(env.observation_space.shape[0], env.action_space, args)
alpha_0_2_11000.load_checkpoint(
    f"checkpoints/SAC_Hockey_11000-gamma0.99-tau0.005-lr0.0003-alpha0.2-autotuneFalse-pera0.1-perb0.1-seed781.pth", evaluate=True)
alpha_0_2_15000 = SAC(env.observation_space.shape[0], env.action_space, args)
alpha_0_2_15000.load_checkpoint(
    f"checkpoints/SAC_Hockey_15000-gamma0.99-tau0.005-lr0.0003-alpha0.2-autotuneFalse-pera0.1-perb0.1-seed781.pth", evaluate=True)

# alpha-0.5
alpha_0_5_4000 = SAC(env.observation_space.shape[0], env.action_space, args)
alpha_0_5_4000.load_checkpoint(
    f"checkpoints/SAC_Hockey_4000-gamma0.99-tau0.005-lr0.0003-alpha0.5-autotuneFalse-pera0.1-perb0.1-seed505.pth", evaluate=True)
alpha_0_5_11000 = SAC(env.observation_space.shape[0], env.action_space, args)
alpha_0_5_11000.load_checkpoint(
    f"checkpoints/SAC_Hockey_11000-gamma0.99-tau0.005-lr0.0003-alpha0.5-autotuneFalse-pera0.1-perb0.1-seed505.pth", evaluate=True)
alpha_0_5_15000 = SAC(env.observation_space.shape[0], env.action_space, args)
alpha_0_5_15000.load_checkpoint(
    f"checkpoints/SAC_Hockey_15000-gamma0.99-tau0.005-lr0.0003-alpha0.5-autotuneFalse-pera0.1-perb0.1-seed505.pth", evaluate=True)

# alpha-auto
alpha_auto_4000 = SAC(env.observation_space.shape[0], env.action_space, args)
alpha_auto_4000.load_checkpoint(
    f"checkpoints/SAC_Hockey_4000-gamma0.99-tau0.005-lr0.0003-alpha0.2-autotuneTrue-pera0.1-perb0.1-seed51.pth", evaluate=True)
alpha_auto_11000 = SAC(env.observation_space.shape[0], env.action_space, args)
alpha_auto_11000.load_checkpoint(
    f"checkpoints/SAC_Hockey_11000-gamma0.99-tau0.005-lr0.0003-alpha0.2-autotuneTrue-pera0.1-perb0.1-seed51.pth", evaluate=True)
alpha_auto_15000 = SAC(env.observation_space.shape[0], env.action_space, args)
alpha_auto_15000.load_checkpoint(
    f"checkpoints/SAC_Hockey_15000-gamma0.99-tau0.005-lr0.0003-alpha0.2-autotuneTrue-pera0.1-perb0.1-seed51.pth", evaluate=True)

# lr-0.00003
lr_0_00003_4000 = SAC(env.observation_space.shape[0], env.action_space, args)
lr_0_00003_4000.load_checkpoint(
    f"checkpoints/SAC_Hockey_4000-gamma0.99-tau0.005-lr3e-05-alpha0.2-autotuneTrue-pera0.1-perb0.1-seed867.pth", evaluate=True)
lr_0_00003_11000 = SAC(env.observation_space.shape[0], env.action_space, args)
lr_0_00003_11000.load_checkpoint(
    f"checkpoints/SAC_Hockey_11000-gamma0.99-tau0.005-lr3e-05-alpha0.2-autotuneTrue-pera0.1-perb0.1-seed867.pth", evaluate=True)
lr_0_00003_15000 = SAC(env.observation_space.shape[0], env.action_space, args)
lr_0_00003_15000.load_checkpoint(
    f"checkpoints/SAC_Hockey_15000-gamma0.99-tau0.005-lr3e-05-alpha0.2-autotuneTrue-pera0.1-perb0.1-seed867.pth", evaluate=True)

# gamma-0.9
gamma_0_9_4000 = SAC(env.observation_space.shape[0], env.action_space, args)
gamma_0_9_4000.load_checkpoint(
    f"checkpoints/SAC_Hockey_4000-gamma0.9-tau0.005-lr0.0003-alpha0.2-autotuneTrue-pera0.1-perb0.1-seed925.pth", evaluate=True)
gamma_0_9_11000 = SAC(env.observation_space.shape[0], env.action_space, args)
gamma_0_9_11000.load_checkpoint(
    f"checkpoints/SAC_Hockey_11000-gamma0.9-tau0.005-lr0.0003-alpha0.2-autotuneTrue-pera0.1-perb0.1-seed925.pth", evaluate=True)
gamma_0_9_15000 = SAC(env.observation_space.shape[0], env.action_space, args)
gamma_0_9_15000.load_checkpoint(
    f"checkpoints/SAC_Hockey_15000-gamma0.9-tau0.005-lr0.0003-alpha0.2-autotuneTrue-pera0.1-perb0.1-seed925.pth", evaluate=True)

# no-PER
no_PER_4000 = SAC(env.observation_space.shape[0], env.action_space, args)
no_PER_4000.load_checkpoint(
    f"checkpoints/SAC_Hockey_4000-gamma0.99-tau0.005-lr0.0003-alpha0.2-autotuneTrue-pera0.0-perb0.1-seed561.pth", evaluate=True)
no_PER_11000 = SAC(env.observation_space.shape[0], env.action_space, args)
no_PER_11000.load_checkpoint(
    f"checkpoints/SAC_Hockey_11000-gamma0.99-tau0.005-lr0.0003-alpha0.2-autotuneTrue-pera0.0-perb0.1-seed561.pth", evaluate=True)
no_PER_15000 = SAC(env.observation_space.shape[0], env.action_space, args)
no_PER_15000.load_checkpoint(
    f"checkpoints/SAC_Hockey_15000-gamma0.99-tau0.005-lr0.0003-alpha0.2-autotuneTrue-pera0.0-perb0.1-seed561.pth", evaluate=True)

# hard-target
hard_target_4000 = SAC(env.observation_space.shape[0], env.action_space, args)
hard_target_4000.load_checkpoint(
    f"checkpoints/SAC_Hockey_4000-gamma0.99-tau1.0-lr0.0003-alpha0.2-autotuneTrue-pera0.1-perb0.1-seed48.pth", evaluate=True)
hard_target_11000 = SAC(env.observation_space.shape[0], env.action_space, args)
hard_target_11000.load_checkpoint(
    f"checkpoints/SAC_Hockey_11000-gamma0.99-tau1.0-lr0.0003-alpha0.2-autotuneTrue-pera0.1-perb0.1-seed48.pth", evaluate=True)
hard_target_15000 = SAC(env.observation_space.shape[0], env.action_space, args)
hard_target_15000.load_checkpoint(
    f"checkpoints/SAC_Hockey_15000-gamma0.99-tau1.0-lr0.0003-alpha0.2-autotuneTrue-pera0.1-perb0.1-seed48.pth", evaluate=True)


trained_agents = {
    "alpha-0.0_4000": alpha_0_0_4000,
    "alpha-0.0_11000": alpha_0_0_11000,
    "alpha-0.0_15000": alpha_0_0_15000,
    "alpha-0.05_4000": alpha_0_05_4000,
    "alpha-0.05_11000": alpha_0_05_11000,
    "alpha-0.05_15000": alpha_0_05_15000,
    "alpha-0.2_4000": alpha_0_2_4000,
    "alpha-0.2_11000": alpha_0_2_11000,
    "alpha-0.2_15000": alpha_0_2_15000,
    "alpha-0.5_4000": alpha_0_5_4000,
    "alpha-0.5_11000": alpha_0_5_11000,
    "alpha-0.5_15000": alpha_0_5_15000,
    "alpha-auto_4000": alpha_auto_4000,
    "alpha-auto_11000": alpha_auto_11000,
    "alpha-auto_15000": alpha_auto_15000,
    "lr-0.00003_4000": lr_0_00003_4000,
    "lr-0.00003_11000": lr_0_00003_11000,
    "lr-0.00003_15000": lr_0_00003_15000,
    "gamma-0.9_4000": gamma_0_9_4000,
    "gamma-0.9_11000": gamma_0_9_11000,
    "gamma-0.9_15000": gamma_0_9_15000,
    "no-PER_4000": no_PER_4000,
    "no-PER_11000": no_PER_11000,
    "no-PER_15000": no_PER_15000,
    "hard-target_4000": hard_target_4000,
    "hard-target_11000": hard_target_11000,
    "hard-target_15000": hard_target_15000
}


# Points, Wins, D, L
against_basic_weak = {k: [0, 0, 0, 0] for k in trained_agents}
against_basic_strong = {k: [0, 0, 0, 0] for k in trained_agents}

competition_results = {k: [0, 0, 0, 0] for k in trained_agents}

agent_names = list(trained_agents.keys())

for i in range(len(agent_names)):
    key = agent_names[i]
    agent = trained_agents[key]

    print(f"{key} vs. weak basic opponent", flush=True)
    for _ in range(NUM_EPS_BASIC_WEAK):
        obs, info = env.reset()
        obs_agent2 = env.obs_agent_two()
        for _ in range(MAX_TIMESTEPS):
            a1 = agent.select_action(obs, evaluate=True)
            a2 = basic_weak.act(obs_agent2)
            obs, r, d, _, info = env.step(np.hstack([a1, a2]))
            obs_agent2 = env.obs_agent_two()
            if d:
                break
        if info["winner"] == 1:
            against_basic_weak[key][0] += 3
            against_basic_weak[key][1] += 1
        elif info["winner"] == 0:
            against_basic_weak[key][0] += 1
            against_basic_weak[key][2] += 1
        else:
            against_basic_weak[key][3] += 1

    print(f"{key} vs. strong basic opponent", flush=True)
    for _ in range(NUM_EPS_BASIC_STRONG):
        obs, info = env.reset()
        obs_agent2 = env.obs_agent_two()
        for _ in range(MAX_TIMESTEPS):
            a1 = agent.select_action(obs, evaluate=True)
            a2 = basic_strong.act(obs_agent2)
            obs, r, d, _, info = env.step(np.hstack([a1, a2]))
            obs_agent2 = env.obs_agent_two()
            if d:
                break
        if info["winner"] == 1:
            against_basic_strong[key][0] += 3
            against_basic_strong[key][1] += 1
        elif info["winner"] == 0:
            against_basic_strong[key][0] += 1
            against_basic_strong[key][2] += 1
        else:
            against_basic_strong[key][3] += 1

    for j in range(i + 1, len(agent_names)):
        key_2 = agent_names[j]
        agent_2 = trained_agents[key_2]
        assert id(agent) != id(agent_2)

        print(f"{key} vs. {key_2}", flush=True)
        for _ in range(NUM_EPS_VS_OTHER_AGENT):
            obs, info = env.reset()
            obs_agent2 = env.obs_agent_two()
            for _ in range(MAX_TIMESTEPS):
                a1 = agent.select_action(obs, evaluate=True)
                a2 = agent_2.select_action(obs_agent2, evaluate=True)
                obs, r, d, _, info = env.step(np.hstack([a1, a2]))
                obs_agent2 = env.obs_agent_two()
                if d:
                    break
            if info["winner"] == 1:
                competition_results[key][0] += 3
                competition_results[key][1] += 1
                competition_results[key_2][3] += 1
            elif info["winner"] == 0:
                competition_results[key][0] += 1
                competition_results[key_2][0] += 1
                competition_results[key][2] += 1
                competition_results[key_2][2] += 1
            else:
                competition_results[key_2][0] += 3
                competition_results[key_2][1] += 1
                competition_results[key][3] += 1

    results_dict = {
        "against_basic_weak": against_basic_weak,
        "against_basic_strong": against_basic_strong,
        "competition_results": competition_results
    }

    with open("eval_competition_results.txt", "wt") as file:
        json.dump(results_dict, file, indent=4)

print("Finished evaluation, saved results to text file.")
