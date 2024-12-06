# PPO
import matplotlib.pyplot as plt
import gymnasium as gym
import time
import numpy as np
from env.FoxGooseEnv import FoxGooseEnv
from models.PPO import PPO
import torch

num_episodes = 100
gamma = 0.9
actor_lr = 1e-3
critic_lr = 1e-2
epsilon = 0.2
eps = 0.2
lmdba = 0.95
epochs = 100
n_state = 33
n_hidden = 128
device = "cuda"
reward_list = []

print("Environment initialization")
env = FoxGooseEnv()
n_state = 33
n_fox_action = env.fox_action_space.n
n_goose_action = env.goose_action_space.n

fox_ppo = PPO(n_state, n_hidden, n_fox_action, epochs,
              actor_lr, critic_lr, lmdba, eps, gamma, epsilon, device)
goose_ppo = PPO(n_state, n_hidden, n_goose_action, epochs,
                actor_lr, critic_lr, lmdba, eps, gamma, epsilon, device)

for episode in range(num_episodes):
    print(f"Episode {episode+1} / {num_episodes}")
    state = env.reset()
    done = False
    fox_reward = 0
    goose_reward = 0

    fox_transition_dict = {
        "states": [],
        "actions": [],
        "rewards": [],
        "next_states": [],
        "dones": []
    }
    goose_transition_dict = {
        "states": [],
        "actions": [],
        "rewards": [],
        "next_states": [],
        "dones": []
    }

    round_cnt = 0
    winner = None
    while not done:
        if episode % 20 == -1:
            print(f"episode {episode}, round {round_cnt}")
            env.render()
            time.sleep(0.05)
        else:
            print(f"Round {round_cnt}", end="\r")
        round_cnt += 1
        if round_cnt > 1000:
            print("Round limit exceeded")
            break
        role = env.role
        if role == "fox":
            if np.random.rand() < 0.5:
                action = fox_ppo.get_action_random(
                    env.fox_action_space, env.fox_mask)
            else:
                action = fox_ppo.get_action(
                    state, env.fox_action_space, env.fox_mask)
            next_state, reward, done, winner = env.step(action)
            fox_transition_dict["states"].append(state)
            fox_transition_dict["actions"].append(action)
            fox_transition_dict["rewards"].append(reward)
            fox_transition_dict["next_states"].append(next_state)
            fox_transition_dict["dones"].append(done)
            fox_reward += reward
        elif role == "goose":
            if np.random.rand() < 0.5:
                action = goose_ppo.get_action_random(
                    env.goose_action_space, env.goose_mask)
            else:
                action = goose_ppo.get_action(state, env.goose_action_space,
                                              env.goose_mask)
            next_state, reward, done, winner = env.step(action)
            goose_transition_dict["states"].append(state)
            goose_transition_dict["actions"].append(action)
            goose_transition_dict["rewards"].append(reward)
            goose_transition_dict["next_states"].append(next_state)
            goose_transition_dict["dones"].append(done)
            goose_reward += reward
        else:
            raise ValueError("invalid role")
        state = next_state
    if round_cnt > 1000:
        continue
    print("winner", winner)
    if winner["winner"] == "fox":
        print("\nFox training")
        fox_ppo.train(fox_transition_dict)
    elif winner["winner"] == "goose":
        print("Goose training")
        goose_ppo.train(goose_transition_dict)
    else:
        raise ValueError("invalid winner")
    reward_list.append({"fox": fox_reward, "goose": goose_reward})
    print(
        f"Episode {episode}, fox reward: {fox_reward}, goose reward: {goose_reward}")

# Plot the reward curve

fox_reward_list = [reward["fox"] for reward in reward_list]
goose_reward_list = [reward["goose"] for reward in reward_list]

plt.plot(fox_reward_list, label="fox")
plt.plot(goose_reward_list, label="goose")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.legend()
plt.show()
