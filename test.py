# PPO
import matplotlib.pyplot as plt
import gymnasium as gym
import time
import numpy as np
from env.FoxGooseEnv import FoxGooseEnv
from models.PPO import PPO
import torch
import os
import json


gamma = 0.9
actor_lr = 1e-3
critic_lr = 1e-2
epsilon = 0.2
eps = 0.2
lmdba = 0.95
epochs = 500
n_state = 33
n_hidden = 128
device = "cuda" if torch.cuda.is_available() else "cpu"
reward_list = []

model_episode = 3400
runs = 5

test_mode = {
    "fox": "weight",
    "goose": "random"
}


print("Environment initialization")
env = FoxGooseEnv()
n_state = 33
n_fox_action = env.fox_action_space.n
n_goose_action = env.goose_action_space.n

fox_ppo = PPO(n_state, n_hidden, n_fox_action, epochs,
              actor_lr, critic_lr, lmdba, eps, gamma, epsilon, device)
goose_ppo = PPO(n_state, n_hidden, n_goose_action, epochs,
                actor_lr, critic_lr, lmdba, eps, gamma, epsilon, device)

fox_ppo.load(f"runs/{runs}/models/{model_episode}/fox")
goose_ppo.load(f"runs/{runs}/models/{model_episode}/goose")

start_time = time.time()

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
    print(f"round {round_cnt}")
    env.render()
    time.sleep(0.04)
    round_cnt += 1
    if round_cnt > 1000:
        print("Round limit exceeded")
        break
    role = env.role
    if role == "fox":
        if test_mode["fox"] == "weight":
            action = fox_ppo.get_action(
                state, env.fox_action_space, env.fox_mask)
            for i in range(len(env.fox_mask)):
                if env.fox_mask[i] == 1:
                    m = env._fox_action_to_move[i]
                    if max(abs(m[0]), abs(m[1])) > 1:
                        action = i
        else:
            action = fox_ppo.get_action_random(
                env.fox_action_space, env.fox_mask)
        next_state, reward, done, winner = env.step(action)
        fox_transition_dict["states"].append(state)
        fox_transition_dict["actions"].append(action)
        fox_transition_dict["rewards"].append(reward)
        fox_transition_dict["next_states"].append(next_state)
        fox_transition_dict["dones"].append(done)
        fox_reward += reward
    elif role == "goose":
        if test_mode["goose"] == "weight":
            action = goose_ppo.get_action(state, env.goose_action_space,
                                          env.goose_mask)
        else:
            action = goose_ppo.get_action_random(
                env.goose_action_space, env.goose_mask)
        # action = goose.get_action(state)
        # raise ValueError("goose should not be random")
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
print("winner", winner)

fox_reward /= round_cnt
goose_reward /= round_cnt
reward_list.append(
    {"fox": fox_reward, "goose": goose_reward, "round": round_cnt})
# print(
#     f"Episode {episode}, fox reward: {fox_reward}, goose reward: {goose_reward}, round: {round_cnt}")

# Plot the reward curve

# fox_reward_list = [reward["fox"] for reward in reward_list]
# goose_reward_list = [reward["goose"] for reward in reward_list]

# plt.plot(fox_reward_list, label="fox")
# plt.plot(goose_reward_list, label="goose")
# plt.xlabel("Episode")
# plt.ylabel("Reward")
# plt.legend()
# plt.show()
