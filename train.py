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

save_threshold = 200
num_episodes = 30000
gamma = 0.9
actor_lr = 1e-3
critic_lr = 1e-2
epsilon = 0.2
eps = 0.2
lmdba = 0.95
epochs = 300
n_state = 33
n_hidden = 128
device = "cuda"
reward_list = []

# save path
if os.path.exists("runs") == False:
    os.mkdir("runs")
path_dir = 0
while os.path.exists(f"runs/{path_dir}") == True:
    path_dir += 1
os.mkdir(f"runs/{path_dir}")
print(f"Save path: runs/{path_dir}")
os.mkdir(f"runs/{path_dir}/models")
os.mkdir(f"runs/{path_dir}/transition")

with open(f"runs/{path_dir}/config.txt", "w") as f:
    f.write(f"time: {time.ctime()}\n")
    f.write(f"num_episodes: {num_episodes}\n")
    f.write(f"gamma: {gamma}\n")
    f.write(f"actor_lr: {actor_lr}\n")
    f.write(f"critic_lr: {critic_lr}\n")
    f.write(f"epsilon: {epsilon}\n")
    f.write(f"eps: {eps}\n")
    f.write(f"lmdba: {lmdba}\n")
    f.write(f"epochs: {epochs}\n")
    f.write(f"n_state: {n_state}\n")
    f.write(f"n_hidden: {n_hidden}\n")
    f.write(f"device: {device}\n")
    f.close()

print("Environment initialization")
env = FoxGooseEnv()
n_state = 33
n_fox_action = env.fox_action_space.n
n_goose_action = env.goose_action_space.n

fox_ppo = PPO(n_state, n_hidden, n_fox_action, epochs,
              actor_lr, critic_lr, lmdba, eps, gamma, epsilon, device)
goose_ppo = PPO(n_state, n_hidden, n_goose_action, epochs,
                actor_lr, critic_lr, lmdba, eps, gamma, epsilon, device)

start_time = time.time()

for episode in range(num_episodes):
    print(
        f"Episode {episode+1} / {num_episodes} - Time(hh:mm:ss): {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
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
        # if episode % 200 == 0 and episode > num_episodes / 2:
        #     print(f"episode {episode}, round {round_cnt}")
        #     env.render()
        #     time.sleep(0.05)
        # else:
        #     print(f"Round {round_cnt}", end="\r")
        round_cnt += 1
        if round_cnt > 2000:
            print("Round limit exceeded")
            break
        role = env.role
        if role == "fox":
            if episode % save_threshold == 0 and episode > num_episodes / 2:
                action = fox_ppo.get_action(
                    state, env.fox_action_space, env.fox_mask)
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
            if episode % save_threshold == 0 and episode > num_episodes / 2:
                action = goose_ppo.get_action(state, env.goose_action_space,
                                              env.goose_mask)
            else:
                action = goose_ppo.get_action_random(
                    env.goose_action_space, env.goose_mask)
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
    if round_cnt > 2000:
        continue
    # print("winner", winner)

    # train
    fox_ppo.train(fox_transition_dict)
    goose_ppo.train(goose_transition_dict)

    # save dict
    np.save(f"runs/{path_dir}/transition/{episode}_fox.json",
            fox_transition_dict)
    np.save(f"runs/{path_dir}/transition/{episode}_goose.json",
            goose_transition_dict)

    fox_reward /= round_cnt
    goose_reward /= round_cnt
    reward_list.append(
        {"fox": fox_reward, "goose": goose_reward, "round": round_cnt})
    # print(
    #     f"Episode {episode}, fox reward: {fox_reward}, goose reward: {goose_reward}, round: {round_cnt}")
    if episode % save_threshold == 0:
        print("Save models")
        os.mkdir(f"runs/{path_dir}/models/{episode}")
        os.mkdir(f"runs/{path_dir}/models/{episode}/fox")
        fox_ppo.save(f"runs/{path_dir}/models/{episode}/fox")
        os.mkdir(f"runs/{path_dir}/models/{episode}/goose")
        goose_ppo.save(f"runs/{path_dir}/models/{episode}/goose")
        print("Models saved: ", episode)

# Plot the reward curve

# fox_reward_list = [reward["fox"] for reward in reward_list]
# goose_reward_list = [reward["goose"] for reward in reward_list]

# plt.plot(fox_reward_list, label="fox")
# plt.plot(goose_reward_list, label="goose")
# plt.xlabel("Episode")
# plt.ylabel("Reward")
# plt.legend()
# plt.show()
