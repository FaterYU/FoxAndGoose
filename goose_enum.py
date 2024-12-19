import matplotlib.pyplot as plt
import time
import numpy as np
from env.FoxGooseEnv import FoxGooseEnv
from env.GridRule import GridRule
import copy
import os
import json

start_time = time.time()

# 枚举

# 定义树结构


class Node:
    def __init__(self, state=None, parent=None):
        self.state = state  # goose还没下
        self.parent = parent
        self.next_visit_state = {}
        self.reward = 0


grid_rule = GridRule()
env = FoxGooseEnv()


def get_fox_location(state):
    for i in range(7):
        for j in range(7):
            if state[i][j] == 'F':
                return (i, j)


def get_goose_location(state):
    locations = []
    for i in range(7):
        for j in range(7):
            if state[i][j] == 'G':
                locations.append((i, j))
    return locations


def state2str(state):
    s = ''
    for i in range(7):
        for j in range(7):
            s += state[i][j]
    return s


def str2state(s):
    state = []
    for i in range(7):
        state.append(list(s[i*7:i*7+7]))
    return state


def expand_fox_prob_step(node, state):
    node_list = []
    fox_location = get_fox_location(state)
    mask = grid_rule.get_fox_mask(state, fox_location)
    win = True
    for i in range(len(mask)):
        if mask[i] == 1:
            win = False
            action = env._fox_action_to_move[i]
            if max(abs(action[0]), abs(action[1])) > 1:
                return None
            new_state = copy.deepcopy(state)
            new_state[fox_location[0]][fox_location[1]] = '.'
            new_state[fox_location[0] +
                      action[0]][fox_location[1]+action[1]] = 'F'
            if check_exist(new_state):
                new_node = htable[state2str(new_state)]
            else:
                new_node = Node(new_state, node)
                htable[state2str(new_state)] = new_node
            node_list.append(new_node)
    if win:
        new_node = Node(None, node)
        node_list.append(new_node)
        htable[state2str(new_state)] = new_node
        back_propagation(new_node)
        raise Exception('win')
    return node_list


def back_propagation(node):
    while node is not None:
        node.reward += 1
        node = node.parent


def expand_goose_prob_step(node):
    goose_locations = get_goose_location(node.state)
    goose_mask = grid_rule.get_goose_mask(node.state, goose_locations)
    for i in range(len(goose_mask)):
        if goose_mask[i] == 1:
            move = env._goose_action_to_move[i]
            move_id = list(move.keys())[0]
            action = move[move_id]
            # 不向下走
            if action[0] > 0:
                continue
            new_state = copy.deepcopy(node.state)
            new_state[goose_locations[move_id][0]
                      ][goose_locations[move_id][1]] = '.'
            new_state[goose_locations[move_id][0] +
                      action[0]][goose_locations[move_id][1]+action[1]] = 'G'
            node_list = expand_fox_prob_step(node, new_state)
            if node_list is not None:
                node.next_visit_state[i] = node_list
    return node


def check_exist(state):
    return state2str(state) in htable


# 初始化
root = Node()
htable = {}

# 生成树
stack = expand_fox_prob_step(root, env.state)
while len(stack) > 0:
    n = stack.pop()
    n = expand_goose_prob_step(n)
    for i in n.next_visit_state:
        for j in n.next_visit_state[i]:
            if j.state is not None and j.next_visit_state == {}:
                stack.append(j)
    env.state = n.state
    env.render()
    print(len(htable), len(stack), root.reward)
    # time.sleep(0.1)
