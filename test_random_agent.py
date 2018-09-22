# python test_agent.py --model checkpoint.pth
import argparse
import sys
import os

from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from dqn_agent import Agent

if __name__ == '__main__':
    
    env = UnityEnvironment(file_name="VisualBanana.app")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents in the environment
    print('Number of agents:', len(env_info.agents))

    # number of actions
    action_size = brain.vector_action_space_size
    print('Number of actions:', action_size)

    # examine the state space 
    state = env_info.visual_observations[0]  
    # import pdb; pdb.set_trace()
    print('States look like:', state)
    print(state.shape)
    plt.imshow(np.squeeze(state))
    plt.show() 
    state_size = len(state)
    print('States have length:', state_size)

    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    state = env_info.visual_observations[0]            # get the current state
    score = 0                                          # initialize the score

    print ("Evaluating agent...")
    while True:
        action = np.random.randint(action_size)  
        env_info = env.step(action)[brain_name]        # send the action to the environment
        next_state = env_info.visual_observations[0]   # get the next state
        reward = env_info.rewards[0]                   # get the reward
        done = env_info.local_done[0]                  # see if episode has finished
        score += reward                                # update the score
        state = next_state                             # roll over the state to next time step
        if done:                                       # exit loop if episode finished
            break
        
    print("Score: {}".format(score))