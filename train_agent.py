# python train_agent.py --episodes 1000 --model checkpoint.pth --plot Score.png
import argparse
from collections import deque
import sys
import os

from unityagents import UnityEnvironment
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch

from dqn_agent import Agent

def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, store_model='checkpoint.pth'):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        save_model (str): path for storing pytoch model
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        state = env_info.visual_observations[0]  
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            next_state = env_info.visual_observations[0]
            reward = env_info.rewards[0] 
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end='')
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=15.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores

def plot_scores(scores, rolling_window=100, save_plot='Score.png'):
    """Plot scores and optional rolling mean using specified window."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    rolling_mean = pd.Series(scores).rolling(rolling_window).mean()
    plt.plot(rolling_mean, linewidth=4);
    plt.savefig('Score.png')
    return rolling_mean

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train the agent in the environment')
    parser.add_argument('--episodes', help="How many episodes to train the agent")
    parser.add_argument('--model', default='checkpoint.pth', help="path where the pytorch model should be stored")
    parser.add_argument('--plot', help="path to save the achieved training score of the agent")

    options = parser.parse_args(sys.argv[1:])
    
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
    # print('States look like:', state)

    state_size = state.shape
    print('States have shape:', state_size)
    in_size = state_size[-1]

    agent = Agent(state_size=in_size, action_size=action_size, seed=0, mode='dueling')

    scores = dqn(n_episodes=int(options.episodes), store_model=options.model)

    plot_scores(scores, rolling_window=100, save_plot=options.plot)
    