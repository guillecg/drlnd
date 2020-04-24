# Adapted from: https://github.com/udacity/deep-reinforcement-learning

import datetime
from pathlib import Path

import numpy as np

import torch

from collections import deque

import matplotlib.pyplot as plt

from unityagents import UnityEnvironment

from p1_navigation.agents.agent_dqn import AgentDQN


SEED = 42
SCORE_TARGET = 30.0
SCORE_WINDOW = 100

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)

# create folder architecture
PROJECT = 'p1_navigation'
START_TIME = datetime.datetime.now().strftime('%m-%d-%Y %Hh%Mm')
EXPERIMENT_FOLDER = f'{PROJECT}/experiments/{START_TIME}'
Path(EXPERIMENT_FOLDER).mkdir(parents=True, exist_ok=False)


if __name__ == '__main__':
    env_path = f'{PROJECT}/Banana_Linux/Banana.x86_64'
    env = UnityEnvironment(file_name=env_path)

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
    state = env_info.vector_observations[0]
    print('States look like:', state)
    state_size = len(state)
    print('States have length:', state_size)

    # define the agent
    agent = AgentDQN(
        state_size=state_size,
        action_size=action_size,
        hyperparams=dict(),
        device=DEVICE,
        seed=SEED
    )

    # training hyperparameters
    n_episodes = 2000  # maximum number of training episodes
    max_t = 1000       # maximum number of timesteps per episode
    eps_start = 1.0    # starting value of epsilon, for epsilon-greedy action selection
    eps_end = 0.01     # minimum value of epsilon
    eps_decay = 0.995  # multiplicative factor (per episode) for decreasing epsilon
    eps = eps_start    # initialize epsilon

    scores = []                                 # scores for each episode
    scores_window = deque(maxlen=SCORE_WINDOW)  # last 100 scores

    # training loop
    for i_episode in range(1, n_episodes + 1):
        state = env_info.vector_observations[0]  # get the current state
        score = 0                                # reset score

        for t in range(max_t):
            action = agent.select_action(state, eps)      # get the action from the agent
            env_info = env.step(action)[brain_name]       # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]                  # get the reward
            done = env_info.local_done[0]                 # see if episode has finished

            # save sarsd tuple into replay buffer
            agent.step(state, action, reward, next_state, done)

            state = next_state  # update current state with next state
            score += reward     # update score with reward

            if done:
                env.reset(train_mode=True)[brain_name]
                break

        scores_window.append(score)          # save most recent score
        scores.append(score)                 # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        print(
            '\rEpisode {}\tAverage Score: {:.2f}'
            .format(i_episode, np.mean(scores_window)),
            end=""
        )

        if i_episode % 100 == 0:
            print(
                '\rEpisode {}\tAverage Score: {:.2f}'
                .format(i_episode, np.mean(scores_window))
            )

        if np.mean(scores_window) >= SCORE_TARGET:
            print(
                '\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'
                .format(i_episode, np.mean(scores_window))
            )

            print(f'Saving weights into {EXPERIMENT_FOLDER} folder...')
            torch.save(
                agent.qnetwork_local.state_dict(),
                f'{EXPERIMENT_FOLDER}/weights_epoch_{i_episode}.pth'
            )

            break

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    # save figure to file
    print(f'Saving figure into {EXPERIMENT_FOLDER} folder...')
    fig.savefig(f'{EXPERIMENT_FOLDER}/scores.png')

    # close the environment
    env.close()
