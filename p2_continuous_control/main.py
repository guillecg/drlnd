import datetime
from pathlib import Path

import numpy as np

import torch

from collections import deque

import matplotlib.pyplot as plt

from unityagents import UnityEnvironment

from p2_continuous_control.agents.agent_td3 import AgentTD3


SEED = 42
SCORE_TARGET = 30.0
SCORE_WINDOW = 100

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# set seeds
torch.manual_seed(SEED)
np.random.seed(SEED)

# create folder architecture
PROJECT = 'p2_continuous_control'
START_TIME = datetime.datetime.now().strftime('%m-%d-%Y_%Hh%Mm')
EXPERIMENT_FOLDER = f'{PROJECT}/experiments/{START_TIME}'
Path(EXPERIMENT_FOLDER).mkdir(parents=True, exist_ok=False)


if __name__ == '__main__':
    env_path = f'{PROJECT}/Reacher_Linux/Reacher.x86_64'
    env = UnityEnvironment(file_name=env_path)

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print(f'There are {states.shape[0]} agents')
    print(f'Each observes a state with length: {state_size}')
    print('The state for the first agent looks like:', states[0])

    # define the agent
    agent = AgentTD3(
        state_size=state_size,
        action_size=action_size,
        hyperparams=dict(),
        device=DEVICE,
        seed=SEED
    )

    # training hyperparameters
    n_episodes = 250   # maximum number of training episodes
    max_t = 1000       # maximum number of timesteps per episode

    scores = []                                 # scores for each episode
    scores_window = deque(maxlen=SCORE_WINDOW)  # last 100 scores
    scores_window_means = []                    # average max scores for each episode

    # training loop
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = env_info.vector_observations              # get the current stat
        agent.reset()                                      # initialize agents
        score = np.zeros(num_agents)                       # initialize scores

        for t in range(max_t):
            actions = agent.select_action(states)       # get the action from the agent
            env_info = env.step(actions)[brain_name]    # send the action to the environment
            next_states = env_info.vector_observations  # get the next state
            rewards = env_info.rewards                  # get the reward
            dones = env_info.local_done                 # see if episode has finished

            # save experience tuple (of each agent) into replay buffer
            for i_agent in range(0, num_agents):
                experience = (
                    states[i_agent],
                    actions[i_agent],
                    rewards[i_agent],
                    next_states[i_agent],
                    dones[i_agent]
                )
                agent.memory.add(data=experience)

            states = next_states  # roll over states to next time step
            score += rewards      # update the scores

            # Train each agent
            agent.learn_batch(timestep=t)

            if np.any(dones):
                break

        # Score is updated for each agent, therefore use mean as an estimate
        score = np.mean(score)

        scores.append(score)
        scores_window.append(score)

        window_score_mean = np.mean(scores_window)  # save mean of window scores
        scores_window_means.append(window_score_mean)

        print(
            '\rEpisode {}\tEpisode total score: {:.2f}\tWindow Score: {:.2f}'
            .format(i_episode, score, window_score_mean),
            end=""
        )

        if i_episode % 100 == 0:
            print(
                '\rEpisode {}\tWindow Score: {:.2f}'
                .format(i_episode, window_score_mean)
            )

        if window_score_mean >= SCORE_TARGET:
            print(
                '\nEnvironment solved in {:d} episodes!\tWindow Score: {:.2f}'
                .format(i_episode, window_score_mean)
            )

            print(f'Saving weights into {EXPERIMENT_FOLDER} folder...')
            torch.save(
                agent.actor_local.state_dict(),
                f'{EXPERIMENT_FOLDER}/weights_actor_episode_{i_episode}.pth'
            )
            torch.save(
                agent.critic_local.state_dict(),
                f'{EXPERIMENT_FOLDER}/weights_critic_episode_{i_episode}.pth'
            )

            break

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.plot(np.arange(1, len(scores) + 1), scores_window_means, label='mean')
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

    # save figure to file
    print(f'Saving figure into {EXPERIMENT_FOLDER} folder...')
    fig.savefig(f'{EXPERIMENT_FOLDER}/scores.png')

    # close the environment
    env.close()
