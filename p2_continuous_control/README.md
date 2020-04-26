
# Project 2: Continuous Control

For this project, we will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment:

![](https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif)
# Project Details
## Task description
- **Goal**: a double-jointed arm has to follow, with its blue end, a green ball that moves around the screen.
- **Reward**: a reward of +0.1 is provided for each step in which the blue end of the arm is within the goal location (green ball).
- **Desired behaviour**: the agent should maximize its reward by following the green ball.
- **Conditions for success**: the task is considered solved when the agent gets an average reward (without discount) of +30 over 100 consecutive episodes.

## Environment description
- **Observation space**: 33 variables corresponding to position, rotation velocity, and angular velocities of the arm.
- **Action space**: each action is a vector of four real numbers, between -1 and 1, indicating the corresponding torque to apply to each of the two joints.

## Additional details
### Multi-agent
The environment provided can be solved using two different versions of the same: one with just one agent, and another one with consists in 20 different agents. In the latter case, the average of all agents is taken into account for the reward.

# Getting Started
1. Download the environment from one of the links below, select the environment that matches your operating system:  
  
   - **_Version 1: One (1) Agent_**  
     - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)  
      - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)  
      - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)  
      - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)  
  
   - **_Version 2: Twenty (20) Agents_**  
     - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)  
      - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)  
      - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)  
      - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)  
      
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.  
  
    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)  
  
2. Place the file in the DRLND GitHub repository, in the `p2_continuous-control/` folder, and unzip (or decompress) the file.

# Instructions
## Project Architecture
```bash
├── agents
│   ├── agent.py     # algorithm implementation
├── experiments
├── models
│   ├── networks.py  # model networks
│   ├── utils.py     # replay buffer, noise class
├── (environment folder)
├── main.py
├── README.md
├── Report.md
```
## Training
The procedure with results and future work can be found in the [Report.ipynb](https://github.com/guillecg/drlnd/blob/master/p2_continuous_control/Report.ipynb) notebook.

Additionally, all the process is summarized in the [main.py](https://github.com/guillecg/drlnd/blob/master/p2_continuous_control/main.py) script, which performs the following tasks when called:
- Creates the **experiment folder** (naming it with the current date and time), where the final plot of scores and all the weights for the successful models will be saved.
- Instantiates the **agent** that is going to be trained.
- Performs the **training** of the agent, following the common structure of a nested loop, one for episodes and another for timesteps.
- **Saves the weights of successful agents** for each epoch.
- **Plots the evolution of scores** after the training process.

