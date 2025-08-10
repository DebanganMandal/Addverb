# Quadruped Simulation using OpenAI Gym and Mujoco
This repo contains codes updated, for my Research Assitantship tenure at Addverb Robotics. I have tried to create a custom gym environment for Quadruped walking simulation, using openai gym api and simulating it on mujoco simulator. I have, till now, used the stable_baselines3 PPO module for quadruped training. You can put your own robot MFCF file in go1_quadruped folder. I have used go1 unitree MJCF file from Mujoco Menagerie repo. <br>

Clone this repo by:
`git clone https://github.com/DebanganMandal/Addverb.git`

## Installations required
1. python>=3.10
2. Anaconda
3. open gym api: `pip install gym` after that, download mujoco package for gym: `pip install gym[mujoco]`
4. Install mujoco (version 3.2.5 or higher)
5. Install stable_baselines3: `pip install stable-baselines3`
   
## How to clone and run
1. Clone the repo in the folder of your choice (Recommended to create a separate conda environment for the project)
2. In the same directory, run `pip install -e .`
3. Check `quadruped_envs` present in `pip list`
4. Change directory to scripts, run `python vec_learn.py`, and you will notice a zip file by the name `ppo_quadruped.zip`, which contains the NN weights and te RL parameters
5. Run `python test.py` and check the test.gif for rendered frames of your trained quadruped

## Ouput VIDEO
<!-- ![Sample Ouput GIF](./scripts/runs/quad_ppo/videos/quad_walk-step-0-to-step-2000.mp4) -->
   
### References
1. https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_go1
2. https://gymnasium.farama.org/
3. https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html
4. https://github.com/nimazareian/quadruped-rl-locomotion/tree/main