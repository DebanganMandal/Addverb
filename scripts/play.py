import gymnasium as gym
import numpy as np
from tqdm import tqdm
from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import torch

env = gym.make('Ant-v5', render_mode = "human")

# print(env.action_space)

# env = gym.make(
#     'Ant-v5',
#     xml_file='~/Desktop/Addverb/examples/quadruped_simulation/go1_quadruped/go1.xml',
#     forward_reward_weight=1,
#     ctrl_cost_weight=0.05,
#     contact_cost_weight=5e-4,
#     healthy_reward=1,
#     main_body=1,
#     healthy_z_range=(0.195, 0.75),
#     include_cfrc_ext_in_observation=True,
#     exclude_current_positions_from_observation=False,
#     reset_noise_scale=0.1,
#     frame_skip=25,
#     max_episode_steps=1000,
#     render_mode = "human"
# )

"""model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
# model.save("sac_ant")
# model = SAC.load("sac_ant")
obs, info = env.reset()
done = False

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()"""

# gym_vec_env = gym.make_vec("Ant-v5", num_envs=2, vectorization_mode="async")
sb3_vec_env = make_vec_env("Ant-v5", n_envs=2)
# print(gym_vec_env)
# print(sb3_vec_env)

# Train and save the model
# Saving the model is divided into 2 parts: one for RL parameters(data), and another for NN weights(parameters), acc. to the documentation

model = PPO("MlpPolicy", sb3_vec_env, verbose=1)
model.learn(total_timesteps=25000)

# Save model
# model.save("ppo_ant")

# Load the saved model
# model = PPO.load("ppo_ant_copy.zip", env=sb3_vec_env)

obs = sb3_vec_env.reset()
action, _states = model.predict(obs)

# print(model.action_space)
# print(action)


# print(obs)
# print(sb3_vec_env.reset_infos)
done = False

while True:
    action, _states = model.predict(obs)
    obs, rewards, done, info = sb3_vec_env.step(action)
    sb3_vec_env.render("")
sb3_vec_env.close()
