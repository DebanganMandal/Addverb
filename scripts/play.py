import gymnasium as gym
import numpy as np
from tqdm import tqdm
from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch

# env = gym.make('Ant-v5', render_mode = "human")

# print(env.action_space)

env = gym.make(
    'Ant-v5',
    xml_file='~/Desktop/Addverb/examples/ant_mujoco_rl/go1_quadruped/go1.xml',
    forward_reward_weight=1,
    ctrl_cost_weight=0.05,
    contact_cost_weight=5e-4,
    healthy_reward=1,
    main_body=1,
    healthy_z_range=(0.195, 0.75),
    include_cfrc_ext_in_observation=True,
    exclude_current_positions_from_observation=False,
    reset_noise_scale=0.1,
    frame_skip=25,
    max_episode_steps=1000,
    render_mode = "human"
)

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
# model.save("sac_ant")
# model = SAC.load("sac_ant")
obs, info = env.reset()
done = False

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

"""vec_env = make_vec_env("Ant-v5", n_envs=4)
print(vec_env)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=25000)
# model.save("ppo_ant")
# model = PPO.load("ppo_ant")
obs, info = vec_env.reset()
done = False

while True:
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = vec_env.step(action)
    vec_env.render("human") """


