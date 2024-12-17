from stable_baselines3.common.env_checker import check_env
from quadruped_envs import QuadrupedWalkPPO
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

env = QuadrupedWalkPPO(render_mode="rgb_array")
check_env(env)
ppo_vec_env = make_vec_env(QuadrupedWalkPPO, n_envs=2)
model = PPO("MlpPolicy", ppo_vec_env, verbose=1)
model.learn(total_timesteps=15000)
model.save("ppo_quadruped")
