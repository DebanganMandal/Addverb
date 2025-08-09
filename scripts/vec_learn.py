from stable_baselines3.common.env_checker import check_env
from quadruped_envs import QuadrupedWalkPPO
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

env = QuadrupedWalkPPO(render_mode="rgb_array")
check_env(env)
ppo_vec_env = make_vec_env(QuadrupedWalkPPO, n_envs=2)
eval_callback = EvalCallback(env, best_model_save_path='./logs/', log_path='./logs/', eval_freq=10000, deterministic=True, render=False)
model = PPO("MlpPolicy", ppo_vec_env, verbose=1, tensorboard_log="./tb")
model.learn(total_timesteps=5_000_000, callback=eval_callback, tb_log_name="ppo_quadruped_run1")
model.save("ppo_quadruped")