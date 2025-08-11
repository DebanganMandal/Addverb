# quad_ppo_train.py
import os
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, VecVideoRecorder
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, CheckpointCallback, BaseCallback

# ---- import your env ----
from quadruped_envs import QuadrupedWalkPPO

RUN_DIR  = "./runs/quad_ppo"
TB_DIR   = os.path.join(RUN_DIR, "tb")
CKPT_DIR = os.path.join(RUN_DIR, "ckpts")
EVAL_DIR = os.path.join(RUN_DIR, "evals")
VIDEO_DIR= os.path.join(RUN_DIR, "videos")
os.makedirs(CKPT_DIR, exist_ok=True); os.makedirs(EVAL_DIR, exist_ok=True); os.makedirs(VIDEO_DIR, exist_ok=True)

SEED = 42
N_ENVS = 8                 # parallel envs (speed)
TOTAL_STEPS = 1_000_000    # locomotion usually needs 1M+

# --------------- env factories ---------------
def make_train_env():
    # render_mode=None: faster
    return QuadrupedWalkPPO(render_mode=None)

def make_eval_env(render_mode=None):
    return QuadrupedWalkPPO(render_mode=render_mode)

# --------------- TRAIN env (Vec + Monitor + Normalize) ---------------
train_env = make_vec_env(make_train_env, n_envs=N_ENVS, seed=SEED)
train_env = VecMonitor(train_env)                                   # ep_rew_mean/ep_len_mean into TB
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

# --------------- EVAL env (single) with shared normalization ---------------
eval_env = make_vec_env(make_eval_env, n_envs=1, seed=SEED + 1)
eval_env = VecMonitor(eval_env)
eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)  # true rewards at eval

# keep eval's obs normalization in sync with training
class SyncVecNormCallback(BaseCallback):
    def __init__(self, train_norm: VecNormalize, eval_norm: VecNormalize, verbose=0):
        super().__init__(verbose); self.train_norm = train_norm; self.eval_norm = eval_norm
    def _on_training_start(self): 
        self.eval_norm.obs_rms = self.train_norm.obs_rms
    def _on_rollout_end(self):    
        self.eval_norm.obs_rms = self.train_norm.obs_rms
    def _on_step(self):           
        return True  # continue training

sync_cb = SyncVecNormCallback(train_env, eval_env)

eval_cb = EvalCallback(
    eval_env,
    best_model_save_path=CKPT_DIR,
    log_path=EVAL_DIR,
    eval_freq=10_000 // N_ENVS,     # ~every 10k steps
    n_eval_episodes=5,
    deterministic=True,
    render=False,
)

ckpt_cb = CheckpointCallback(save_freq=100_000 // N_ENVS, save_path=CKPT_DIR, name_prefix="ppo_quad")

# --------------- PPO (solid defaults for locomotion) ---------------
model = PPO(
    "MlpPolicy",
    train_env,
    seed=SEED,
    verbose=1,
    tensorboard_log=TB_DIR,
    n_steps=2048,            # per env → 2048*8 = 16384 rollout
    batch_size=4096,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    learning_rate=3e-4,
    device="cuda",            # MLP fine on CPU; set "cuda" if you want
)

# --------------- TRAIN ---------------
model.learn(
    total_timesteps=TOTAL_STEPS,
    callback=CallbackList([sync_cb, eval_cb, ckpt_cb]),
    tb_log_name="ppo_quadruped_v1",
)

# save artifacts
model.save(os.path.join(CKPT_DIR, "ppo_quadruped_final"))
train_env.save(os.path.join(CKPT_DIR, "vecnormalize_final.pkl"))

# --------------- PLAYBACK & VIDEO (off-screen) ---------------
# Tip (headless): export MUJOCO_GL=egl; ensure nvidia-smi works
play_env = make_vec_env(lambda: make_eval_env("rgb_array"), n_envs=1, seed=SEED + 123)
play_env = VecMonitor(play_env)
# load normalization stats from training
stats_path = os.path.join(CKPT_DIR, "vecnormalize_final.pkl")
play_env = VecNormalize.load(stats_path, play_env)
play_env.training = False
play_env.norm_reward = False

# optional video
try:
    play_env = VecVideoRecorder(
        play_env,
        VIDEO_DIR,
        record_video_trigger=lambda step: step == 0,
        video_length=2000,
        name_prefix="quad_walk",
    )
except Exception:
    print("Video recording disabled (install moviepy to enable).")

# use best checkpoint if present
try:
    policy = PPO.load(os.path.join(CKPT_DIR, "best_model"))
except FileNotFoundError:
    policy = model

obs = play_env.reset()
for _ in range(2000):
    action, _ = policy.predict(obs, deterministic=True)  # SB3 clips to action_space automatically
    if action.ndim == 1:  # guard for (act_dim,) → (1, act_dim)
        action = action[np.newaxis, :]
    obs, rewards, dones, infos = play_env.step(action)
    if dones[0]:
        obs = play_env.reset()

play_env.close()
print("Done. TensorBoard:", TB_DIR)
print("Videos in:", VIDEO_DIR)
