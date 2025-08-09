from stable_baselines3 import PPO
from quadruped_envs import QuadrupedWalkPPO
import imageio, numpy as np

env = QuadrupedWalkPPO(render_mode="rgb_array")  # critical for off-screen
model = PPO.load("ppo_quadruped.zip", device="cpu")

obs, info = env.reset(seed=0)
frames = []

try:
    for t in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # request a specific size to avoid zero-dim contexts on some systems
        frame = env.render()  # Gymnasium’s mujoco env respects render_mode
        if frame is not None and t % 5 == 0:
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            frames.append(frame)

        if terminated or truncated:
            obs, info = env.reset()
finally:
    env.close()  # prevents those OffScreenViewer.__del__ EGL errors

imageio.mimsave("../media/test.gif", frames, duration=0.05)
