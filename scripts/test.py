from stable_baselines3 import PPO
from quadruped_envs import QuadrupedWalkPPO
import cv2
import imageio

vec_env = QuadrupedWalkPPO(render_mode="rgb_array")
model = PPO.load("ppo_quadruped.zip")

obs, info = vec_env.reset()
frames = []

for _ in range(500):
    action, _stages = model.predict(obs)
    obs, reward, done, truncated, info = vec_env.step(action)
    image = vec_env.render()
    if _%5 == 0:
        frames.append(image)

    if done or truncated:
        obs, info = vec_env.reset()

# with imageio.get_writer("../media/test.gif", mode="I") as writer:
#     for idx, frame in enumerate(frames):
#         writer.append_data(frame)