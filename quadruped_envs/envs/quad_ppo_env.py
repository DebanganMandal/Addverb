import gymnasium as gym
import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os
from typing import Dict, Tuple, Union

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class QuadrupedWalkPPO(MujocoEnv, utils.EzPickle):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"], "render_fps": 100}

    def __init__(
        self,
        frame_skip: int = 5,
        forward_reward_weight: float = 1,
        ctrl_cost_weight: float = 0.05,
        contact_cost_weight: float = 5e-4,
        healthy_reward: float = 1.0,
        main_body: Union[int, str] = 1,
        terminate_when_unhealthy: bool = True,
        healthy_z_range: Tuple[float, float] = (0.2, 1.0),
        contact_force_range: Tuple[float, float] = (-1.0, 1.0),
        include_cfrc_ext_in_observation: bool = True,
        exclude_current_positions_from_observation: bool = False,
        reset_noise_scale: float = 0.1,
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        **kwargs,
        ):

        utils.EzPickle.__init__(
            self,
            frame_skip,
            forward_reward_weight,
            ctrl_cost_weight,
            contact_cost_weight,
            healthy_reward,
            main_body,
            terminate_when_unhealthy,
            healthy_z_range,
            contact_force_range,
            include_cfrc_ext_in_observation,
            exclude_current_positions_from_observation,
            reset_noise_scale,
            **kwargs
            )
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._healthy_reward = healthy_reward
        self._main_body = main_body
        self._terminate_when_unhealthy = terminate_when_unhealthy
        
        self._healthy_z_range = healthy_z_range
        self._exclude_current_positions_from_observation = (exclude_current_positions_from_observation)
        self._include_cfrc_ext_in_observation = include_cfrc_ext_in_observation

        self._contact_force_range = contact_force_range

        self._main_body = main_body
        
        observation_space = Box(-np.inf, np.inf, (115,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            os.path.normpath(os.path.join(os.path.dirname(__file__), "../../go1_quadruped/go1.xml")),
            frame_skip,
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )


        self.observation_structure = {
            "skipped_qpos": 2*exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size,
            "qvel": self.data.qvel,
            "cfrc_ext": self.data.cfrc_ext[1:].size*include_cfrc_ext_in_observation,
        }

    @property
    def contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(np.square(self.contact_forces))
        return contact_cost
    
    @property
    def healthy_reward(self):
        return self.is_healthy * self._healthy_reward
    
    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy
    
    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action):
        # Simulation
        xy_position_before = self.data.body(self._main_body).xpos[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.body(self._main_body).xpos[:2].copy()

        # Update new x and y velocities
        xy_velocity = (xy_position_after - xy_position_before)/self.dt
        x_velocity, y_velocity = xy_velocity

        # Get observation
        observation = self._get_obs()
        
        # Get rewards
        forward_reward = x_velocity * self._forward_reward_weight
        healthy_reward = self.healthy_reward
        
        rewards = forward_reward + healthy_reward

        # Get costs
        control_cost = self.control_cost(action)
        contact_cost = self.contact_cost
        costs = control_cost + contact_cost

        # Find total reward
        reward = rewards - costs

        reward_info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -control_cost,
            "reward_contact": -contact_cost,
            "reward_survive": healthy_reward,
        }

        # Check termination
        terminated = (not self.is_healthy) and self._terminate_when_unhealthy

        if self.render_mode == "human":
            self.render()

        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            **reward_info,
        }

        return observation, reward, terminated, False, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = (self.init_qvel + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv))

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
    
    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        if self._include_cfrc_ext_in_observation:
            contact_force = self.contact_forces[1:].flatten()
            return np.concatenate((position, velocity, contact_force))
        else: return np.concatenate((position, velocity))
       
        
        