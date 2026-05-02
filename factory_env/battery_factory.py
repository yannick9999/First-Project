import gymnasium as gym
import numpy as np
from gymnasium import spaces

class BatteryFactoryEnv(gym.Env):
    """
    A simple battery factory simulator.
    One production line with 4 stages:
    0 - Cell Assembly
    1 - Electrolyte Filling
    2 - Sealing
    3 - Quality Check
    """

    def __init__(self):
        super().__init__()

        # 3 possible actions: 0 = continue, 1 = slow down, 2 = maintenance stop
        self.action_space = spaces.Discrete(3)

        # Observation: [current_stage, machine_health, battery_quality]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0]),
            high=np.array([3, 100, 100]),
            dtype=np.float32
        )

        # Internal state
        self.current_stage = 0
        self.machine_health = 100.0
        self.battery_quality = 100.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_stage = 0
        self.machine_health = 100.0
        self.battery_quality = 100.0

        observation = np.array([
            self.current_stage,
            self.machine_health,
            self.battery_quality
        ], dtype=np.float32)

        return observation, {}
    
    def step(self, action):
        # Apply action effects
        if action == 0:  # continue
            self.machine_health -= 5
        elif action == 1:  # slow down
            self.machine_health -= 2
            self.battery_quality -= 3
        elif action == 2:  # maintenance stop
            self.machine_health = min(100, self.machine_health + 20)
            self.battery_quality -= 10

        # Advance to next stage
        self.current_stage += 1

        # Calculate reward
        if self.machine_health <= 0:
            reward = -50
            terminated = True
        elif self.current_stage >= 4:
            reward = self.battery_quality
            terminated = True
        else:
            reward = 0
            terminated = False

        observation = np.array([
            self.current_stage,
            self.machine_health,
            self.battery_quality
        ], dtype=np.float32)

        return observation, reward, terminated, False, {}
    