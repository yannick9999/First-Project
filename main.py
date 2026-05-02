from factory_env import BatteryFactoryEnv

env = BatteryFactoryEnv()

observation, info = env.reset()
print(f"Initial observation: {observation}")

for step in range(4):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Step {step + 1} | Action: {action} | Observation: {observation} | Reward: {reward} | Terminated: {terminated}")

    if terminated:
        break