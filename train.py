from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.env_checker import check_env
from factory_env import BatteryFactoryEnv
import matplotlib.pyplot as plt

env = BatteryFactoryEnv()
check_env(env)
print("Environment check passed!")

def train_agent(algorithm, env, timesteps=10000):
    model = algorithm("MlpPolicy", env, verbose=0)
    model.learn(total_timesteps=timesteps)
    return model

def evaluate_agent(model, env, episodes=100):
    total_rewards = []
    
    for episode in range(episodes):
        obs, info = env.reset()
        episode_reward = 0
        terminated = False
        
        while not terminated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
        
        total_rewards.append(episode_reward)
    
    return total_rewards

if __name__ == "__main__":
    env = BatteryFactoryEnv()
    
    print("Training DQN...")
    dqn_model = train_agent(DQN, env, timesteps=10000)
    
    print("Training PPO...")
    ppo_model = train_agent(PPO, env, timesteps=10000)
    
    print("Training A2C...")
    a2c_model = train_agent(A2C, env, timesteps=10000)
    
    print("Evaluating all agents...")
    dqn_rewards = evaluate_agent(dqn_model, env)
    ppo_rewards = evaluate_agent(ppo_model, env)
    a2c_rewards = evaluate_agent(a2c_model, env)
    
    print(f"DQN  - Average reward: {sum(dqn_rewards)/len(dqn_rewards):.2f}")
    print(f"PPO  - Average reward: {sum(ppo_rewards)/len(ppo_rewards):.2f}")
    print(f"A2C  - Average reward: {sum(a2c_rewards)/len(a2c_rewards):.2f}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(dqn_rewards, label="DQN", alpha=0.7)
    plt.plot(ppo_rewards, label="PPO", alpha=0.7)
    plt.plot(a2c_rewards, label="A2C", alpha=0.7)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Algorithm Comparison - Battery Factory")
    plt.legend()
    plt.tight_layout()
    plt.savefig("comparison.png")
    print("Plot saved as comparison.png")