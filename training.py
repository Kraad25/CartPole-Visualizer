from stable_baselines3 import PPO
from Environment import CartPoleEnv
from stable_baselines3.common.env_checker import check_env

env = CartPoleEnv()
check_env(env)  # Optional: checks Gym compliance

try:
    model = PPO.load("ppo_cartpole", env=env)
    print("Loaded existing model")
except:
    model = PPO("MlpPolicy", env, verbose=1)
    print("Training new model")

model.learn(total_timesteps=100)
model.save("ppo_cartpole")