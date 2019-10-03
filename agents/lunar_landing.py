import gym, math, time
from gym import wrappers
import numpy as np

# Play with keyboard: 
# python examples/agents/keyboard_agent.py LunarLander-v2

# Solution example:
# python gym/envs/box2d/lunar_lander.py

env = gym.make("LunarLander-v2")
observation = env.reset()
for _ in range(100):
  env.render()
  action = env.action_space.sample()
  observation, reward, done, info = env.step(action)
  print(observation)

  if done:
    observation = env.reset()
env.close()
