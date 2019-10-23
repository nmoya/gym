from gym import envs
import gym, math
from gym import wrappers
import numpy as np
from collections import deque

# Observation
# idx 	Observation 	Min 	Max
# 0 	Cart Position - 2.4 	2.4
# 1 	Cart Velocity - Inf 	Inf
# 2 	Pole Angle 	~ -41.8° 	~ 41.8°
# 3 	Pole Velocity At Tip - Inf 	Inf

# Actions
# Num 	Action
# 0 	Push cart to the left
# 1 	Push cart to the right

# Reward
# Reward is 1 for every step taken, including the termination step

# Episode Termination
# Pole Angle is more than ±12°
# Cart Position is more than ±2.4 (center of the cart reaches the edge of the display)
# Episode length is greater than 200

# Try with keyboard: 1 Push cart to the left, 2 Push cart to the right
# python examples/agents/keyboard_agent.py CartPole-v0

class QLearn():
  def __init__(self, observation_space, action_space):
    self.states_per_obs = (1, 1, 6, 12)
    self.gamma = 1.0 # discount factor [0, 1]
    self.alpha = 0.1  # learning rate [0, 1]
    self.observation_space = observation_space
    self.action_space = action_space
    self.min_epsilon = 0.1
    self.epsilon = 1.0
    self.epsilon_decay = 0.9995
    # creates a 5-d matrix to store all 72 states with 2 actions each:
    # self.Q[0][0][0][0] => [0, 0]
    self.Q = np.zeros(self.states_per_obs + (action_space.n,))

  def normalize(self, value, lower_bound, upper_bound):
    # Returns a value within [-1, 1]
    return (value + abs(lower_bound)) / (upper_bound - lower_bound)

  def linear_transform(self, value, max_value):
    # raw_value is a value within [-max_value, max_value]
    raw_value = int(round((max_value - 1) * value))
    # Will return a number within [0, max_value-1]
    return min((max_value - 1), max(0, raw_value))

  def discretize(self, obs):
    upper_bounds = [self.observation_space.high[0], 0.5,
                    self.observation_space.high[2], math.radians(50)]
    lower_bounds = [self.observation_space.low[0], -0.5,
                    self.observation_space.low[2], -math.radians(50)]
    ratios = [self.normalize(obs[i], lower_bounds[i], upper_bounds[i])
              for i in range(len(obs))]
    new_obs = [self.linear_transform(
        ratios[i], self.states_per_obs[i]) for i in range(len(obs))]
    return tuple(new_obs)

  def decide_action(self, state, episode):
    # argmax = index with max value in an array
    # since the each index represents an action
    # so the index with max value is the best action
    if (np.random.random() <= self.epsilon):
      action = self.action_space.sample()
    else:
      action = np.argmax(self.Q[state])
    self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
    return action

  def alpha_fn(self, x):
    return max(self.alpha, min(
        1.0, 1.0 - math.log10((x+1) / 25)))

  def update(self, state, action, reward, new_state, done, episode):
    value_next_state = np.max(self.Q[new_state])
    alpha = self.alpha_fn(episode)
    self.Q[state][action] = self.Q[state][action] + \
      alpha * (reward + self.gamma * value_next_state - self.Q[state][action])

def cartpole():
  # outdir = './videos/'
  env = gym.make('CartPole-v0')
  # env = wrappers.Monitor(env, outdir, force=True)
  agent = QLearn(env.observation_space, env.action_space)
  past_rewards = deque(maxlen=100)
  for i_episode in range(500):
    state = agent.discretize(env.reset())
    current_reward = 0
    for t in range(200):
      # if i_episode > 300:
      # env.render()
      action = agent.decide_action(state, i_episode)
      new_state, reward, done, _ = env.step(action)
      new_state = agent.discretize(new_state)
      agent.update(state, action, reward, new_state, done, i_episode)
      state = new_state
      current_reward += reward
      if done:
        break
    
    past_rewards.append(current_reward)
    print("Eps: {:04d} Reward: {:3.2f} Avg-Reward: {:3.2f} Explo: {:.3f}".format(
        i_episode, current_reward, np.average(past_rewards), agent.epsilon))
  env.close()

cartpole()
