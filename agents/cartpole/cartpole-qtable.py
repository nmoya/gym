import gym, math, time
from gym import wrappers
import numpy as np

class QLearn():
  def __init__(self, observation_space, action_space):
    self.buckets = (1, 1, 6, 12)
    self.gamma = 1.0  # discount factor [0, 1]
    self.min_alpha = 0.1  # learning rate [0, 1]
    self.observation_space = observation_space
    self.action_space = action_space
    self.nbr_states = env.observation_space.shape[0]
    self.nbr_actions = action_space.n
    # creates a 5-d matrix to store all 72 states with 2 actions each:
    # self.Q[0][0][0][0] => [0, 0]
    self.Q = np.zeros(self.buckets + (action_space.n,))

  def discretize(self, obs):
    upper_bounds = [self.observation_space.high[0], 0.5,
                    self.observation_space.high[2], math.radians(50)]
    lower_bounds = [self.observation_space.low[0], -0.5,
                    self.observation_space.low[2], -math.radians(50)]
    ratios = [(obs[i] + abs(lower_bounds[i])) /
              (upper_bounds[i] - lower_bounds[i]) for i in range(len(obs))]
    new_obs = [int(round((self.buckets[i] - 1) * ratios[i]))
              for i in range(len(obs))]
    new_obs = [min(self.buckets[i] - 1, max(0, new_obs[i]))
              for i in range(len(obs))]
    # discretize all the atributes within [0, bucket[i]]
    return tuple(new_obs)

  def decide_action(self, state, episode):
    # argmax = index with max value in an array
    # since the each index represents an action
    # so the index with max value is the best action
    epsilon = max(
        0.1, min(1, 1.0 - math.log10((episode + 1) / 25)))
    return self.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.Q[state])

  def update(self, state, action, reward, new_state, episode):
    value_new_state = np.max(self.Q[new_state])
    alpha = max(self.min_alpha, min(
        1.0, 1.0 - math.log10((episode + 1) / 25)))
    self.Q[state][action] = self.Q[state][action] + \
      alpha * (reward + self.gamma * value_new_state - self.Q[state][action])


outdir = './videos/'
env = gym.make('CartPole-v0')
# env = wrappers.Monitor(env, outdir, force=True)
agent = QLearn(env.observation_space, env.action_space)
for i_episode in range(500):
  # time.sleep(0.5)
  state = agent.discretize(env.reset())
  for t in range(200):
    env.render()
    action = agent.decide_action(state, i_episode)
    new_state, reward, done, _ = env.step(action)
    new_state = agent.discretize(new_state)
    agent.update(state, action, reward, new_state, i_episode)
    state = new_state
    if t == 199:
      print("WON!!!", i_episode)
    if done:
      print("Episode {} finished after {} timesteps".format(i_episode, t+1))
      break
env.close()
