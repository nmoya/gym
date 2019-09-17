import gym, time
import numpy as np

class QLearn():
    def __init__(self, nbr_states, nbr_actions):
        self.gamma = 0.95 # discount factor [0, 1]
        self.alpha = 0.8 # learning rate [0, 1]
        self.Q = np.zeros([nbr_states, nbr_actions])
        self.nbr_states = nbr_states
        self.nbr_actions = nbr_actions

    def decide_action(self, state, episode):
        # argmax = index with max value in an array
        # since the each index represents an action
        # so the index with max value is the best action
        noise = np.random.randn(1, self.nbr_actions) * (1. / (episode + 1))
        action_values = self.Q[state, :]
        return np.argmax(action_values + noise)

    def update(self, state, action, reward, new_state):
        value_new_state = np.max(self.Q[new_state, :])
        self.Q[state, action] = self.Q[state, action] + \
            self.alpha * (reward + self.gamma * value_new_state - self.Q[state, action])


env = gym.make('FrozenLake-v0')
agent = QLearn(env.observation_space.n, env.action_space.n)
success = steps = reward = 0
nbr_episodes = 10
for i_episode in range(nbr_episodes):
    state = env.reset()
    for t in range(100):
        # env.render()
        action = agent.decide_action(state, i_episode)
        new_state, reward, done, _ = env.step(action)
        agent.update(state, action, reward, new_state)
        state = new_state
        if done:
            if reward == 1.0:
                success += 1
                steps += t+1
                # print("[SUCCESS] ep: {} steps: {}".format(i_episode, t+1))
            break
print("Success: ", success)
print("AVG Steps: ", steps / success)
env.close()
