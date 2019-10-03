import random
import gym
import h5py
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from agents.maxlist import MaxList
import agents.serializer as serializer

ENV_NAME = "CartPole-v1"
GAMMA = 0.95
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000000
BATCH_SIZE = 20

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.995

class DQNetwork:
    def __init__(self, nbr_input_parameters, action_space, model):
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        if model is None:
            self.exploration_rate = EXPLORATION_MAX
            self.model = Sequential()
            self.model.add(Dense(24, input_shape=(
                nbr_input_parameters,), activation="relu"))
            self.model.add(Dense(24, activation="relu"))
            self.model.add(Dense(self.action_space, activation="linear"))
        else:
            self.exploration_rate = EXPLORATION_MIN
            self.model = model
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space)
        q_values = self.model.predict(state)
        return np.argmax(q_values[0])

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA *
                            np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def cartpole(model=None):
    env = gym.make(ENV_NAME)
    nbr_input_parameters = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNetwork(nbr_input_parameters, action_space, model)
    past_steps = MaxList(100)
    for run in range(200):
        state = env.reset()
        state = np.reshape(state, [1, nbr_input_parameters])
        for step in range(500):
            env.render()
            action = dqn_solver.act(state)
            state_next, reward, terminal, info = env.step(action)
            reward = reward if not terminal else -reward
            state_next = np.reshape(state_next, [1, nbr_input_parameters])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal:
                past_steps.append(step+1)
                print("Episode {} finished after {} timesteps with average {}".format(
                    run, step+1, np.average(past_steps.get())))
                break
            dqn_solver.experience_replay()
        
    # Save learnings when training
    if model is None:
        agent_state = serializer.save_model("cartpole-dqn.json", dqn_solver.model)

if __name__ == "__main__":
    # cartpole()
    cartpole(serializer.load_model("cartpole-dqn.json"))
