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

# Play with keyboard: 
# python examples/agents/keyboard_agent.py LunarLander-v2

# Solution example:
# python gym/envs/box2d/lunar_lander.py

# register(
#     id='LunarLander-v2',
#     max_episode_steps=1000,
#     reward_threshold=200,
# )

# register(
#     id='LunarLanderContinuous-v2',
#     max_episode_steps=1000,
#     reward_threshold=200,
# )

ENV_NAME = "LunarLander-v2"
GAMMA = 0.99
LEARNING_RATE = 0.0001

MEMORY_SIZE = 250000
BATCH_SIZE = 32

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.99941

class DQNetwork:
    def __init__(self, nbr_input_parameters, action_space, model, target_model):
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)
        self.training = model is None
        self.state_size = nbr_input_parameters

        if model is None:
            self.exploration_rate = EXPLORATION_MAX
            self.model = self.create_network(nbr_input_parameters, action_space)
            self.target_model = self.create_network(nbr_input_parameters, action_space)
        else:
            self.exploration_rate = 0
            self.model = model
            self.target_model = target_model
        self.model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))
        self.target_model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

    def create_network(self, size_input, size_output):
        model = Sequential()
        model.add(Dense(32, input_shape=(size_input,), activation="relu"))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(size_output, activation="linear"))
        return model

    def save(self, model_name, target_model_name):
        serializer.save_model(model_name, self.model)
        serializer.save_model(target_model_name, self.target_model)

    def reshape(self, s):
        return np.reshape(s, [1, self.state_size])

    def remember(self, state, action, reward, next_state, done):
        if self.training:
            self.memory.append((self.reshape(state), action,
                                reward, self.reshape(next_state), done))

    def choose_action(self, state):
        if not self.training or np.random.rand() > self.exploration_rate:
            q_values = self.model.predict(self.reshape(state))
            return np.argmax(q_values[0])
        return np.random.choice(self.action_space)

    def optimize(self):
        if len(self.memory) < BATCH_SIZE or not self.training:
            return
        batch = np.array(random.sample(self.memory, BATCH_SIZE))
        s = np.vstack(batch[:, 0])
        a = np.array(batch[:, 1], dtype=int)
        r = np.array(batch[:, 2], dtype=float)
        s_ = np.vstack(batch[:, 3])
        non_terminal_states = np.where(batch[:, 4] == False)
        if len(non_terminal_states[0]) > 0: # The selected batch has states that are non terminal
            a_ = np.argmax(self.model.predict(s_)[non_terminal_states, :][0], axis=1)
            r[non_terminal_states] += np.multiply(
                GAMMA, self.target_model.predict(s_)[non_terminal_states, a_][0])

        y = self.model.predict(s)
        y[range(BATCH_SIZE), a] = r
        self.model.fit(s, y, verbose=0)
    
    def update(self):
        if self.training:
            for i, layer in enumerate(self.model.layers):
                self.target_model.layers[i].set_weights(layer.get_weights())

            self.exploration_rate *= EXPLORATION_DECAY
            self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def lander(model=None, target_model=None):
    env = gym.make(ENV_NAME)
    nbr_input_parameters = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNetwork(nbr_input_parameters,
                           action_space, model, target_model)
    past_rewards = deque(maxlen=100)
    for run in range(5000):
        state = env.reset()
        current_reward = 0
        for step in range(1000):
            # if not dqn_solver.training:
            env.render()
            action = dqn_solver.choose_action(state)
            state_next, reward, terminal, info = env.step(action)
            current_reward += reward

            dqn_solver.remember(state, action, reward, state_next, terminal)
            dqn_solver.optimize()

            state = state_next
            if terminal or current_reward < -250:
                break

        dqn_solver.update()
        past_rewards.append(current_reward)
        print("Eps: {:04d} Steps: {:04d} Reward: {:3.2f} Avg-Reward: {:3.2f} Explo: {:.3f}".format(
            run, step+1, current_reward, np.average(past_rewards), dqn_solver.exploration_rate))
        
    # Save learnings when training
    if dqn_solver.training:
        dqn_solver.save("lunar-lander-dqn.json", "lunar-lander-dqn-target.json")


if __name__ == "__main__":
    # lander()
    lander(serializer.load_model("lunar-lander-dqn.json"),
           serializer.load_model("lunar-lander-dqn-target.json"))
