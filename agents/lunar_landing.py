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
EXPLORATION_DECAY = 0.996


class DQNetwork:
    def __init__(self, nbr_input_parameters, action_space, model):
        self.action_space = action_space
        self.memory = deque(maxlen=MEMORY_SIZE)

        if model is None:
            self.exploration_rate = EXPLORATION_MAX
            self.model = Sequential()
            self.model.add(Dense(32, input_shape=(
                nbr_input_parameters,), activation="relu"))
            self.model.add(Dense(32, activation="relu"))
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
    
    def update(self):
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


def lander(model=None):
    env = gym.make(ENV_NAME)
    nbr_input_parameters = env.observation_space.shape[0]
    action_space = env.action_space.n
    dqn_solver = DQNetwork(nbr_input_parameters, action_space, model)
    past_rewards = MaxList(100)
    for run in range(5000):
        state = env.reset()
        state = np.reshape(state, [1, nbr_input_parameters])
        cum_reward = 0
        for step in range(1000):
            # if not model is None:
            # env.render()
            action = dqn_solver.act(state)
            state_next, env_reward, terminal, info = env.step(action)
            cum_reward += env_reward
            reward = env_reward if not terminal else -env_reward
            state_next = np.reshape(state_next, [1, nbr_input_parameters])
            dqn_solver.remember(state, action, reward, state_next, terminal)
            state = state_next
            if terminal or cum_reward < -250:
                past_rewards.append(env_reward)
                print("Eps: {:04d} Steps: {:04d} Reward: {:3.2f} Avg-Reward: {:3.2f} Explo: {:.3f}".format(
                    run, step+1, env_reward, np.average(past_rewards.get()), dqn_solver.exploration_rate))
                break
            # if step % 4 == 0:
            dqn_solver.experience_replay()
        dqn_solver.update()

    # Save learnings when training
    if model is None:
        agent_state = serializer.save_model(
            "lunar-lander-dqn.json", dqn_solver.model)


if __name__ == "__main__":
    lander()
    # lander(serializer.load_model("lunar-lander-dqn.json"))
