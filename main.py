import gym
import keras.layers
import tensorflow as tf
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannGumbelQPolicy
from rl.memory import SequentialMemory

def buildModel(observations, actions):
    return Sequential([
        Flatten(input_shape=(1,observations)),
        Dense(24, activation='relu'),
        Dense(24, activation='relu'),
        Dense(actions, activation='softmax'),
    ])

def buildAgent(model, actions):
    policy = BoltzmannGumbelQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model = model, memory = memory, policy = policy, nb_actions = actions, nb_steps_warmup=100, target_model_update=1e-2)
    
    return dqn


env = gym.make('CartPole-v1')

observations = env.observation_space.shape[0]
actions = env.action_space.n

model = buildModel(observations, actions)
agent = buildAgent(model, actions)


agent.compile(Adam(lr=0.001), metrics=['mse'])
agent.fit(env, nb_steps=10000, visualize=False)

agent.test(env, nb_episodes=100, visualize=True)



