import numpy as np
from agents.Actor import Actor
from agents.Critic import Critic
from ReplayBuffer import ReplayBuffer
from agents.OUNoise import OUNoise

class DDPG():
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        #Policy Model & Value Model
        self.actorLocal = Actor(self.state_size, self.action_size, self.action_low, self.action_high)
        self.criticLocal = Critic(self.state_size, self.action_size)
        self.actorTarget = Actor(self.state_size, self.action_size, self.action_low, self.action_high)   
        self.criticTarget = Critic(self.state_size, self.action_size)

        #Initializing target model with local model params
        self.criticTarget.model.set_weights(self.criticLocal.model.get_weights())
        self.actorTarget.model.set_weights(self.actorLocal.model.get_weights())
        
        #Replay Buffer
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        self.noise = OUNoise(self.action_size, 0, 0.1, 0.25)
        self.discountGamma = 0.9

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
        self.memory.add(self.last_state, action, reward, next_state, done)
        if len(self.memory) > self.batch_size:
            exp = self.memory.sample()
            self.learn(exp)
        self.last_state = next_state

    def act(self, state):
        state = np.reshape(state, [-1, self.state_size])
        action = self.actorLocal.model.predict(state)[0]
        return list(action + self.noise.sample()) 

    def learn(self, exp):
        """
            https://docs.scipy.org/doc/numpy/reference/generated/numpy.vstack.html
            Vertical Stacking of arrays
            This took a long time to get in place :). Thanks to some other references in github too for examples. 
        """
        state = np.vstack([ex.state for ex in exp if ex is not None])
        action = np.array([ex.action for ex in exp if ex is not None]).reshape(-1, self.action_size)
        reward = np.array([ex.reward for ex in exp if ex is not None]).reshape(-1, 1)
        done = np.array([ex.done for ex in exp if ex is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([ex.next_state for ex in exp if ex is not None])

        actions_next = self.actorTarget.model.predict_on_batch(next_states)
        QTargets_next = self.criticTarget.model.predict_on_batch([next_states, actions_next])

        Q_targets = reward + self.discountGamma * QTargets_next * (1 - done)
        self.criticLocal.model.train_on_batch(x=[state, action], y=Q_targets)

        actionGradients = np.reshape(self.criticLocal.get_action_gradients([state, action, 0]), (-1, self.action_size))
        self.actorLocal.train_fn([state, actionGradients, 1]) 

        # Soft-update target models
        self.criticTarget.model.set_weights(0.01 * np.array(self.criticLocal.model.get_weights()) + (1 - 0.01) * np.array(self.criticTarget.model.get_weights()))
        self.actorTarget.model.set_weights(0.01 * np.array(self.actorLocal.model.get_weights()) + (1 - 0.01) * np.array(self.actorTarget.model.get_weights()))