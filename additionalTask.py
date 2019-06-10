import numpy as np
from physics_sim import PhysicsSim

class additionalTask():
    def __init__(self, target_pos, init_pose=None, init_velocities=None, init_angle_velocities=None,
                 runtime=5.):
                
        self.target_pos = target_pos
        hovering = 500
        #Setting low and high for the actions
        self.action_high = 1.2 * hovering
        self.action_low = 0.8 * hovering
        self.actionB = (self.action_high+self.action_low)/2.0
        self.actionM = (self.action_high-self.action_low)/2.0
        self.action_size = 1
        
        #Init the velocities to blank defaults if not given to us
        if(init_velocities is None):
            init_velocities = np.array([0.0, 0.0, 0.0])
        # Here we create a physics simulation for the task
        # print("start", init_pose, init_velocities, init_angle_velocities)
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.state_size = len(self.get_state())

    def get_reward(self):
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        #return reward
        # Using Huber Loss
        huberLoss = (self.sim.pose[2]-self.target_pos[2]) ** 2 + (0.1 * (self.sim.linear_accel[2]**2))
        return(np.maximum(1.0 - 0.5 * 0.5 * (np.sqrt(1 + (huberLoss / 0.5) ** 2) - 1), 0))

    def get_state(self):
        return np.array([ (self.sim.pose[:3] - self.target_pos)[2], self.sim.v[2], self.sim.linear_accel[2] ])

    def convert(self, action):
        #Using the tanh constants
        return (action * self.actionM) + self.actionB

    def step(self, action):
        rotorSpeed = self.convert(action)
        done = self.sim.next_timestep(rotorSpeed*np.ones(4))
        if (self.get_reward()) <= 0:
            done = True
        return self.get_state(), self.get_reward(), done

    def reset(self):
        self.sim.reset()
        return self.get_state()