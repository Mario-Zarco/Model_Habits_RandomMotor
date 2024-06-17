import numpy as np

'''
@author Mario Zarco email: mzar066@aucklanduni.ac.nz or zarco.mario.al@gmail.com 
'''

min_env = -2.0
max_env = 2.0
width_env = 4.0


class Agent:

    def __init__(self, x=0.0, seed=123456789):
        self.x = x
        self.s = 0.0
        self.m = 0.0
        np.random.seed(seed)

    def sensor_update(self, d):
        self.s = 1.0 / (1.0 + 1.0*d**2)

    # The robot's velocity is equivalent to the state of its motor
    def step(self, stepsize):
        new_x = self.x + stepsize*self.m
        if new_x < min_env:
            new_x = new_x + width_env
        elif new_x > max_env:
            new_x = new_x - width_env
        self.x = new_x

    # Methods to randomize parameters.

    def randomize_position(self):
        self.x = np.random.uniform(min_env, max_env)

    def randomize_motor(self):
        self.m = np.random.uniform(-1, 1)

    def random_position_between(self, x_min, x_max):
        self.x = np.random.uniform(x_min, x_max)

    # Set and get methods

    def set_motor(self, m):
        self.m = m

    def get_motor_value(self):
        return self.m

    def get_sensor_value(self):
        return self.s

    def set_seed(self, new_seed):
        np.random.seed(new_seed)

    def reset_agent(self):
        self.x = 0.0
        self.s = 0.0
        self.m = 0.0
