import numpy as np
from numba import vectorize, float64

'''
Numpy-based implementation of the IDSM controller.
References:
- Egbert M. D., Barandiaran X. E. (2014) Modeling habits as self-sustaining patterns of sensorimotor behavior. 
Frontiers in Human Neuroscience, 8(590). 
- Egbert M. D. (2018) Investigations of an Adaptive and Autonomous Sensorimotor Individual. 
The 2018 Conference on Artificial Life: A Hybrid of the European Conference on Artificial Life (ECAL) 
and the International Conference on the Synthesis and Simulation of Living Systems (ALIFE), 343-350.

@author Mario Zarco email: mzar066@aucklanduni.ac.nz or zarco.mario.al@gmail.com 
'''

kw = -0.0025
kd = 1000.0


@vectorize([float64(float64)])
def weightsf(w):
    if w < -2000.0:
        return 0.0
    elif w > 2000.0:
        return 2.0
    else:
        return 2.0 / (1.0 + np.exp(kw * w))


@vectorize([float64(float64)])
def distancef(norm):
    if norm > 0.6:
        return 0.0
    else:
        return 2.0 / (1.0 + np.exp(kd * (norm ** 2)))


class IDSM:

    def __init__(self, motor_dim, sensor_dim, stepsize, seed=123456789):
        self.motor_dim = motor_dim
        self.dimension = motor_dim + sensor_dim
        self.stepsize = stepsize
        self.max_nodes = 10000
        time_delay = int(1.0/stepsize)
        self.kt = 1.0
        self.kdeg = -1.0
        self.kreinf = 10.0

        self.w = np.zeros((self.max_nodes, 1))
        self.Np = np.zeros((self.max_nodes, self.dimension))
        self.Nv = np.zeros((self.max_nodes, self.dimension))
        # self.SM = np.zeros((self.max_nodes, self.dimension))

        self.Nw = np.zeros((self.max_nodes, 1))
        self.Nd = np.zeros((self.max_nodes, 1))

        self.n_nodes = 0  # added nodes
        self.a_nodes = 0  # activated nodes
        self.delay = np.full((self.max_nodes, 1), time_delay)

        np.random.seed(seed)
        self.k_ro = 10.0
        self.k_R = 2.0
        self.k_p = 0.02
        # self.R = np.random.normal(0, np.sqrt(2))
        # self.ro = np.random.normal(0, np.sqrt(10))
        self.ro = self.k_ro * np.random.randn(self.motor_dim)
        self.R = self.k_R * np.random.randn(self.motor_dim)

        self.rand_flag = False

    def weights_distances(self, Np):
        self.Nw[0:self.n_nodes] = weightsf(self.w[0:self.n_nodes])
        # self.SM[0:self.n_nodes] = np.full((self.n_nodes, 1), Np)
        SM = np.full((self.n_nodes, self.dimension), Np)
        norm = np.linalg.norm(self.Np[0:self.n_nodes] - SM, axis=1, keepdims=True)
        self.Nd[0:self.n_nodes] = distancef(norm)

    def density(self, nodes):
        return np.sum(self.Nw[0:nodes] * self.Nd[0:nodes])

    def add_one_node(self, Np, Nv):
        self.Np[self.n_nodes] = Np
        self.Nv[self.n_nodes] = Nv
        self.n_nodes += 1
        self.a_nodes += 1
        # self.weights_distances(Np)

    def add_node(self, Np, Nv):
        self.weights_distances(Np)
        if self.density(self.n_nodes) < self.kt:
            self.Np[self.n_nodes] = Np
            self.Nv[self.n_nodes] = Nv
            self.n_nodes += 1
            return True
        else:
            return False

    def weights_update(self):
        self.w[0:self.n_nodes] += self.stepsize * (self.kdeg + self.kreinf * self.Nd[0:self.n_nodes])
        self.delay[self.a_nodes:self.n_nodes] -= 1
        # self.a_nodes = (self.delay[0:self.n_nodes] == 0).sum()
        self.a_nodes = np.count_nonzero(self.delay[0:self.n_nodes] == 0)

    def gamma(self, Np):
        a = self.Np[0:self.a_nodes] - np.full((self.a_nodes, self.dimension), Np)
        Nv_norm = (1.0 / np.linalg.norm(self.Nv[0:self.a_nodes], axis=1, keepdims=True)) * self.Nv[0:self.a_nodes]
        return a - (np.sum(a * Nv_norm, axis=1, keepdims=True) * Nv_norm)

    # def influence(self, Np):
    #     if self.a_nodes == 0:
    #         return np.zeros(self.motor_dim)
    #     else:
    #         # print self.density(self.a_nodes), " - ", Np , " - ", (1.0/self.density(self.a_nodes))
    #         return (1.0/self.density(self.a_nodes)) * np.sum(self.Nw[0:self.a_nodes] * self.Nd[0:self.a_nodes] *
    #                 (self.Nv[0:self.a_nodes, 0:self.motor_dim] + self.gamma(Np)[0:self.a_nodes, 0:self.motor_dim]),
    #                                                           axis=0, keepdims=True)[0]

    def influence(self, Np):
        if self.a_nodes == 0:
            return np.zeros(self.motor_dim)
        elif self.density(self.a_nodes) > 0:
            # print(self.gamma(Np)[0:self.a_nodes, 0:self.motor_dim])
            return (1.0/self.density(self.a_nodes)) * np.sum(self.Nw[0:self.a_nodes] * self.Nd[0:self.a_nodes] *
                    (self.Nv[0:self.a_nodes, 0:self.motor_dim] + self.gamma(Np)[0:self.a_nodes, 0:self.motor_dim]),
                                                             axis=0, keepdims=True)[0]
        else:
            return np.zeros(self.motor_dim)

    # New functions

    # def local_density(self):
    #     return np.sum(self.Nd[0:self.a_nodes])

    def s_function(self):
        return 1.0 / (1.0 + np.exp(20.0*np.sum(self.Nd[0:self.a_nodes]) - 20.0))

    def random_motor_activity(self):
        if np.random.uniform(0, 1) < self.k_p:
            # self.R = np.random.normal(0, np.sqrt(2))
            # self.ro = np.random.normal(0, np.sqrt(10))
            self.R = self.k_R * np.random.randn(self.motor_dim)
            self.ro = self.k_ro * np.random.randn(self.motor_dim)
        else:
            self.R = self.R + self.stepsize * self.ro

    def influence_with_random_motor_activity(self, Np):
        self.random_motor_activity()
        s = self.s_function()
        return (1.0 - s) * self.influence(Np) + s * self.R

    def random_motor_activity2(self):
        if np.random.uniform(0, 1) < self.k_p:
            self.rand_flag = True
            self.R = self.k_R * np.random.randn(self.motor_dim)

    def influence_with_random_motor_activity2(self, Np):
        self.random_motor_activity2()
        s = self.s_function()
        # print(s, self.density(self.a_nodes))
        return (1.0 - s) * self.influence(Np) + s * self.R

    def set_k_ro(self, new_k_ro):
        self.k_ro = new_k_ro
        self.ro = self.k_ro * np.random.randn(self.motor_dim)

    def set_k_R(self, new_k_R):
        self.k_R = new_k_R
        self.R = self.k_R * np.random.randn(self.motor_dim)

    def set_k_p(self, new_k_p):
        self.k_p = new_k_p

    def set_seed(self, new_seed):
        np.random.seed(new_seed)

    def reset_idsm(self):
        self.w = np.zeros((self.max_nodes, 1))
        self.Np = np.zeros((self.max_nodes, self.dimension))
        self.Nv = np.zeros((self.max_nodes, self.dimension))
        # self.SM = np.zeros((self.max_nodes, self.dimension))

        self.Nw = np.zeros((self.max_nodes, 1))
        self.Nd = np.zeros((self.max_nodes, 1))

        self.n_nodes = 0  # added nodes
        self.a_nodes = 0  # activated nodes
        time_delay = int(1.0 / self.stepsize)
        self.delay = np.full((self.max_nodes, 1), time_delay)
