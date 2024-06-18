import numpy as np
import matplotlib.pyplot as plt
from IDSM import IDSM
from one_dimensional_agent import Agent
import os

'''
@author Mario Zarco email: mzar066@aucklanduni.ac.nz or zarco.mario.al@gmail.com 
'''

min_n = 0.0
max_n = 1.0


def normalize(value, min_, max_):
    return np.interp(value, (min_, max_), (min_n, max_n))


def inverse(value, min_, max_):
    return np.interp(value, (min_n, max_n), (min_, max_))


if __name__ == "__main__":

    agent = Agent()
    stepsize = 0.01
    idsm = IDSM(1, 1, stepsize)

    number = 2

    k_p = 0.02

    k_R = 2.0

    print("p - ", k_p, " - R - ", k_R)

    seed_agent = 123456789 + number
    seed_idsm = 987654321 + number

    # print(agent.x, " - ", agent.m, " - ", agent.s)

    agent.set_seed(seed_agent)
    agent.randomize_position()
    agent.randomize_motor()
    agent.sensor_update(agent.x)

    idsm.set_seed(seed_idsm)
    idsm.set_k_R(k_R)
    idsm.set_k_p(k_p)

    # print(idsm.a_nodes, " - ", idsm.n_nodes)

    Np = np.array([normalize(agent.m, -1, 1), agent.s])
    Np_prev = Np

    time1 = np.arange(0, 200, stepsize)

    node_pos = []
    node_vel = []

    SM_pos = []
    SM_vel = []

    agent_pos = []
    agent_mot = []

    distance = []
    weightf = []
    weights = []

    for t in time1:

        dudt = idsm.influence_with_random_motor_activity2(Np)
        Np_motor = Np[0] + stepsize * dudt[0]

        # agent.m = inverse(Np_motor, -1, 1)
        agent.set_motor(inverse(Np_motor, -1, 1))
        agent.step(stepsize)
        agent.sensor_update(agent.x)

        Np = np.array([normalize(agent.m, -1, 1), agent.s])
        Nv = (Np - Np_prev) / stepsize

        if t >= 100:
            agent_pos.append(agent.x)
            agent_mot.append(agent.m)
            SM_pos.append(Np)
            SM_vel.append(Nv)

        if idsm.add_node(Np, Nv):
            node_pos.append(Np)
            node_vel.append(Nv)
        idsm.weights_update()
        weights.append(idsm.w[0][0])
        weightf.append(idsm.Nw[0][0])
        distance.append(idsm.Nd[0][0])
        Np_prev = Np

    # plt.plot(time1, distance)
    # plt.plot(time1, weights, "b")
    # plt.plot(time1, weightf, "r")
    # plt.plot(time1, distance, "g")
    # plt.show()

    ax = plt.gca()

    print(len(node_pos))
    print(len(node_vel))
    i = 0
    for nodep, nodev in zip(node_pos, node_vel):
        alpha = np.interp(idsm.w[i][0], (np.min(idsm.w[0:idsm.n_nodes]), np.max(idsm.w[0:idsm.n_nodes])), (0, 1))
        print(np.min(idsm.w[0:idsm.n_nodes]), np.max(idsm.w[0:idsm.n_nodes]), alpha)
        circle = plt.Circle((nodep[0], nodep[1]), 0.02, alpha=alpha, color="blue")
        ax.add_artist(circle)
        nodev = np.array(nodev)
        n_nodev = nodev / np.linalg.norm(nodev)
        ax.arrow(nodep[0]-0.02*n_nodev[0], nodep[1]-0.02*n_nodev[1], 0.02*n_nodev[0], 0.02*n_nodev[1], width=0.004, color="white")
        i +=1
    # plt.show()

    x, y = np.meshgrid(np.arange(0.0, 1.05, 0.05), np.arange(0.0, 1.05, 0.05))
    smstates = np.append(x.reshape(-1, 1), y.reshape(-1, 1), axis=1)

    # for state in smstates:
    #     print(state[0])
    #     break

    SM_pos = np.array(SM_pos)
    print(SM_pos.shape)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.plot(np.array(SM_pos)[:,0], np.array(SM_pos)[:, 1])
    plt.show()
