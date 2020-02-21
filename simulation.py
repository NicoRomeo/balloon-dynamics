# -*- coding: utf-8 -*-
"""

Balloon network simulation -- unequal ring canals, optimization over ring canal
radii and zero pressure radius. Corrected physical parameters. 
Created on Wed Feb 12 2020

@author: Nicolas Romeo
"""

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import scipy.integrate as scint
import pandas as pd

# Global definition of the graph laplacian
Mf = np.array([
[-4,  1,  1,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],
 [1, -4,  0,  1,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0],
 [1,  0, -3,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0],
 [0,  1,  0, -3,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0],
 [1,  0,  0,  0, -2,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],
 [0,  1,  0,  0,  0, -2,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0],
 [0,  0,  1,  0,  0,  0, -2,  0,  0,  0,  0,  0,  0,  0,  1,  0],
 [0,  0,  0,  1,  0,  0,  0, -2,  0,  0,  0,  0,  0,  0,  0,  1],
 [1,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0],
 [0,  1,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0],
 [0,  0,  1,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0],
 [0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0],
 [0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0],
 [0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0],
 [0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0, -1,  0],
 [0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0, -1]])

# Graph edge matrix
E = np.array([
  [-1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
  [1,  -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
  [-1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
  [1,   0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
  [0,  -1,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
  [0,  1,  0,  -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
  [-1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
  [1,  0,  0,  0,  -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
  [0, -1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
  [0,  1,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
  [0,  0, -1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
  [0,  0,  1,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0],
  [0,  0,  0, -1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0],
  [0,  0,  0,  1,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  0],
  [-1,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],
  [1,  0,  0,  0,  0,  0,  0,  0,  -1,  0,  0,  0,  0,  0,  0,  0],
  [0, -1,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0],
  [0,  1,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0],
  [0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0],
  [0,  0,  1,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0],
  [0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0],
  [0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0],
  [0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],
  [0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0],
  [0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0],
  [0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0, -1,  0,  0],
  [0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  1,  0],
  [0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0, -1,  0],
  [0,  0,  0,  0,  0,  0,  0, -1,  0,  0,  0,  0,  0,  0,  0,  1],
  [0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0, -1]])

layers = [[0], [1, 2, 4, 8], [3, 5, 6, 9, 10, 12], [7, 11, 13, 14], [15]] # layer indices
cell_layers = [[1], [2, 3, 5, 9], [4, 6, 7, 10, 11, 13], [8, 12, 14, 15], [16]] # cell number indices

edges_layers = [[1, 2, 4, 8], [3, 5, 6, 9, 10, 12], [7, 11, 13, 14], [15]] # indices of edges in layer i
ring_radius = np.array([5.23 / 20, 5.4 / 20, 5.3 / 20, 4.0 / 20]) # in ell_0 units

def createW(E, rings): # function to create the weighted graph laplacian from ring canal sizes
    weight_vec = np.zeros(30)
    for i in range(4):
        for j in edges_layers[i]:
            weight_vec[2*(j-1)] = rings[i]**4
            weight_vec[2*(j-1)+1] = rings[i]**4
    Weights = np.diag(weight_vec,k=0)
    We = - .5 *  (E.T  @ Weights) @E
    return We


class Simulation(object):  # main simulation object

    def __init__(self, T, Voocyte, n, rho):
        self.T = T  # dynamical matrix
        self.N = 16

        self.n = n  # exponent in finite-size correction to Young-Laplace's law
        self.rho = rho  # max pressure radius

        self.layers = [[0], [1, 2, 4, 8], [3, 5, 6, 9, 10, 12], [7, 11, 13, 14], [15]]
        # initialize volumes
        self.volumes = np.zeros(16)
        nu = .86
        u = 13.
        for i in range(1, 5):
            for lay in self.layers[i]:
                self.volumes[lay] = u**(nu**i)
        self.volumes[0] = np.sum(self.volumes[1:])
        self.volumes = (Voocyte / self.volumes[0]) * self.volumes
        self.volumes_initial = self.volumes.copy()

        # Setting up the vlue of the max-pressure radius
        self.r0 = self.rho * self.volumes**(2 / 3) / (self.n + 1)**(2 / self.n)
        self.r0n = self.r0**(self.n)

        # weigths for comparison to experiments (g_d in methods)
        self.volumes_weights = np.array([(len(self.layers[i])) / 15 for i in range(1, 5)])

    def run(self):  # Solving for the dynamical evolution

        def pressure(v):
            radii = v**(1 / 3)
            return (1. + self.gamma * radii) / radii * (1 - self.r0n * radii**(-self.n))

        def dyna(t, v):
            return self.T @ pressure(v)

        t0 = 0.
        t1 = 10000.
        tspan = (t0, t1)
        res = scint.solve_ivp(dyna, tspan, self.volumes[:],
                              dense_output=False,
                              vectorized=False)
        self.res = res
        return res

    def normalize_layers(self):
        vols = self.res.y
        simtime = self.res.t

        # vol_oo = vols[0,:]
        normalized = np.zeros_like(vols)
        for i in range(16):
            normalized[i, :] = vols[i, :] / vols[i, 0]

        norm_layers = np.zeros((len(layers), normalized.shape[1]))

        for i in range(len(layers)):
            for lay in layers[i]:
                norm_layers[i, :] = np.mean(normalized[layers[i], :], axis=0)
        return norm_layers, simtime


class DataComp(object):  # Class to handle the error grids and comparison to experimental data
    def __init__(self, param_space, data):
        self.param_space = param_space
        self.data = data

    def compute_error(self, sim):
        A, t = sim.normalize_layers()

        dataset = self.data
        E = 0.
        weights = sim.volumes_weights
        for i in range(len(layers) - 1):
            cur_time = dataset[i][0]
            Aint = np.interp(cur_time, t, A[i + 1, :], left=1, right=0)
            E += weights[i] * la.norm(dataset[i][1] - Aint)
        print(E)
        return E

    def parameter_error(self):

        r1 = self.param_space[0]
        r2 = self.param_space[1]
        r3 = self.param_space[2]
        r4 = self.param_space[3]
        rho_s = self.param_space[4]

        self.Error = np.zeros((r1.size, r2.size, r3.size, r4.size, rho_s.size))

        miniE = float('inf')
        n = len(rho_s) * len(r1) * len(r2) * len(r3) * len(r4)
        count = 0
        for i in range(len(r1)):
            for j in range(len(r2)):
                for k in range(len(r3)):
                    for l in range(len(r4)):
                        for m in range(len(rho_s)):
                            rings = np.array([r1[i], r2[j], r3[k], r4[l]])
                            Wr = createW(E, rings)
                            sim = Simulation(Wr, 100, 6, rho_s[m])
                            sim.run()
                            err = self.compute_error(sim)
                            self.Error[i, j, k, l, m] = err
                            if err < miniE:
                                miniE = err
                                argmin = (i, j, k, l, m)
                            count += 1
                    print(str(count / n * 100) + "% progress")
        return self.Error, argmin


def savesim(rings, rho, filename):
        Ws = createW(E, rings)
        st = Simulation(Ws, 100, 6, rho)
        st.run()
        A, t = st.normalize_layers()
        dfa = np.vstack((t, A))  # Averages over layers
        dfb = np.vstack((st.res.t, st.res.y))  # all cells volumes
        np.savetxt(filename + "layers.csv", dfa, delimiter=",")
        np.savetxt(filename + "cells.csv", dfb, delimiter=",")
        print("Saved " + filename + ".csv")


if __name__ == "__main__":

    W = createW(E, ring_radius)
    sim = Simulation(W, 100, 6, .5)
    res = sim.run()
    plt.figure("Example run, volumes")
    plt.plot(res.t, res.y.T)

    Alay0 = np.zeros((5, len(res.t)))

    for i in range(0, 5):
        for lay in layers[i]:
            Alay0[i, :] += res.y[lay, :] / res.y[lay, 0]
        Alay0[i, :] /= len(layers[i])

    colors = ['blue', 'red', 'green', 'gold']

    plt.figure("Example run, Average per layer")
    for i in range(4):
        plt.plot(res.t, Alay0[1:, i].T, color=colors[i])
    plt.show()

    A_test, t_test = sim.normalize_layers()
    colors = ['blue', 'red', 'green', 'gold']

    expdata = pd.read_csv(r"area_averages_011320.csv")
    keys = expdata.keys()

    exp_time = (4 / np.pi) * (1 * 1e-6 / 2e-6) * (1 / 60)  # simulation time unit in minutes
    dataset = [(np.array(expdata[keys[2 * i]] / exp_time), np.array(expdata[keys[2 * i + 1]]**(3 / 2))) for i in range(len(keys) // 2)]

    plt.figure("Experimental data")
    for i in range(len(keys) // 2):
        plt.plot(expdata[keys[2 * i]], expdata[keys[2 * i + 1]]**(3 / 2))
    plt.show()

    cropped_data = []
    plt.figure("Experimental data, NaNs removed")
    for i in range(len(dataset)):
        time_indices = np.where(np.logical_not(np.isnan(dataset[i][0][:])) * np.logical_not(np.isnan(dataset[i][1][:])))
        plt.plot(dataset[i][0][time_indices], dataset[i][1][time_indices])
        cropped_data.append((dataset[i][0][time_indices], dataset[i][1][time_indices]))
    plt.show()

    def simrings(rings, rho, n, data):
        Ws = createW(E, rings)
        st = Simulation(Ws, 100, n, rho)
        st.run()
        plt.figure("plot rings")
        A, t = st.normalize_layers()
        colors = ['blue', 'red', 'green', 'gold']
        for i in range(1, A.shape[0]):
            plt.plot(t * exp_time, A[i, :].T, color=colors[i - 1])
        plt.legend(["Layer 1", "Layer 2", "Layer 3", "Layer 4"])
        for i in range(len(data)):
            plt.plot(data[i][0] * exp_time, data[i][1],
                 color=colors[i], linestyle='--')
        plt.xlabel("time (min)")
        plt.ylabel("Relative volume")
        plt.show()
        return st

    r1_range = (5.23 / 20) * (1 + np.linspace(-.8, .2, 12))
    r2_range = (5.4 / 20) * (1 + np.linspace(-.7, .3, 12))
    r3_range = (5.3 / 20) * (1 + np.linspace(-.7, .3, 12))
    r4_range = (4.0 / 20) * (1 + np.linspace(-.3, .7, 12))

    rho_range = np.linspace(0.6, .8, 10)


    test_range = (np.array([5, 6, 7]), np.array([.2, .4, .6]))


    DC = DataComp((r1_range, r2_range, r3_range, r4_range, rho_range), cropped_data)
    Err, argmin = DC.parameter_error()

    ## Uncomment to save grid search result as .npz

    #np.savez_compressed("gridsave", 
    #                    rho=rho_range, r1=r1_range, r2=r2_range, r3=r3_range, r4=r4_range,
    #                    Error=Err)   


    a,b,c,d, e = argmin
    ring_opt = [r1_range[a], r2_range[b], r3_range[c], r4_range[d]]
    sopt = simrings(ring_opt, rho_range[e], 6, dataset)

    print(Err)

    # plots the experimental data over the current plot
    fig = plt.gcf()
    colors = ['blue', 'red', 'green', 'gold']
    for i in range(len(dataset)):
        plt.plot(dataset[i][0]*exp_time, dataset[i][1],
                 color=colors[i], linestyle='--')


    ## Grid error plots plotting - Mean projection

    plt.figure("Err 1,2")
    #plt.imshow(Err[:,:,c,d,e].T, extent=[r1_range[0], r1_range[-1], r2_range[0], r2_range[-1]], aspect="auto")
    plt.imshow(np.mean(Err, axis=(2, 3, 4)).T, extent=[r1_range[0], r1_range[-1], r2_range[0], r2_range[-1]], aspect="auto")
    #plt.contourf(r1_range, r2_range, Err[:,:,c,d,e].T, cmap='gray')
    plt.xlabel("$r_1$")
    plt.ylabel("$r_2$")
    plt.colorbar()
    plt.figure("Err 2,3")
    plt.imshow(np.mean(Err, axis=(0, 3, 4)).T, extent=[r2_range[0], r2_range[-1], r3_range[0], r3_range[-1]], aspect="auto")
    plt.xlabel("$r_2$")
    plt.ylabel("$r_3$")
    plt.colorbar()
    plt.figure("Err 3,4")
    plt.imshow(np.mean(Err, axis=(0,1,4)).T, extent=[r3_range[0], r3_range[-1], r4_range[0], r4_range[-1]], aspect="auto")
    plt.xlabel("$r_3$")
    plt.ylabel("$r_4$")
    plt.colorbar()
    plt.figure("Err 1,4")
    plt.imshow(np.mean(Err, axis=(1, 2, 4)).T, extent=[r1_range[0], r1_range[-1], r4_range[0], r4_range[-1]], aspect="auto")
    plt.xlabel("$r_1$")
    plt.ylabel("$r_4$")
    plt.colorbar()
    plt.figure("Err 1,rho")
    plt.imshow(np.mean(Err, axis=(1,2,3)).T, extent=[r1_range[0], r1_range[-1], rho_range[0], rho_range[-1]], aspect="auto")
    plt.xlabel("$r_1$")
    plt.ylabel("$\\rho$")
    plt.colorbar()
    plt.figure("Err 2,rho")
    plt.imshow(np.mean(Err, axis=(0,2,3)).T, extent=[r2_range[0], r2_range[-1], rho_range[0], rho_range[-1]], aspect="auto")
    plt.xlabel("$r_2$")
    plt.ylabel("$\\rho$")
    plt.colorbar()
    plt.figure("Err 3,rho")
    plt.imshow(np.mean(Err, axis=(0,1,3)).T, extent=[r3_range[0], r3_range[-1], rho_range[0], rho_range[-1]], aspect="auto")
    plt.xlabel("$r_3$")
    plt.ylabel("$\\rho$")
    plt.colorbar()

    plt.show()
