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
import time
import scipy.integrate as scint
import scipy.optimize as scopt
import scipy.sparse as sp
import scipy.interpolate as sinterpol
import pandas as pd


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

#Mf2 = Mf @ Mf

E2 = -.5 * (E.T @ E)
E4 = E2 @ E2

layers = [[0], [1,2,4,8], [3,5,6,9,10,12],[7,11,13,14], [15]] # layer indices
cell_layers = [[1], [2,3,5,9], [4,6,7,10,11,13],[8,12,14,15], [16]] # cell number indices

edges_layers = [[1, 2, 4, 8], [3, 5, 6, 9, 10, 12], [7, 11, 13, 14], [15]]
ring_radius = np.array([5.23/20, 5.4/20, 5.3/20, 4.0/20]) # in ell_0 units

weight_vec = np.zeros(30)

for i in range(4):
    for j in edges_layers[i]:
        weight_vec[2*(j-1)] = ring_radius[i]**4
        weight_vec[2*(j-1)+1] = ring_radius[i]**4



Weights = np.diag(weight_vec,k=0)
#Weights = diagm(0 =>ones(30))

W = - .5 *  (E.T  @ Weights) @E

def createW(E, rings):
    weight_vec = np.zeros(30)
    for i in range(4):
        for j in edges_layers[i]:
            weight_vec[2*(j-1)] = rings[i]**4
            weight_vec[2*(j-1)+1] = rings[i]**4
    Weights = np.diag(weight_vec,k=0)
    We = - .5 *  (E.T  @ Weights) @E
    return We


class Simulation(object):
    
    def __init__(self, W, Voocyte, gamma, n, G, rho):
        self.W = W
        self.T =  W
        self.N = 16
        
        self.gamma = gamma
        self.n = n
        self.rho = rho
        
        self.G = G
        
        self.layers = [[0], [1,2,4,8], [3,5,6,9,10,12],[7,11,13,14], [15]]
        ## initialize volumes
        self.volumes = np.zeros(16)
        nu = .86
        
        def fopt(u):
            return Voocyte - 4*u**(nu) - 6*u**(nu**2)  - 4* u**(nu**3) - u**(nu**4)
        
        def jac(u):
            return -4*nu*u**(nu - 1) -6*nu**2 * u**(nu**2 - 1) - 4* nu**3 * u**(nu**3 - 1) - nu**4 * u**(nu**4 - 1)
         #u = scopt.root(fopt, Voocyte, jac=jac).x
        u = 13.  ## recompute u for different value of Voocyte!
        for i in range(1,5):
            for lay in self.layers[i]:
                self.volumes[lay] =  u**(nu**i)
        self.volumes[0] = np.sum(self.volumes[1:])
        self.volumes = (100./self.volumes[0]) * self.volumes
        self.volumes_initial = self.volumes.copy()
        #self.volumes_weights = np.ones(4)/4
        
        self.volumes_weights = np.array(
                [(len(self.layers[i]))/15 for i in range(1,5)]
                )
        #self.volumes_weigths = self.volumes_weights/np.sum(self.volumes_weights)
        
        self.r0 = self.rho *self.volumes**(2/3) / (self.n+1)**(2/self.n)
        self.r0n = self.r0**(self.n)
        
    def reset_volumes(self):
        self.volumes = self.volumes_initial.copy()
        self.r0 = self.rho *self.volumes**(2/3) / (self.n+1)**(2/self.n)
        self.r0n = self.r0**(self.n)
    
    

    
    def dpress(self):
        radii = self.radii
        
        return (-radii**(-2) + self.n * self.r0n * radii**(-self.n)
                + self.gamma * (self.n-1) * self.r0n * radii**(-self.n)
                )
        
    def run(self):
        
        def pressure(v):
        
            radii = (v)**(1/3)
        
            return (1. + self.gamma * radii)/(radii ) * (1 - self.r0n*(radii )**(-self.n)) 
    
        def dpress(v):
            radii = (v)**(1/3)
            
            return (-(radii)**(-2) + self.n * self.r0n * (radii)**(-self.n-1)
                    + self.gamma * (self.n-1) * self.r0n * (radii)**(-self.n)
                    )
        
        def Grad(t, v):
            return self.T  * (np.ones(16) @ dpress(v).T)
       
        #jacs = sp.coo_matrix(self.T)
                             
        def dyna(t, v):
            return self.T @ pressure(v)
        
        t0 = 0.
        t1 = 10000.
        tspan = (t0, t1)
#        res = scint.solve_ivp(dyna, tspan, self.volumes[:], method="RK45", max_step=.05,
#                              dense_output=False,
#                              vectorized=False, jac=Grad, jac_sparsity=jacs)
        res = scint.solve_ivp(dyna, tspan, self.volumes[:],
                              dense_output=False,
                              vectorized=False)
        
        self.res = res
        return res
        
    def normalize(self):
        simtime = self.res.t # shape(... , )
        vols = self.res.y  # shape (16, ...)
        
        vol_oo = vols[0,:]
        normalized = np.zeros_like(vols)
        for i in range(16):
            normalized[i,:] = vols[i,:]/vols[i,0]
        norm_areas = normalized**(2/3)
        
        
        end_time_index = np.where(np.abs(norm_areas[15,:]-  .4) < 5e-2)[0]
        
        simtime_adj = simtime/simtime[int((end_time_index[0]+end_time_index[-1])/2)]
        
        return norm_areas, simtime_adj
    
    def normalize_layers(self):
        vols = self.res.y
        simtime = self.res.t
        
        #vol_oo = vols[0,:]
        normalized = np.zeros_like(vols)
        for i in range(16):
            normalized[i,:] = vols[i,:]/vols[i,0]
        
        norm_layers = np.zeros((len(layers), normalized.shape[1]))
        
        for i in range(len(layers)):
            for lay in layers[i]:
                norm_layers[i,:] = np.mean(normalized[layers[i],:], axis=0)
            #norm_layers[i,:] /= len(layers[i])
        
#        #end_time = np.where(np.abs(norm_areas[15,:] - .4) < 1e-2)[0]
#        end_time = np.where(np.isclose(norm_layers[1,:], .25, rtol=1e-3))[0]
#        if len(end_time) == 0:
#            #end_time = np.where(np.abs(norm_areas[15,:] - .4) < 5e-2)[0]
#            end_time = np.where(np.isclose(norm_layers[1,:], .25, rtol=1e-2))[0]
#        if len(end_time) == 0:
#            #end_time = np.where(np.abs(norm_areas[15, :] - .4) < 1e-1)[0]
#            end_time = np.where(np.isclose(norm_layers[1,:], .25, rtol=5e-2))[0]
#            
#        simtime_adj = simtime/simtime[int((end_time[0]+end_time[-1])/2)]
        
        return norm_layers, simtime
    
        
        
class DataComp(object):
    def __init__(self, param_space, data):
        self.param_space = param_space
        self.data = data


    def compute_error(self, sim):
        
        #try:         
        A, t = sim.normalize_layers()
        #index_crop =  np.where(t <1)
        #t_c = t[index_crop]
        #Ac = A[:, index_crop][:,0,:]
        dataset = self.data
        E = 0.
        weights = sim.volumes_weights
        for i in range(len(layers)-1):
            #inter = sinterpol.interp1d(t, A[i+1,:], kind='linear')
            cur_time = dataset[i][0]
            #Aint = inter(cur_time)
            Aint = np.interp(cur_time, t, A[i+1,:], left=1, right=0)
            E += weights[i] * la.norm(dataset[i][1] - Aint)    
        print(E)
        return E
        #except:
            #print('Nan')
            #return float('Nan')
                
    def parameter_error(self):
        
        r1 = self.param_space[0]
        r2 = self.param_space[1]
        r3 = self.param_space[2]
        r4 = self.param_space[3]
        rho_s = self.param_space[4]
        
        self.Error = np.zeros((r1.size, r2.size, r3.size, r4.size, rho_s.size))
        
        miniE = float('inf')
        #argmin = (None, None, None)
        n = len(rho_s) * len(r1) * len(r2) * len(r3) * len(r4)
        
        count = 0
        for i in range(len(r1)):
            for j in range(len(r2)):
                for k in range(len(r3)):
                    for l in range(len(r4)):
                        for m in range(len(rho_s)):
                            rings = np.array([r1[i], r2[j], r3[k], r4[l]])
                            Wr = createW(E, rings)
                            sim = Simulation(Wr, 100, 0, 6, 1, rho_s[m])
                            sim.run()
                            err = self.compute_error(sim)
                            self.Error[i,j,k,l,m] = err
                            if err < miniE:
                                miniE = err
                                argmin = (i, j, k, l, m)
                            count += 1
                    print(str(count/n * 100) + "% progress")
        return self.Error, argmin


if __name__ == "__main__":
    
    sim = Simulation(W, 100, 0, 6, 10, .3)
    res = sim.run()
    
    plt.figure("volumes")
    plt.plot(res.t, res.y.T)
    
    Alay0 = np.zeros((5, len(res.t)))
    
    for i in range(0,5):
        for lay in layers[i]:
            Alay0[i,:] += res.y[lay,:]/res.y[lay,0]
        Alay0[i,:] /= len(layers[i])
        
    plt.figure("test average layer 1")
    plt.plot(res.t, Alay0[1:,:].T)
    #plt.plot(res.t, res.y[0,:].T)
    plt.show()
        
    A_test, t_test = sim.normalize_layers()
    colors = ['blue', 'red', 'green', 'gold']
    
    
#    plt.figure("test normalize 1")
#    plt.plot(t_test, A_test[1:,:].T)
#    
#    index_crop =  np.where(t_test<8)
#    t_c = t_test[index_crop]
#    Ac = A_test[:, index_crop][:,0,:]
#    
#    #plt.plot(t_c, Ac.T)
#    plt.plot(t_c, Ac[1:,:].T)
#    plt.show()
    

    
    #expdata = pd.read_csv("averages_expdata111819.csv")
    expdata = pd.read_csv(r"C:\Users\Nicolas Romeo\Dropbox (MIT)\Research\OocytePumps\area_averages_011320.csv")
    keys = expdata.keys()
    
    exp_time = (4/np.pi)*(1*1e-6/2e-6)*(1/60)
    dataset = [(np.array(expdata[keys[2*i]]/exp_time), np.array(expdata[keys[2*i+1]]**(3/2))) for i in range(len(keys)//2)]
    
    plt.figure("exp data")
    for i in range(len(keys)//2):
        plt.plot(expdata[keys[2*i]], expdata[keys[2*i+1]]**(3/2))
    plt.show()  
    
    plt.figure("exp data rescaled")
    for i in range(len(dataset)):
        plt.plot(dataset[i][0], dataset[i][1], color=colors[i], linestyle='--')
    plt.show()
    
    cropped_data = []
    plt.figure("cropped exp data")
    for i in range(len(dataset)):
        time_indices = np.where(np.logical_not(np.isnan(dataset[i][0][:]))*np.logical_not(np.isnan(dataset[i][1][:])) )
        
        plt.plot(dataset[i][0][time_indices], dataset[i][1][time_indices])
        cropped_data.append((dataset[i][0][time_indices], dataset[i][1][time_indices]))
    plt.show()

    
    def simrings(rings, rho, n, data):
        Ws = createW(E, rings)
        st = Simulation(Ws, 100, 0, n, 10, rho)
        st.run()
        plt.figure("plot rings")
        A, t = st.normalize_layers() 
        #index_crop =  np.where(t<8)
        #t_c = t[index_crop]
        #Ac = A[:, :][:,0,:]
        colors = ['blue', 'red', 'green', 'gold']
        #exptime = (4/np.pi)*(1*1e-6/2e-6)*(1/60)
        for i in range(1,A.shape[0]):
            plt.plot(t*exp_time, A[i,:].T, color=colors[i-1])
        plt.legend(["Layer 1", "Layer 2", "Layer 3", "Layer 4"])
        #fig = plt.gcf()
        #colors = ['blue', 'red', 'green', 'gold']
        for i in range(len(data)):
            plt.plot(data[i][0]*exp_time, data[i][1],
                 color=colors[i], linestyle='--')
        plt.xlabel("time (min)")
        plt.ylabel("Relative volume")
        plt.show()
        return st
    
    n_range = np.linspace(4, 10, 10)
    #rho_range = np.linspace(0.2, 1.2, 10)
    
    #np.array([1., 6.8/8.23, 6.49/8.23, 5.14/8.23])
    r1_range = (5.23/20)*(1 +  np.linspace(-.8,.2, 12))
    r2_range = (5.4/20)*(1 + np.linspace(-.7,.3, 12))
    r3_range = (5.3/20)*(1 +  np.linspace(-.7,.3, 12))
    r4_range = (4.0/20)*(1 +  np.linspace(-.3,.7, 12))
    
    rho_range = np.linspace(0.6, .8, 10)
    
    
    test_range = (np.array([5, 6, 7]), np.array([.2, .4, .6]))
   
    #DC = DataComp((n_range, rho_range), cropped_data)
    #DC = DataComp(test_range, cropped_data)
    
#    DC = DataComp((r1_range, r2_range, r3_range, r4_range), cropped_data)
#    Err, argmin = DC.parameter_error_rc()
#    np.savez_compressed("wtdata_err_grid_rc55_4",
#                        r1=r1_range, r2=r2_range, r3=r3_range, r4=r4_range,
#                        Error=Err)
#    
#    a,b,c,d = argmin
#    ring_opt = [r1_range[a], r2_range[b], r3_range[c], r4_range[d]]
#    sopt = simrings(ring_opt, 0.55, 6)
#    
    
    DC2 = DataComp((r1_range, r2_range, r3_range, r4_range, rho_range), cropped_data)
    Err, argmin = DC2.parameter_error()
    np.savez_compressed("wtdata2_err_grid_5optim_volume_weight2_13", 
                        rho=rho_range, r1=r1_range, r2=r2_range, r3=r3_range, r4=r4_range,
                        Error=Err)   
    
    
    a,b,c,d, e = argmin
    ring_opt = [r1_range[a], r2_range[b], r3_range[c], r4_range[d]]
    sopt = simrings(ring_opt, rho_range[e], 6, dataset)
    
    print(Err)
    
    def loaddata(path):
        loaded = np.load(path)
        Err = loaded['Error']
        rho_range = loaded['rho']
        r1_range = loaded['r1']
        r2_range = loaded['r2']
        r3_range = loaded['r3']
        r4_range = loaded['r4']
        argmin = np.argmin(Err)
        
        (a, b, c, d, e) = np.unravel_index(argmin, Err.shape)
        plt.figure("Err 1,2")
        plt.imshow(np.mean(Err, axis=(2,3)).T, extent=[rho_range[0], rho_range[-1], r2_range[0], r2_range[-1]], aspect="auto")
        #plt.contourf(rho_range, r2_range, np.mean(Err, axis=(2,3)).T, cmap='gray')
        plt.xlabel("$\\rho$")
        plt.ylabel("$r_2$")
        plt.colorbar()
        plt.figure("Err 2,3")
        plt.imshow(Err[a,:,:,d], extent=[r2_range[0], r2_range[-1], r3_range[0], r3_range[-1]], aspect="auto")
        #plt.contourf(r2_range, r3_range, np.mean(Err, axis=(0,3)).T, cmap='gray')
        plt.xlabel("$r_2$")
        plt.ylabel("$r_3$")
        plt.colorbar()
        plt.figure("Err 3,4")
        plt.imshow(Err[a,b,:,:], extent=[r3_range[0], r3_range[-1], r4_range[0], r4_range[-1]], aspect="auto")
        #plt.contourf(r3_range, r4_range, np.mean(Err, axis=(0,1)).T, cmap='gray')
        plt.xlabel("$r_3$")
        plt.ylabel("$r_4$")
        plt.colorbar()
        plt.figure("Err 1,4")
        plt.imshow(Err[:,b,c,:], extent=[rho_range[0], rho_range[-1], r4_range[0], r4_range[-1]], aspect="auto")
        #plt.contourf(rho_range, r4_range, np.mean(Err, axis=(1,2)).T, cmap='gray')
        plt.xlabel("$\\rho$")
        plt.ylabel("$r_4$")
        plt.colorbar()
        plt.show()
        
        plt.figure("Err 1,2 - plane")
        plt.imshow(np.mean(Err, axis=(2,3)), extent=[rho_range[0], rho_range[-1], r2_range[0], r2_range[-1]], aspect="auto")
        #plt.contourf(rho_range, r2_range, Err[:,:,c,d].T, cmap='gray')
        plt.xlabel("$\\rho$")
        plt.ylabel("$r_2$")
        plt.colorbar()
        plt.figure("Err 2,3 - plane")
        plt.imshow(Err[a,:,:,d], extent=[r2_range[0], r2_range[-1], r3_range[0], r3_range[-1]], aspect="auto")
        #plt.contourf(r2_range, r3_range, Err[a,:,c,:].T, cmap='gray')
        plt.xlabel("$r_2$")
        plt.ylabel("$r_3$")
        plt.colorbar()
        plt.figure("Err 3,4 - plane")
        plt.imshow(Err[a,b,:,:], extent=[r3_range[0], r3_range[-1], r4_range[0], r4_range[-1]], aspect="auto")
        #plt.contourf(r3_range, r4_range, Err[a,b,:,:].T, cmap='gray')
        plt.xlabel("$r_3$")
        plt.ylabel("$r_4$")
        plt.colorbar()
        plt.figure("Err 1,4 - plane")
        plt.imshow(Err[:,b,c,:].T, extent=[rho_range[0], rho_range[-1], r4_range[0], r4_range[-1]], aspect="auto")
        #plt.contourf(rho_range, r4_range, Err[:,b,c,:].T, cmap='gray')
        plt.xlabel("$\\rho$")
        plt.ylabel("$r_4$")
        plt.colorbar()
        plt.show()
        return ((rho_range, r2_range, r3_range, r4_range), Err)
    
    fig = plt.gcf()
    colors = ['blue', 'red', 'green', 'gold']
    for i in range(len(dataset)):
        plt.plot(dataset[i][0]*exp_time, dataset[i][1],
                 color=colors[i], linestyle='--')

    #plt.style.use("dark_background")
    plt.figure("Err 1,2")
    #plt.imshow(Err[:,:,c,d,e].T, extent=[r1_range[0], r1_range[-1], r2_range[0], r2_range[-1]], aspect="auto")
    plt.imshow(np.mean(Err, axis=(2,3,4)).T, extent=[r1_range[0], r1_range[-1], r2_range[0], r2_range[-1]], aspect="auto")
    #plt.contourf(r1_range, r2_range, Err[:,:,c,d,e].T, cmap='gray')
    plt.xlabel("$r_1$")
    plt.ylabel("$r_2$")
    plt.colorbar()
    plt.figure("Err 2,3")
    plt.imshow(np.mean(Err, axis=(0,3,4)).T, extent=[r2_range[0], r2_range[-1], r3_range[0], r3_range[-1]], aspect="auto")
    #plt.contourf(r2_range, r3_range, np.mean(Err, axis=(0,3)).T, cmap='gray')
    plt.xlabel("$r_2$")
    plt.ylabel("$r_3$")
    plt.colorbar()
    plt.figure("Err 3,4")
    plt.imshow(np.mean(Err, axis=(0,1,4)).T, extent=[r3_range[0], r3_range[-1], r4_range[0], r4_range[-1]], aspect="auto")
    #plt.contourf(r3_range, r4_range, np.mean(Err, axis=(0,1)).T, cmap='gray')
    plt.xlabel("$r_3$")
    plt.ylabel("$r_4$")
    plt.colorbar()
    plt.figure("Err 1,4")
    plt.imshow(np.mean(Err, axis=(1,2,4)).T, extent=[r1_range[0], r1_range[-1], r4_range[0], r4_range[-1]], aspect="auto")
    #plt.contourf(rho_range, r4_range, np.mean(Err, axis=(1,2)).T, cmap='gray')
    plt.xlabel("$r_1$")
    plt.ylabel("$r_4$")
    plt.colorbar()
    plt.figure("Err 1,rho")
    plt.imshow(np.mean(Err, axis=(1,2,3)).T, extent=[r1_range[0], r1_range[-1], rho_range[0], rho_range[-1]], aspect="auto")
    #plt.contourf(rho_range, r4_range, np.mean(Err, axis=(1,2)).T, cmap='gray')
    plt.xlabel("$r_1$")
    plt.ylabel("$\\rho$")
    plt.colorbar()
    plt.figure("Err 2,rho")
    plt.imshow(np.mean(Err, axis=(0,2,3)).T, extent=[r2_range[0], r2_range[-1], rho_range[0], rho_range[-1]], aspect="auto")
    #plt.contourf(rho_range, r4_range, np.mean(Err, axis=(1,2)).T, cmap='gray')
    plt.xlabel("$r_2$")
    plt.ylabel("$\\rho$")
    plt.colorbar()
    plt.figure("Err 3,rho")
    plt.imshow(np.mean(Err, axis=(0,1,3)).T, extent=[r3_range[0], r3_range[-1], rho_range[0], rho_range[-1]], aspect="auto")
    #plt.contourf(rho_range, r4_range, np.mean(Err, axis=(1,2)).T, cmap='gray')
    plt.xlabel("$r_3$")
    plt.ylabel("$\\rho$")
    plt.colorbar()
    
    
    plt.show()
    
   
    def savesim(rings, rho, filename):
        Ws = createW(E, rings)
        st = Simulation(Ws, 100, 0, 6, 10, rho)
        st.run()
        A, t = st.normalize_layers()
        dfa = np.vstack((t, A))
        dfb = np.vstack((st.res.t, st.res.y))
        np.savetxt(filename+"layers.csv", dfa, delimiter=",")
        np.savetxt(filename+"cells.csv", dfb, delimiter=",")
        print("Saved "+filename+".csv")
        
    def envelope(argmin, pranges, filename):
        argmin = np.array(argmin)
        
        for i in range(5):
            if argmin[i] !=0:
                rings = [pranges[j][argmin[j] - 1*(i==j)] for j in range(4)]
                rho = pranges[4][argmin[4] - 1*(i==4)]
                savesim(rings, rho, filename+"_" +str(i)+"1")
            else:
                rings = [pranges[j][argmin[j]]  - (pranges[j][1]- pranges[j][0])*(i==j) for j in range(4)]
                rho = pranges[4][argmin[4]] - (pranges[4][1]- pranges[4][0])*(i==4)
                savesim(rings, rho, filename+"_" +str(i)+"1")
            if argmin[i] != len(pranges[i])-1:
                rings = [pranges[j][argmin[j] + 1*(i==j)] for j in range(4)]
                rho = pranges[4][argmin[4] + 1*(i==4)]
                savesim(rings, rho, filename+"_" +str(i)+"2")
            else:
                rings = [pranges[j][argmin[j]] + (pranges[j][-1]- pranges[j][-2])*(i==j) for j in range(4)]
                rho = pranges[4][argmin[4]] + (pranges[4][-1]- pranges[4][-2])*(i==4)
                savesim(rings, rho, filename+"_" +str(i)+"2")
                
        
#    
#    plt.figure("err log")
#    plt.imshow(np.log(Err), extent=[rho_range[0], rho_range[-1], n_range[0], n_range[-1]], 
#                aspect="auto")
#    plt.xlabel("$R_0^{2/3}$")
#    plt.ylabel("$n$")
#    plt.colorbar()
#    plt.show()
#    
#    plt.figure("err")
#    plt.imshow(Err, extent=[rho_range[0], rho_range[-1], n_range[0], n_range[-1]], aspect="auto")
#    plt.xlabel("$\\rho$")
#    plt.ylabel("$n$")
#    plt.colorbar()
#    plt.show()
    
    A_data, t_data = sopt.normalize_layers()
    index_crop =  np.where(t_data<1)
    t_c = t_data[index_crop]
    Ac = A_data[:, index_crop][:,0,:]
    dfa = np.vstack((t_c, Ac))
#    np.savetxt("sim_rho055_n6_010520.csv", dfa, delimiter=",")

