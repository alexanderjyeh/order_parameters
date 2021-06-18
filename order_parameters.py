#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 11:26:56 2020

@author: glaucus
"""

import numpy as np
from scipy.spatial.distance import squareform, pdist

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

def rg(frame):
    """returns normalized Rg for set of particles"""
    pnum = frame.shape[0] #number of particles
    centered = frame - np.average(frame, axis=0) #translated to center of mass
    squ_rad = np.sum(centered**2, axis=1) #square radial dist at each particle [um^2]
    Rg = np.sqrt(np.average(squ_rad))
    #ideal hexagonal close packed analytical formula
    # Equ 13 in https://doi.org/10.1063/1.4951698
    Rg_hex = 1*np.sqrt(pnum*5)/3  
    return Rg/Rg_hex

def local_psi6(frame, coordination_shell = 2.64):
    """Takes in a single frame of particle coordinates (N x 2) and outputs
    local 6-fold bond orientational order (psi6) values at each particle 
    (N x 1) array of complex values"""
    pnum = frame.shape[0] #number of particles
    #pairwise distances, transformed to square matrix for ease of access
    pairs = squareform(pdist(frame)) 
    
    psi6s = np.zeros(pnum, dtype=complex) #accumulates local psi6
    for p in range(pnum):
        #get array of indices of particles within one coordination shell
        n_idx = np.nonzero((pairs[p]<coordination_shell) & (pairs[p]!=0))[0] 
        if n_idx.size != 0:
            n_coords = frame[n_idx] - frame[p] #neighbor displacements
            #calculate angle between each neighbor and x-axis
            thetas = np.arctan2(n_coords[:,1], n_coords[:, 0])
            #theta.size is also number of neighbors, so we take the average
            # Equ 14 in https://doi.org/10.1063/1.4951698
            psi6s[p] = np.average(np.exp(thetas*6j))
    
    return psi6s

def local_c6(frame, coordination_shell = 2.64):
    """Takes in a single frame of particle coordinates (Nx2) and outputs
    local 6-fold connectivity. This provides an integer value from 0 to 6
    inclusive, representing the number of crystalline near neighbors. The
    necessary precalculations are also output."""
    # necessary precalculations
    pnum = frame.shape[0] #number of particles
    #pairwise distances, transformed to square matrix for ease of access
    pairs = squareform(pdist(frame)) 
    psi6s = local_psi6(frame, coordination_shell = coordination_shell)
    conj_psi6s = psi6s.conjugate()
    
    c6s = np.zeros(pnum, dtype=float) #accumulates local c6
    for p in range(pnum):
        #get array of indices of particles within one coordination shell
        n_idx = np.nonzero((pairs[p]<coordination_shell) & (pairs[p]!=0))[0] 
        if n_idx.size != 0:
            #following S19 in supp info of https://doi.org/10.1039/C3SM50809A
            products = psi6s[p]*conj_psi6s[n_idx]
            numer = np.abs(np.real(products))
            denom = np.abs(products)
            chi6s = numer/denom
            #following S20 in supp info of https://doi.org/10.1039/C3SM50809A
            mask = chi6s>=0.32 #criterion
            c6s[p] = np.sum(mask) #counting cells which match criterion
    
    return psi6s, c6s

def global_psi6(frame):
    """Takes in a frame with 2D particle coordinates and returns the global
    6-fold bond orientational order (float from [0,1])"""
    l_psi6s = local_psi6(frame)
    comp_ave = np.average(l_psi6s) #average over local order
    #find magnitude of result
    # Equ 15 in https://doi.org/10.1063/1.4951698
    psi6 = np.sqrt(comp_ave*np.conj(comp_ave)) 
    return np.real(psi6)

def global_psi6_from_local(l_psi6):
    """Outputs global psi6 given a (N x 1) array corresponding to local psi6."""
    ave = l_psi6.mean()
    return np.real(np.sqrt(ave*ave.conjugate()))

def global_c6(frame):
    """Takes in a frame with 2D particle coordinates and returns the average 
    local 6-fold connectivity order. Also returns necessary prerequisite 
    values: psi6, chi6, and c6 for each particle."""
    psi6s, c6s = local_c6(frame)
    return c6s.mean()/c6_hex(frame.shape[0])

def global_c6_from_local(l_c6):
    """Outputs average c6 given a (N x 1) array corresponding to local c6."""
    ave = l_c6.mean()
    return ave/l_c6.shape[0]

def shells(pnum):
    """from particle number, calculate number of shells assuming hexagonal crystal"""
    # equation 8 from SI of https://doi.org/10.1039/C2LC40692F
    return -1/2 + np.sqrt((pnum-1)/3 + 1/4)

def c6_hex(pnum):
    """returns C6 for a hexagonal cluster of the same size"""
    # equation 10 from SI of https://doi.org/10.1039/C2LC40692F
    s = shells(pnum)
    return 6*(3*s**2 + s)/pnum

if __name__=="__main__":
    #%% analyzing a perfect crystal with hexagonal morphology
    raw = np.genfromtxt("37_hex_crystal.txt")*2 # bring particle radius to 1
    coords = raw[:,1:3]  # stripping particle number and z coord
    pnum = coords.shape[0] # get total number of particles
    
    # as every particle is in a perfect crystal, we can calculate C6 my hand
    g_c6 = (3*6                 # six corners are 3 coordinated
            + 4*(2*6)           # 2 particles per edge are 4 coordinated
            + 6*(2*(3+4) + 5))  # the interior is 6 coordinated
    g_c6 = g_c6/37
    
    # print the hand calculated values
    ideal = c6_hex(pnum)    
    print(f"hand calc'd: {g_c6:0.5f}")
    print(f"ideal: {ideal:0.5f}")
    
    # print the computed values
    psi6s, c6s = local_c6(coords)
    calc_g_c6 = global_c6(coords)
    calc_g_psi6 = global_psi6_from_local(psi6s)
    print(f"calculated c6: {calc_g_c6:0.5f}")
    print(f"calculated psi6: {calc_g_psi6:0.5f}")
    
    cmap = cm.get_cmap('Dark2', 4)
    fig = plt.figure()
    fig.set_size_inches(7, 5.25)
    ax = fig.add_subplot(111)
    ax.set_title("$<C_{6,i}>$ for each particle in a hexagonal crystal")
    sc = ax.scatter(coords[...,0], coords[...,1], c=c6s, cmap=cmap)
    cbar = fig.colorbar(sc, ax=ax)
    ax.set_aspect("equal")
    
    fig = plt.figure()
    fig.set_size_inches(7, 5.25)
    ax = fig.add_subplot(111)
    ax.set_title("$<\Psi_{6,i}>$ for each particle in a hexagonal crystal")
    qu = ax.quiver(coords[...,0], coords[...,1],
                   psi6s[...,0], psi6s[...,1])
    ax.set_aspect("equal")
    
    #%% looking at loose collection of particles
    raw = np.genfromtxt("loose.txt")
    coords = raw[:,1:3]  #stripping particle number and z coord
       
    psi6s, c6s = local_c6(coords)
    calc_g_c6 = global_c6(coords)
    calc_g_psi6 = global_psi6_from_local(psi6s)
    print(f"calculated c6: {calc_g_c6:0.5f}")
    print(f"calculated psi6: {calc_g_psi6:0.5f}")
    
    cmap = cm.get_cmap('Dark2', 2)
    fig = plt.figure()
    fig.set_size_inches(7, 5.25)
    ax = fig.add_subplot(111)
    ax.set_title("$<C_{6,i}>$ for each particle in a disordered fluid")
    sc = ax.scatter(coords[...,0], coords[...,1], c=c6s, cmap=cmap)
    cbar = fig.colorbar(sc, ax=ax)
    ax.set_aspect("equal")
    
    #%% looking at bicrystal
    raw = np.genfromtxt("39_bicrystal.txt")
    coords = raw[:,1:3]  #stripping particle number and z coord
       
    psi6s, c6s = local_c6(coords)
    calc_g_c6 = global_c6(coords)
    calc_g_psi6 = global_psi6_from_local(psi6s)
    print(f"calculated c6: {calc_g_c6:0.5f}")
    print(f"calculated psi6: {calc_g_psi6:0.5f}")
    
    cmap = cm.get_cmap('Dark2', 5)
    fig = plt.figure()
    fig.set_size_inches(7, 5.25)
    ax = fig.add_subplot(111)
    ax.set_title("$<C_{6,i}>$ for each particle in a bicrystal")
    sc = ax.scatter(coords[...,0], coords[...,1], c=c6s, cmap=cmap)
    cbar = fig.colorbar(sc, ax=ax)
    ax.set_aspect("equal")
    
    cmap = cm.get_cmap('hsv')
    fig = plt.figure()
    fig.set_size_inches(7, 5.25)
    ax = fig.add_subplot(111)
    ax.set_title("Particles in a bicrystal colored by phase of $<\Psi_{6,i}>$")
    sc = ax.scatter(coords[...,0], coords[...,1],
                    c=(np.angle(psi6s)*180/np.pi), cmap=cmap)
    cbar = fig.colorbar(sc, ax=ax)
    ax.set_aspect("equal")
    
    fig = plt.figure()
    fig.set_size_inches(7, 5.25)
    ax = fig.add_subplot(111)
    ax.set_title("$<\Psi_{6,i}>$ for each particle in a bicrystal")
    qu = ax.quiver(coords[...,0], coords[...,1],
                   psi6s.real, psi6s.imag)
    ax.set_aspect("equal")
