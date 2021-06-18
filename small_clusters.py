#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:43:15 2021

@author: Alex Yeh
"""

import os, csv
from itertools import permutations, product

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap
from scipy.spatial.transform import Rotation as R
from scipy.spatial.distance import squareform, pdist

from timeit import default_timer as timer

import curved_analysis as ca

def sector(angle, index):
    """Returns points in rhombus defined by [0,1] and [0,1]
    rotated by angle up to (index, index)"""
    rad = angle * (np.pi/180)
    a = np.array([1, 0, 0])
    b = np.array([np.cos(rad), np.sin(rad), 0])
    pairs = [pair for pair in product(range(index), repeat=2) if sum(pair) < index]
    sector = []
    for i, j in pairs:
        tot = i*a + j*b
        sector.append(tot)
    return np.array(sector)

def hex_crystal(angle, index):
    """returns a hexagonal close packed array of points with spacing of 1"""
    raw = sector(60, index)
    sec_num = raw.shape[0] #number of particles in a sector
    
    #this is bad, as we have ~6*index repeated particles
    raw = sector(60, index)
    sec_num = raw.shape[0] #number of particles in a sector
    
    #this is bad, as we have ~6*index repeated particles
    rotated = np.ones((sec_num*6,3))
    rotated[:sec_num] = raw
    for i in range(1,6):
        diff = (i*(np.pi/3)) #necessary rotation
        r1 = R.from_rotvec([0, 0, diff])
        rotated[i*sec_num:(i+1)*sec_num] = r1.apply(raw)
    
    return np.unique(rotated.round(decimals=6), axis=0)
    
def save_to(filename, points, radius):
    ca.save_xyz(points, filename+".xyz")
    ca.fake_nfo(points, filename, radius=radius)
    
def flat_histogram(frames, outer=30, bin_num=15):
    """plots counts of particles by distance from origin in x-y plane
    """
    xycoords = frames[:,:,:2]
    frame_tot, part_tot, _ = xycoords.shape
    
    #gives 1-d array of all distances from origin in x-yplane
    xydist = np.linalg.norm(xycoords, axis=2).flatten()
    #extracts bin edges for use in calculating area
    hbin_edge = np.histogram_bin_edges(xydist,
                                        bins=bin_num,
                                        range=(0, outer))
    widths = hbin_edge[1:] - hbin_edge[:-1]
    mids = hbin_edge[:-1] + widths
    
    #annular area formula as defined below:
    #https://en.wikipedia.org/wiki/Annulus_(mathematics)
    area = np.pi*(hbin_edge[1:]**2 - hbin_edge[:-1]**2)
    
    #get count within each bin
    hval, hbin = np.histogram(xydist, bins=hbin_edge)
    
    #calculate maximum area fraction
    eta_cp = 1/(np.sqrt(3)*1**2) #[count per area]
    
    #calculate area of a single particle, assumes a=1, kappa =a/10
    #or kappa^-1 = 10 nm
    part_area = np.pi*1.044**2
    
    #convert to area fraction
    number_frac = hval / (frame_tot * area)
    area_frac = number_frac * part_area
    return area_frac, hbin

if __name__=="__main__":
    #%% making hexagonal close packed arrays of particles
    
    test = hex_crystal(60, 4)*2
    save_to(f"{test.shape[0]}_hex_crystal", test, 1)
    
    fig = plt.figure()
    fig.set_size_inches(7, 5.25)
    ax = fig.add_subplot(111)
    ax.scatter(test[...,0], test[...,1])
    ax.set_aspect("equal")
    fig.savefig(f"{test.shape[0]}_hex_crystal.png", bbox_inches='tight')
    
    neigh = np.zeros(test.shape[0])
    for i, _ in enumerate(test):
        for j in range(i):
            if np.linalg.norm(test[i]-test[j])<=1.3:
                neigh[i] += 1
                neigh[j] += 1
    
    colors = ["tab:orange", "tab:purple", "tab:blue", "tab:gray"]
    cmap = ListedColormap(colors)
    fig = plt.figure()
    fig.set_size_inches(7, 5.25)
    ax = fig.add_subplot(111)
    sc = ax.scatter(test[...,0], test[...,1], c=neigh, cmap=cmap)
    cbar = fig.colorbar(sc, ax=ax)
    ax.set_aspect("equal")
    
    fig.savefig(f"{test.shape[0]}_hex_crystal_labelled.png", bbox_inches='tight')
    
    #%% making a bicrystal
    r1 = R.from_rotvec([0, 0, np.pi/6])
    offset = r1.apply(test)
    
    unrot_mask = test[:,0]>0
    kept = np.sum(unrot_mask)
    rot_mask = offset[:,0]<=0.01
    new = np.sum(rot_mask)
    combined = -np.ones((kept+new,3))
    combined[:kept] = test[unrot_mask]
    combined[kept:] = offset[rot_mask]
    combined[:kept,0] += 0.45 #make sure two halves don't clash
    dists = pdist(combined)
    print(min(dists))
    
    fig = plt.figure()
    fig.set_size_inches(7, 5.25)
    ax = fig.add_subplot(111)
    ax.scatter(test[unrot_mask,0], test[unrot_mask,1])
    ax.scatter(offset[rot_mask,0], offset[rot_mask,1])
    ax.set_aspect("equal")
    
    fig = plt.figure()
    fig.set_size_inches(7, 5.25)
    ax = fig.add_subplot(111)
    ax.scatter(combined[:,0], combined[:,1])
    ax.set_aspect("equal")
    
    save_to(f"{combined.shape[0]}_bicrystal", combined, 1)
