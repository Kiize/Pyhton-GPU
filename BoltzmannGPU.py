import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
from matplotlib.animation import PillowWriter
import time
start = time.time()
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device("cuda")

#Initial conditions

n_particles = 200
r = torch.rand((2, n_particles)).to(device)
ixr = r[0]>0.5
ixl = r[0]<= 0.5

ids = torch.arange(n_particles)

#plt.figure(figsize=(5,5))
#plt.scatter(r[0][ixr].cpu(), r[1][ixr].cpu(), color='r', s=6)
#plt.scatter(r[0][ixl].cpu(), r[1][ixl].cpu(), color='b', s=6)

v = torch.zeros((2, n_particles)).to(device)
v[0][ixr] = -500
v[0][ixl] = 500

#Distances

ids_pairs = torch.combinations(ids, 2).to(device)
x_pairs = torch.combinations(r[0], 2).to(device)
y_pairs = torch.combinations(r[1], 2).to(device)

dx_pairs = torch.diff(x_pairs, axis = 1).ravel()
dy_pairs = torch.diff(y_pairs, axis = 1).ravel()

d_pairs = torch.sqrt(dx_pairs**2 + dy_pairs**2)

#Velocities of collisions
radius = 5E-3
ids_pairs_collide = ids_pairs[d_pairs < 2*radius]

v1 = v[:, ids_pairs_collide[:,0]]
v2 = v[:, ids_pairs_collide[:,1]]
r1 = r[:, ids_pairs_collide[:,0]]
r2 = r[:, ids_pairs_collide[:,1]]

v1new = v1 - torch.sum( (v1- v2)*(r1 -r2), axis = 0)/torch.sum((r1 - r2)**2, axis=0 ) * (r1 -r2)
v2new = v2 - torch.sum( (v1- v2)*(r1 -r2), axis = 0)/torch.sum((r1 - r2)**2, axis=0 ) * (r2 -r1)

#Functions
def get_deltad_pairs(r):
    dx = torch.diff(torch.combinations(r[0], 2).to(device)).squeeze()
    dy = torch.diff(torch.combinations(r[1], 2).to(device)).squeeze()
    return torch.sqrt(dx**2 + dy**2)

def compute_new_v(v1, v2, r1, r2):
    v1new = v1 - torch.sum( (v1- v2)*(r1 -r2), axis = 0)/torch.sum((r1 - r2)**2, axis=0 ) * (r1 -r2)
    v2new = v2 - torch.sum( (v1- v2)*(r1 -r2), axis = 0)/torch.sum((r1 - r2)**2, axis=0 ) * (r2 -r1)
    return v1new, v2new

def motion(r, v, id_pairs, ts, dt, d_cutoff):
    rs = torch.zeros((ts, r.shape[0], r.shape[1])).to(device)
    vs = torch.zeros((ts, v.shape[0], v.shape[1])).to(device)
    #Stato iniziale
    rs[0] = r
    vs[0] = v

    for i in range(1, ts):
        ic = id_pairs[get_deltad_pairs(r) < d_cutoff]
        v[:, ic[:, 0]], v[:, ic[:, 1]] = compute_new_v(v[:, ic[:, 0]], v[:, ic[:, 1]], r[:, ic[:, 0]], r[:, ic[:, 1]])

        v[0, r[0]>1] = -torch.abs(v[0, r[0]>1])
        v[0, r[0]<0] = torch.abs(v[0, r[0]<0])
        v[1, r[1]>1] = -torch.abs(v[1, r[1]>1])
        v[1, r[1]<0] = torch.abs(v[1, r[1]<0])

        r = r + v*dt
        rs[i] = r
        vs[i] = v
    return rs, vs

rs, vs = motion(r, v, ids_pairs, ts=1000, dt=8E-6, d_cutoff=2*radius)

fig, ax = plt.subplots(1, 1, figsize=(5,5))
"""
xred, yred = rs[0][0][ixr], rs[0][1][ixr]
xblue, yblue = rs[0][0][ixl], rs[0][1][ixl]

circles_red = [plt.Circle((xi, yi), radius=radius, linewidth = 0) for xi, yi in zip(xred, yred)]
circles_blue = [plt.Circle((xi, yi), radius=radius, linewidth = 0) for xi, yi in zip(xblue, yblue)]

cred = matplotlib.collections.PatchCollection(circles_red, facecolors = 'red')
cblue = matplotlib.collections.PatchCollection(circles_blue, facecolors = 'blue')

ax.add_collection(cred)
ax.add_collection(cblue)
ax.set_xlim(0,1)
ax.set_ylim(0,1)
"""

ax.clear()
vmin = 0
vmax = 1
ax.set_xlim(0,1)
ax.set_ylim(0,1)
markersize = 2*radius*ax.get_window_extent().width / (vmax - vmin) * 72./fig.dpi
red, = ax.plot([], [], 'o', color = 'red', markersize = markersize)
blue, = ax.plot([], [], 'o', color = 'blue', markersize = markersize)

def animate(i):
    xred, yred = rs[i][0][ixr].cpu(), rs[i][1][ixr].cpu()
    xblue, yblue = rs[i][0][ixl].cpu(), rs[i][1][ixl].cpu()
    red.set_data(xred, yred)
    blue.set_data(xblue, yblue)
    return red, blue

writer = animation.FFMpegWriter(fps = 30)
ani = animation.FuncAnimation(fig, animate, frames=500, interval=50, blit=True) #blit permette di non rifare le cose che non cambiano
 


plt.show()
end = time.time()
print('time = ', end - start)
print(device)