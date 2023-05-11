import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
from matplotlib.animation import PillowWriter
import time

start = time.time()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Parameters
velocity = 100
n_particles = 1000
radius = 5E-3
gravity = -1E5

#Boltz Distribution
bins = np.linspace(0, 10*velocity, 60)
vv = np.linspace(0, 10*velocity, 1000)
mbeta = (velocity**2/12 - gravity/4)**(-1)
fv = mbeta*vv*np.exp(-mbeta*vv**2/2)

#Initial conditions
r = torch.rand((2, n_particles)).to(device)
ixr = r[0]>0.5
ixl = r[0]<= 0.5
ids = torch.arange(n_particles)

vsfer = torch.rand((2, n_particles)).to(device)
vsfer[0] = vsfer[0] * velocity
vsfer[1] = vsfer[1] * 2 * np.pi

'''
Uniform velocity modulo
'''

v = torch.zeros((2, n_particles)).to(device)
v[0] = vsfer[0]*torch.cos(vsfer[1])
v[1] = vsfer[0]*torch.sin(vsfer[1])

g = torch.zeros((2,n_particles)).to(device)
g[1] = gravity

#Distances
ids_pairs = torch.combinations(ids, 2).to(device)
x_pairs = torch.stack([r[0][ids_pairs[:,0]], r[0][ids_pairs[:,1]]]).T
y_pairs = torch.stack([r[1][ids_pairs[:,0]], r[1][ids_pairs[:,1]]]).T

dx_pairs = torch.diff(x_pairs, axis = 1).ravel()
dy_pairs = torch.diff(y_pairs, axis = 1).ravel()

d_pairs = torch.sqrt(dx_pairs**2 + dy_pairs**2)

#Functions
def get_deltad_pairs(r, ids_pairs):
    dx = torch.diff(torch.stack([r[0][ids_pairs[:,0]], r[0][ids_pairs[:,1]]]).T).squeeze()
    dy = torch.diff(torch.stack([r[1][ids_pairs[:,0]], r[1][ids_pairs[:,1]]]).T).squeeze()
    return dx**2 + dy**2

def compute_new_v(v1, v2, r1, r2):
    v1new = v1 - torch.sum( (v1- v2)*(r1 -r2), axis = 0)/torch.sum((r1 - r2)**2, axis=0 ) * (r1 -r2)
    v2new = v2 - torch.sum( (v1- v2)*(r1 -r2), axis = 0)/torch.sum((r1 - r2)**2, axis=0 ) * (r2 -r1)
    return v1new, v2new

def motion(r, v, id_pairs, ts, dt, d_cutoff):
    rs = torch.zeros((ts, r.shape[0], r.shape[1])).to(device)
    vs = torch.zeros((ts, v.shape[0], v.shape[1])).to(device)
    #Initial state
    rs[0] = r
    vs[0] = v

    for i in range(1, ts):
        ic = id_pairs[get_deltad_pairs(r, ids_pairs) < d_cutoff**2]
        v[:, ic[:, 0]], v[:, ic[:, 1]] = compute_new_v(v[:, ic[:, 0]], v[:, ic[:, 1]], r[:, ic[:, 0]], r[:, ic[:, 1]])

        v[0, r[0]>1] = -torch.abs(v[0, r[0]>1])
        v[0, r[0]<0] = torch.abs(v[0, r[0]<0])
        v[1, r[1]>1] = -torch.abs(v[1, r[1]>1])
        v[1, r[1]<0] = torch.abs(v[1, r[1]<0])

        r = r + v*dt + 1/2*g*(dt)**2
        v = v + g*dt
        
        rs[i] = r
        vs[i] = v
    return rs, vs

rs, vs = motion(r, v, ids_pairs, ts=1000, dt=8E-5, d_cutoff=2*radius)

fig, ax = plt.subplots(1, 2, figsize=(15, 10))
ax[0].clear()
vmin = 0
vmax = 1
ax[0].set_xlim(0,1)
ax[0].set_ylim(0,1)
markersize = 2*radius*ax[0].get_window_extent().width / (vmax - vmin) * 72./fig.dpi
red, = ax[0].plot([], [], 'o', color = 'red', markersize = markersize)
blue, = ax[0].plot([], [], 'o', color = 'blue', markersize = markersize)
_, _, bar_container = ax[1].hist(torch.sqrt(torch.sum(vs[0]**2, axis = 0)).cpu(), bins=bins, density=True)
ax[1].plot(vv, fv)
ax[1].set_ylim(top=np.max(fv) + 0.001)

def prepare_animation(bar_container):

    def animate(i):
        n, _ = np.histogram(torch.sqrt(torch.sum(vs[i]**2, axis = 0)).cpu(), bins=bins, density=True)
        xred, yred = rs[i][0][ixr].cpu(), rs[i][1][ixr].cpu()
        xblue, yblue = rs[i][0][ixl].cpu(), rs[i][1][ixl].cpu()
        red.set_data(xred, yred)
        blue.set_data(xblue, yblue)
        for i, patch in enumerate(bar_container.patches):
            patch.set_height(n[i])
        return bar_container.patches

    return animate

def red_blue(i):
    xred, yred = rs[i][0][ixr].cpu(), rs[i][1][ixr].cpu()
    xblue, yblue = rs[i][0][ixl].cpu(), rs[i][1][ixl].cpu()
    red.set_data(xred, yred)
    blue.set_data(xblue, yblue)
    return red, blue

writer = animation.FFMpegWriter(fps = 30)
ani1 = animation.FuncAnimation(fig, prepare_animation(bar_container), frames=500, interval=50, blit=True) #blit permette di non rifare le cose che non cambiano
ani2 = animation.FuncAnimation(fig, red_blue, frames=500, interval=50, blit=True) #blit permette di non rifare le cose che non cambiano
 

end = time.time()
print('time = ', end - start)


plt.show()