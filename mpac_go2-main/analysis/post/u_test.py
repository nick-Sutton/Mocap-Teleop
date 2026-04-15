import numpy as np
import h5py
import matplotlib.pyplot as p
from scipy.spatial.transform import Rotation as R

p.rcParams.update({
  "text.usetex": True,
  "font.size": 22})

file = h5py.File("data/latest.h5",'r')
#file = h5py.File("data/20201118/104916_a1.h5",'r')
data = file['time_series']

u = data['u'][:,0:3]
print(len(u[0]))
N=1
for i in range(len(u[0])):
  mov_avg = np.convolve(u[:,i], np.ones((N,))/N, mode='valid')
  p.plot(mov_avg)

p.grid()
#p.title('ID-QP Standing with Velocity Safety Constraint')
p.xlabel('time (ms)')
#p.ylabel('$\dot{q}$ (rad/s)')

p.show()

