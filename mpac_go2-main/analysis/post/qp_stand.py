import numpy as np
import h5py
import matplotlib.pyplot as p
from datetime import datetime
from scipy.spatial.transform import Rotation as R

p.rcParams.update({
  "text.usetex": True,
  "font.size": 22})

file = h5py.File("data/latest.h5",'r')
data = file['time_series']
i_init = 0
for i_init in range(20000,len(data)):
  if data['ctrl_mode_curr'][i_init].decode('ascii') == 'stand_safe_qp':
    break

i_init -= 600
i_final = i_init+3500

N=10
for i in range(len(data['qd'][0,:])):
  mov_avg = np.convolve(data['qd'][i_init:i_final,i], np.ones((N,))/N, mode='valid')
  p.plot(1000*(data['epoch_time'][i_init:i_final-N+1] - data['epoch_time'][i_init]),mov_avg)

#p.plot(data['epoch_time'][i_init:] - data['epoch_time'][i_init],data['qd'][i_init:,:])
p.grid()
p.axhline(y=1, color='r', linestyle='--')
p.axhline(y=-1, color='r', linestyle='--')
p.ylim(-1.9,2.6)
p.xlim(-170,3850)
p.yticks([-1,0,1,2])
p.title('Safe ID-QP Standing $u$ control')
p.xlabel('time (ms)')
p.ylabel('$\dot{q}$ (rad/s)')

p.show()
