import numpy as np
import h5py
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as p

p.rcParams.update({
  "text.usetex": True,
  "font.size": 22})

file = h5py.File("data/latest.h5",'r')
data = file['time_series']

for i_init in range(6000,len(data)):
  if data['ctrl_mode_curr'][i_init].decode('ascii') == 'stand_qp':
    break

print(i_init)

p.plot(data['epoch_time'][i_init:] - data['epoch_time'][i_init],data['q'][i_init:,6:]-data['q_des'][i_init:,:])
p.grid()
#p.ylim(-1.9,2.6)
#p.xlim(-170,3850)
#p.yticks([-1,0,1,2])
p.title('Position Tracking')
p.xlabel('time (ms)')
p.ylabel('$q$ (rad)')

p.show()
