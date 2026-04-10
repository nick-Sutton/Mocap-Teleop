import numpy as np
import h5py
import matplotlib.pyplot as p
from scipy.spatial.transform import Rotation as R

p.rcParams.update({
  "text.usetex": True,
  "font.size": 22})

file = h5py.File("data/latest.h5",'r')
#file = h5py.File("te_bayesian/video.h5",'r')
#file = h5py.File("",'r')
data = file['time_series']

p.plot((data[1:]['epoch_time']-data[:-1]['epoch_time'])*1000)
#p.plot((data[1:]['tictoc']-data[:-1]['tictoc']))
#print(np.mean((data[1:]['epoch_time']-data[:-1]['epoch_time'])*1000))
p.plot(data['cycle_duration']+2)
#p.plot(data['epoch_time'])
print(np.mean(data['cycle_duration']))
#p.legend(['q\_rx','q\_ry','q\_rz','qd\_rx','qd\_ry','qd\_rz'])
#p.legend(['qd\_x','qd\_y','qd\_z'])
#p.plot(mode)
p.grid()
#p.title('ID-QP Standing with Velocity Safety Constraint')
p.xlabel('time (ms)')
#p.ylabel('$\dot{q}$ (rad/s)')

p.show()

