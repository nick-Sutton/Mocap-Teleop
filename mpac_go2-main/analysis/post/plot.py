import numpy as np
import h5py
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as p
# from sklearn.linear_model import LinearRegression

p.rcParams.update({
  "text.usetex": False,
  "font.size": 12})

file = h5py.File("build/data/latest.h5",'r')
data = file['common_timeseries']

#qd = data['qd'][7500:9500,6:-1]
time = data['epoch_time']
qd_raw = data['qd']
q = data['q']
f = data['f']

M = 15.2062
g = 9.81
mu = 0.7

N = len(q)
x_data = np.zeros(N-1)
y_data = np.zeros(N-1)

# moving average
window_size =100
qd = np.convolve(qd_raw[:,0], np.ones(window_size)/window_size, mode='valid')
L = len(qd)
qdd = np.zeros(L)
O = 14300

# simplified estimation
for i in range(L-1):
  delta_t = time[i+1]-time[i]
  qdd[i] = (qd[i+1] - qd[i])/delta_t
  x_data[i] = qdd[i] + mu*g
  y_data[i] = M*qdd[i] - (f[i,0]+f[i,1]+f[i,2]+f[i,3])


x_data1 = x_data[0:O]
y_data1 = y_data[0:O]

L1 = len(x_data1)

x_data2 = x_data[O:-1]
y_data2 = y_data[O:-1]

L2 = len(x_data2)

# Linear regression
X_mean = np.mean(x_data2)
y_mean = np.mean(y_data2)

m = np.sum((x_data2 - X_mean) * (y_data2 - y_mean)) / np.sum((x_data2 - X_mean)**2)
b = y_mean - m * X_mean


print(f"Slope (m): {m}, Intercept (b): {b}")

y_pred = m*x_data2 + b

#N=10
#for i in range(len(qd[0])):
#  mov_avg = np.convolve(qd[:,i], np.ones((N,))/N, mode='valid'

p.figure()
p.plot(data['epoch_time'][:] - data['epoch_time'][0],data['f'][:])
p.grid()
p.axhline(y=1, color='r', linestyle='--')
p.axhline(y=-1, color='r', linestyle='--')
#p.ylim(-1.9,2.6)
#p.xlim(-170,3850)
#p.yticks([-1,0,1,2])
p.title('force')
p.xlabel('time (ms)')
p.ylabel('f_x (N)')
p.show(block=False)

p.figure()
# p.plot(data['epoch_time'][:] - data['epoch_time'][0],data['q'][:,0])
# p.plot(data['epoch_time'][:] - data['epoch_time'][0],data['q'][:,1])
p.plot(data['q'][:,0],data['q'][:,1])
p.grid()
# p.axhline(y=1, color='r', linestyle='--')
# p.axhline(y=-1, color='r', linestyle='--')
#p.ylim(-1.9,2.6)
#p.xlim(-170,3850)
#p.yticks([-1,0,1,2])
p.title('position')
p.xlabel('p_x (m)')
p.ylabel('p_y (m)')
p.show(block=False)

p.figure()
p.plot(data['epoch_time'][:] - data['epoch_time'][0],data['qd'][:,0])
p.plot(data['epoch_time'][0:L]- data['epoch_time'][0], qd)
p.grid()
p.axhline(y=1, color='r', linestyle='--')
p.axhline(y=-1, color='r', linestyle='--')
#p.ylim(-1.9,2.6)
#p.xlim(-170,3850)
#p.yticks([-1,0,1,2])
p.title('velocity')
p.xlabel('time (ms)')
p.ylabel('v_x (m/x)')
p.show(block=False)

p.figure()
p.plot(data['epoch_time'][0:L] - data['epoch_time'][0],qdd)
p.grid()
p.axhline(y=1, color='r', linestyle='--')
p.axhline(y=-1, color='r', linestyle='--')
#p.ylim(-1.9,2.6)
#p.xlim(-170,3850)
#p.yticks([-1,0,1,2])
p.title('acceleration')
p.xlabel('time (ms)')
p.ylabel('a_x (m/x)')
p.show(block=False)

p.figure()
p.plot(data['epoch_time'][0:O] - data['epoch_time'][0],y_data1)
p.grid()
p.axhline(y=1, color='r', linestyle='--')
p.axhline(y=-1, color='r', linestyle='--')
#p.ylim(-1.9,2.6)
#p.xlim(-170,3850)
#p.yticks([-1,0,1,2])
p.title('force offset')
p.xlabel('time (ms)')
p.ylabel('F (N)')
p.show(block=False)


p.figure()
p.scatter(x_data2, y_data2, color='b', marker='o', alpha=0.7, label="Data points")
p.scatter(x_data1, y_data1, color='y', marker='o', alpha=0.7, label="Data points")
p.plot(x_data2, y_pred, color="red", label="Regression Line")
# Add labels and title
p.xlabel("X-axis")
p.ylabel("Y-axis")
p.title("Scatter Plot Example")
p.show()