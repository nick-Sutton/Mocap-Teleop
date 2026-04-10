from atnmy.mpac_cmd import *
from time import sleep
from math import atan2, sin, cos, sqrt, pi
# from qpsolvers import solve_qp

def min_angle_diff(th1, th2):
  return (th1-th2) - 2*pi * round((th1-th2)/(2*pi));

# simple proportional control with saturation
def walk_to(x, y, th):
  tlm_data = get_tlm_data()
  K = [1, 1, 1]
  v_max = [0.2, 0.1, 0.3]
  x_delta = tlm_data['q'][0] - x
  y_delta = tlm_data['q'][1] - y
  th_curr = tlm_data['q'][5]
  th_delta = min_angle_diff(th_curr, th)
  while (abs(x_delta) > 0.1 or
         abs(y_delta) > 0.1 or
         abs(th_delta) > 0.1):
    x_body_frame = x_delta*cos(-th_curr) - y_delta*sin(-th_curr)
    y_body_frame = x_delta*sin(-th_curr) + y_delta*cos(-th_curr)
    x_vel_cmd = -K[0]*(x_body_frame)
    y_vel_cmd = -K[1]*(y_body_frame)
    th_vel_cmd = -K[2]*th_delta
    speed_scale = 1
    if (abs(x_vel_cmd) > v_max[0]):
      speed_scale = min(speed_scale,v_max[0]/abs(x_vel_cmd))
    if (abs(y_vel_cmd) > v_max[1]):
      speed_scale = min(speed_scale,v_max[1]/abs(y_vel_cmd))
    if (abs(th_vel_cmd) > v_max[2]):
      speed_scale = min(speed_scale,v_max[2]/abs(th_vel_cmd))

    x_vel_cmd*=speed_scale
    y_vel_cmd*=speed_scale
    th_vel_cmd*=speed_scale
 
    walk_idqp(vx=x_vel_cmd,vy=y_vel_cmd,vrz=th_vel_cmd)
    while (tlm_data['q'][2] < 0.2):
      sleep(0.5)
      tlm_data = get_tlm_data()
    sleep(0.1)
    tlm_data = get_tlm_data()
    x_delta = tlm_data['q'][0] - x
    y_delta = tlm_data['q'][1] - y
    th_curr = tlm_data['q'][5]
    th_delta = min_angle_diff(th_curr, th)
  walk_idqp(vx=0,vy=0,vrz=0)
  sleep(1)
  stand_idqp()

# unicycle model walk to
def walk_to_unicycle(x, y, th):
  tlm_data = get_tlm_data()
  K = [1, 1]
  v_max = [0.2, 0.3]
  x_delta = tlm_data['q'][0] - x
  y_delta = tlm_data['q'][1] - y
  p_delta = sqrt(x_delta**2 + y_delta**2)
  th_curr = tlm_data['q'][5]
  th_delta = min_angle_diff(tlm_data['q'][5], atan2(-y_delta, -x_delta))
  while (p_delta > 0.15):
    if (abs(th_delta) < pi/4):
      x_vel_cmd = K[0]*p_delta
      x_vel_cmd = min(v_max[0],x_vel_cmd)
    else:
      x_vel_cmd = 0
    th_vel_cmd = -K[1]*th_delta
    th_vel_cmd = max(-v_max[1],min(v_max[1],th_vel_cmd))

    walk_idqp(vx=x_vel_cmd,vy=0,vrz=th_vel_cmd)
    while (tlm_data['q'][2] < 0.2):
      sleep(0.5)
      tlm_data = get_tlm_data()
    sleep(0.1)
    tlm_data = get_tlm_data()
    x_delta = tlm_data['q'][0] - x
    y_delta = tlm_data['q'][1] - y
    p_delta = sqrt(x_delta**2 + y_delta**2)
    th_delta = min_angle_diff(tlm_data['q'][5], atan2(-y_delta, -x_delta))
  walk_to(x, y, th)


def set_cbf(x, y, r, epsilon):
  tlm_data = get_tlm_data()
  x_m = tlm_data['q'][0]
  y_m = tlm_data['q'][1]

  h = (x_m-x)*(x_m-x) + (y_m-y)*(y_m-y) - (r+epsilon)




