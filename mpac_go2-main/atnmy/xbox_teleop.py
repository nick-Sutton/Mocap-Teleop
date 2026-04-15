from evdev import InputDevice, ecodes, list_devices
from enum import Enum
from mpac_cmd import *
import os
import psutil    

devices = [InputDevice(path) for path in list_devices()]
for device in devices:
  if device.name == "Xbox Wireless Controller":
    break;

print(device)
#print(device.capabilities(verbose=True))

class Mode(Enum):
  DEFAULT = 0
  FREE_WALK = 1
  FREE_STAND = 2

Mode = Mode.DEFAULT

vx_lim = [-0.2,0.2]
vy_lim = [-0.15,0.15]
vrz_lim = [-0.5,0.5]

rx_lim = [-0.3,0.3]
ry_lim = [-0.3,0.3]
rz_lim = [-0.3,0.3]

free_walk_vx = 0
free_walk_vy = 0
free_walk_vrz = 0

free_stand_rx = 0
free_stand_ry = 0
free_stand_rz = 0

left_analog_x = 0
left_analog_y = 0

supress_output = False


#xbox_mappings = {
#  'D_PAD_X': 16,
#  'D_PAD_Y': 17,
#  'BTN_A': 304,
#  'BTN_B': 305,
#  'BTN_X': 306,
#  'BTN_Y': 307,
#  'BTN_LB': 308,
#  'BTN_RB': 309,
#  'BTN_VIEW': 310,
#  'BTN_MENU': 311,
#                }

def free_walk_on():
  global Mode
  Mode = Mode.FREE_WALK
  walk_idqp()

def stand_mode_on():
  global Mode
  Mode = Mode.FREE_STAND
  stand_idqp()

def handle_analog_sticks(event):
  global supress_output
  supress_output = True
  global Mode
  global free_walk_vx
  global free_walk_vy
  global free_walk_vrz
  global free_stand_rx
  global free_stand_ry
  global free_stand_rz
  if Mode == Mode.FREE_WALK:
    if event.code == 0: #left X
      free_walk_vy = (1-event.value/65535.)*(vy_lim[1]-vy_lim[0])+vy_lim[0]
    if event.code == 1: #left y
      free_walk_vx = (1-event.value/65535.)*(vx_lim[1]-vx_lim[0])+vx_lim[0]
    if event.code == 3: #right x
      free_walk_vrz = (1-event.value/65535.)*(vrz_lim[1]-vrz_lim[0])+vrz_lim[0]
    walk_idqp(vx=free_walk_vx,vy=free_walk_vy,vrz=free_walk_vrz)

  elif Mode == Mode.FREE_STAND:
    if event.code == 0: #left X
      free_stand_rx = -((1-event.value/65535.)*(ry_lim[1]-ry_lim[0])+ry_lim[0])
    if event.code == 1: #left y
      free_stand_ry = (1-event.value/65535.)*(rx_lim[1]-rx_lim[0])+rx_lim[0]
    if event.code == 3: #right x
      free_stand_rz = (1-event.value/65535.)*(rz_lim[1]-rz_lim[0])+rz_lim[0]
    #round poses. TODO something weird happening in the QP with commands close 
    #together in quick succession...
    quat = 0.1
    
    stand_idqp(rx=free_stand_rx,ry=free_stand_ry,rz=free_stand_rz)
  else:
    pass

def toggle_ctrl():
  if ("ctrl" in (p.name() for p in psutil.process_iter())):
    print("stopping ctrl and tlm")
    #os.system("sudo killall -s SIGINT ctrl &")
    os.system("killall -s SIGINT ctrl &")
    os.system("killall -s SIGINT tlm &")
  else:
    os.system("./build/ctrl&")
    #os.system("sudo ./build/ctrl --io_mode=hardware &")
    os.system("./build/tlm &")
    print("starting ctrl and tlm")

#                name,                 event.type, event.code, event.value, reset_mode,                         function, message
xbox_mappings = [['D_PAD_LEFT',     ecodes.EV_ABS,         16,          -1,       True, lambda:walk_idqp(vy= 0.15), "walk_left"],
                 ['D_PAD_RIGHT',    ecodes.EV_ABS,         16,           1,       True, lambda:walk_idqp(vy=-0.15), "walk_right"],
                 ['D_PAD_UP',       ecodes.EV_ABS,         17,          -1,       True, lambda:walk_idqp(vx= 0.2),  "walk_forward"],
                 ['D_PAD_DOWN',     ecodes.EV_ABS,         17,           1,       True, lambda:walk_idqp(vx=-0.2),  "walk_backward"],
                 ['BTN_A',          ecodes.EV_KEY,        304,           1,       True, lambda:lie(),               "lie"],
                 ['BTN_B',          ecodes.EV_KEY,        305,           1,       True, stand_mode_on,              "stand_idqp"],
                 ['BTN_X',          ecodes.EV_KEY,        306,           1,       True, free_walk_on,               "free_walk_on"],
                 ['BTN_Y',          ecodes.EV_KEY,        307,           1,       True, lambda:jump(),              "jump"],
                 ['BTN_LB',         ecodes.EV_KEY,        308,           1,       True, lambda:walk_idqp(vrz=0.5),  "rotate_ccw"],
                 ['BTN_RB',         ecodes.EV_KEY,        309,           1,       True, lambda:walk_idqp(vrz=-0.5), "rotate_cw"],
                 ['BTN_VIEW',       ecodes.EV_KEY,        310,           1,       True, toggle_ctrl,                "toggling ctrl"],
                 ['BTN_MENU',       ecodes.EV_KEY,        311,           1,       True, lambda:soft_stop(),         "soft_stop"],
                 ['LEFT_ANALOG_X',  ecodes.EV_ABS,          0,      'args',      False, handle_analog_sticks,       "left_analog_x"],
                 ['LEFT_ANALOG_Y',  ecodes.EV_ABS,          1,      'args',      False, handle_analog_sticks,       "left_analog_y"],
                 ['RIGHT_ANALOG_X', ecodes.EV_ABS,          3,      'args',      False, handle_analog_sticks,       "right_analog_x"],
                 ['RIGHT_ANALOG_Y', ecodes.EV_ABS,          4,      'args',      False, handle_analog_sticks,       "right_analog_y"],
                 ]


ctrl_running = False


for event in device.read_loop():
  #print(event)
  #print(Mode)
  for x in xbox_mappings:
    if ((x[1] == event.type and x[2] == event.code) and
        (x[3] == event.value or x[3] == 'args')):
      if x[4]:
        Mode = Mode.DEFAULT
      if (x[3] == 'args'):
        x[5](event)
      else:
        x[5]()
      if not supress_output:
        print(x[0] + ": " + x[6])
      supress_output = False
