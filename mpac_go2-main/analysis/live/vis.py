import sys

sys.path.insert(1, "mpac_core/vis")
from vis_core import *

import math

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel

import numpy as np
from pyqtgraph.ptime import time
import pyqtgraph.opengl as gl
from pyqtgraph import Vector

import copy

#pg.setConfigOptions(antialias=True)
filename = 'build/data/latest.h5'
if len(sys.argv) > 1:
  filename = sys.argv[1]

vis = VisWindow(filename)
vis.setFont(QtGui.QFont("Roboto"))

############# 3D VIEW LAYOUT ########################
view3D = DataLayout()
vis.addVisLayout("3D view", view3D)
robot_view = RobotViewWidget('robot/robot.urdf', vis.terrain)
robot_view.opts['distance'] = 2.5
view3D.addWidget(robot_view, 2, 1, 2, 1)
avg_win = 100

def view3D_update(t_min, t_max, t_curr):
  x_mean = np.sum(vis.data['common_timeseries'][max(0,t_curr-avg_win):t_curr]['q'][:,0])/avg_win
  y_mean = np.sum(vis.data['common_timeseries'][max(0,t_curr-avg_win):t_curr]['q'][:,1])/avg_win
  z_mean = np.sum(vis.data['common_timeseries'][max(0,t_curr-avg_win):t_curr]['q'][:,2])/avg_win
  if (not math.isnan(x_mean) and
      not math.isnan(y_mean) and
      not math.isnan(z_mean)):
    robot_view.setCameraPosition(Vector(x_mean, y_mean, 0))
  robot_view.update_data(t_curr, vis.data, vis.attributes)

view3D.update = view3D_update

vis.switchVisLayout(0)

#################### MAIN #################################

if __name__ == '__main__':
  vis.run()
