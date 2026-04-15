import socket
import numpy as np
import atexit
import threading
from ctypes import *
import os
import importlib

has_rerun = False
rerun_initialized = False
try:
  import time
  import rerun as rr
  from mpac_rerun.robot_logger import RobotLogger
  import scipy.spatial.transform as st

  has_rerun = True
except ImportError:
  has_rerun = False

logger_running = False
rerun_thrd = None

UDP_IP = "127.0.0.1"

ctr_ports = list(map(int, os.environ.get("CTRL_PORTS", "8081").split(",")))
atn_ports = list(map(int, os.environ.get("ATNMY_PORTS", "8082").split(",")))
assert len(ctr_ports) == len(atn_ports), f"Number of UDP ports must match number of ATN ports, got {len(ctr_ports)} and {len(atn_ports)}"

def start_logger():
  global rerun_initialized
  if not rerun_initialized:
    rr.init("robot_logger", spawn=False, recording_id="robot_logger")
    rerun_initialized = True
  else:
    try:
      importlib.reload(rr)
      rr.init("robot_logger", spawn=False, recording_id="robot_logger")
    except:
      pass
  global logger_running
  logger_running = True

  def rerun_thread():
    print("Initializing logger...")
    loggers = [RobotLogger.from_zoo("go2") for _ in range(len(ctr_ports))]
    t = time.time()
    for logger in loggers:
      logger.log_initial_state(logtime=t)
    print("Logger started!")
    while logger_running:
      t = time.time()
      for logger, rerun_state in zip(loggers, get_rerun_state()):
        logger.log_state(logtime=t, **rerun_state)
      time.sleep(1 / 20)

    log_path = f"{os.environ['HOME']}/.cache/mpac/rerun_logs/robot_logger_{int(time.time())}.rrd"
    # Create folder for rerun logs if it doesn't exist
    print(f"Saving log to {log_path}")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    rr.save(log_path)

  global rerun_thrd
  rerun_thrd = threading.Thread(target=rerun_thread)
  rerun_thrd.daemon = False
  rerun_thrd.start()


def stop_logger():
  global logger_running
  logger_running = False
  global rerun_thrd
  rerun_thrd.join()


def get_rerun_state():
  tlm_data = get_tlm_data()

  if len(ctr_ports) == 1:
    ttlm_data = [tlm_data]
  else:
    ttlm_data = tlm_data

  outs = list()
  for i in range(len(ctr_ports)):
    out = dict()

    joint_positions = ttlm_data[i]["q"][6:]
    out_joint = dict()
    out_joint["FL_hip_joint"] = joint_positions[0]
    out_joint["FL_thigh_joint"] = joint_positions[1]
    out_joint["FL_calf_joint"] = joint_positions[2]
    out_joint["FR_hip_joint"] = joint_positions[3]
    out_joint["FR_thigh_joint"] = joint_positions[4]
    out_joint["FR_calf_joint"] = joint_positions[5]
    out_joint["RL_hip_joint"] = joint_positions[6]
    out_joint["RL_thigh_joint"] = joint_positions[7]
    out_joint["RL_calf_joint"] = joint_positions[8]
    out_joint["RR_hip_joint"] = joint_positions[9]
    out_joint["RR_thigh_joint"] = joint_positions[10]
    out_joint["RR_calf_joint"] = joint_positions[11]

    out["joint_positions"] = out_joint
    out["base_position"] = ttlm_data[i]["q"][:3]
    roll, pitch, yaw = ttlm_data[i]["q"][3:6]
    out["base_orientation"] = st.Rotation.from_euler("xyz", [roll, pitch, yaw]).as_quat()
    outs.append(out)
  return outs


class buff(Structure):
  _fields_ = [("mode", c_int), ("pad", c_int), ("disc", c_int * 16), ("cont", c_double * 16)]



print("UDP target IP:", UDP_IP)
print("UDP target port:", ctr_ports)
print("ATNMY target port:", atn_ports)

sockets = [socket.socket(socket.AF_INET, socket.SOCK_DGRAM) for _ in range(len(ctr_ports))]
for i, port in enumerate(atn_ports):
  sockets[i].bind((UDP_IP, port))


def hard_stop():
  x = buff()
  x.mode = 0
  for i, port in enumerate(ctr_ports):
    sockets[i].sendto(bytes(x), (UDP_IP, port))


def soft_stop():
  print("Soft Stop")
  x = buff()
  x.mode = 1
  for i, port in enumerate(ctr_ports):
    sockets[i].sendto(bytes(x), (UDP_IP, port))


def lie():
  x = buff()
  x.mode = 2
  for i, port in enumerate(ctr_ports):
    sockets[i].sendto(bytes(x), (UDP_IP, port))


def stand_idqp(h=0.25, rx=0, ry=0, rz=0):
  x = buff()
  x.mode = 5
  x.cont[0] = h
  x.cont[1] = rx
  x.cont[2] = ry
  x.cont[3] = rz
  for i, port in enumerate(ctr_ports):
    sockets[i].sendto(bytes(x), (UDP_IP, port))


def walk_pd(gait=0):
  x = buff()
  x.mode = 7
  x.disc[0] = gait
  for i, port in enumerate(ctr_ports):
    sockets[i].sendto(bytes(x), (UDP_IP, port))


def walk_idqp(h=0.25, vx=0, vy=0, vrz=0):
  x = buff()
  x.mode = 8
  x.cont[0] = h
  x.cont[1] = vx
  x.cont[2] = vy
  x.cont[3] = vrz
  for i, port in enumerate(ctr_ports):
    sockets[i].sendto(bytes(x), (UDP_IP, port))


def walk_quasi_idqp(h=0.25, vx=0, vy=0, vrz=0):
  x = buff()
  x.mode = 9
  x.cont[0] = h
  x.cont[1] = vx
  x.cont[2] = vy
  x.cont[3] = vrz
  for i, port in enumerate(ctr_ports):
    sockets[i].sendto(bytes(x), (UDP_IP, port))


def walk_quasi_planned(h=0.22, vx=0, vy=0, vrz=0, x=0, y=0, rz=0, mode="vel"):
  b = buff()
  b.mode = 10
  b.disc[0] = 1 if mode == "pos" else 0
  b.cont[0] = h
  b.cont[1] = vx
  b.cont[2] = vy
  b.cont[3] = vrz
  b.cont[4] = x
  b.cont[5] = y
  b.cont[6] = rz
  for i, port in enumerate(ctr_ports):
    sockets[i].sendto(bytes(b), (UDP_IP, port))


def bound(vx=0):
  x = buff()
  x.mode = 11
  x.cont[0] = vx
  for i, port in enumerate(ctr_ports):
    sockets[i].sendto(bytes(x), (UDP_IP, port))


def jump(x_vel=0, y_vel=0, z_vel=2):
  x = buff()
  x.mode = 12
  x.cont[0] = x_vel
  x.cont[1] = y_vel
  x.cont[2] = z_vel
  for i, port in enumerate(ctr_ports):
    sockets[i].sendto(bytes(x), (UDP_IP, port))


def land():
  x = buff()
  x.mode = 13
  for i, port in enumerate(ctr_ports):
    sockets[i].sendto(bytes(x), (UDP_IP, port))


def traj_track(traj_num=0):
  x = buff()
  x.mode = 6
  x.disc[0] = traj_num
  for i, port in enumerate(ctr_ports):
    sockets[i].sendto(bytes(x), (UDP_IP, port))


def calibrate(traj_num=0):
  x = buff()
  x.mode = 14
  x.disc[0] = traj_num
  for i, port in enumerate(ctr_ports):
    sockets[i].sendto(bytes(x), (UDP_IP, port))


# run soft_stop when script is exited
atexit.register(soft_stop)

# order for q is: x, y, z,
#                rx, ry, rz,
#                fl1, fl2, fl3,
#                fr1, fr2, fr3,
#                bl1, bl2, bl3,
#                br1, br2, br3
# order for u is: fl1, fl2, fl3,
#                fr1, fr2, fr3,
#                bl1, bl2, bl3,
#                br1, br2, br3
tlm_types = np.dtype(
  [
    ("start_time_sec", np.int64),
    ("start_time_nano", np.int64),
    ("cycle_count", np.uint64),
    ("cycle_duration", np.double),
    ("compute_duration", np.double),
    ("tictoc", np.double),
    ("path_compute_duration", np.double),
    ("q", np.double, (18,)),
    ("qd", np.double, (18,)),
    ("qdd_sim", np.double, (18,)),
    ("u", np.double, (12,)),
    ("act_mode", np.int32, (12,)),
    ("u_des", np.double, (12,)),
    ("q_des", np.double, (12,)),
    ("qd_des", np.double, (12,)),
    ("f", np.double, (4,)),
    ("temp", np.double, (12,)),
    ("ctrl_curr", np.int32, (1,)),
    ("ctrl_curr_disc_args", np.int32, (16,)),
    ("ctrl_curr_cont_args", np.double, (16,)),
    ("ctrl_next", np.int32, (1,)),
    ("ctrl_next_disc_args", np.int32, (16,)),
    ("ctrl_next_cont_args", np.double, (16,)),
    ("ctrl_des", np.int32, (1,)),
    ("ctrl_des_disc_args", np.int32, (16,)),
    ("ctrl_des_cont_args", np.double, (16,)),
    ("prim_path_len", np.int32, (1,)),
    ("prim_path_1", np.int32, (1,)),
    ("prim_path_1_disc_args", np.int32, (16,)),
    ("prim_path_1_cont_args", np.double, (16,)),
    ("prim_path_2", np.int32, (1,)),
    ("prim_path_2_disc_args", np.int32, (16,)),
    ("prim_path_2_cont_args", np.double, (16,)),
    ("prim_path_3", np.int32, (1,)),
    ("prim_path_3_disc_args", np.int32, (16,)),
    ("prim_path_3_cont_args", np.double, (16,)),
    ("prim_path_4", np.int32, (1,)),
    ("prim_path_4_disc_args", np.int32, (16,)),
    ("prim_path_4_cont_args", np.double, (16,)),
    ("prim_path_5", np.int32, (1,)),
    ("prim_path_5_disc_args", np.int32, (16,)),
    ("prim_path_5_cont_args", np.double, (16,)),
    ("prim_path_6", np.int32, (1,)),
    ("prim_path_6_disc_args", np.int32, (16,)),
    ("prim_path_6_cont_args", np.double, (16,)),
    ("prim_path_7", np.int32, (1,)),
    ("prim_path_7_disc_args", np.int32, (16,)),
    ("prim_path_7_cont_args", np.double, (16,)),
    ("prim_path_8", np.int32, (1,)),
    ("prim_path_8_disc_args", np.int32, (16,)),
    ("prim_path_8_cont_args", np.double, (16,)),
  ],
  align=True,
)
locks = [threading.Lock() for _ in range(len(ctr_ports))]
tlm_data = [None] * len(ctr_ports)


def get_tlm_data():
  if len(ctr_ports) == 1:
    return tlm_data[0]
  else:
    return tlm_data


def tlm_read_thread(socket_idx):
  global tlm_data
  buf = None
  while True:
    buf, addr = sockets[socket_idx].recvfrom(3248)  # size of tlm packet
    buf_size = len(buf)
    element_size = np.dtype(tlm_types).itemsize

    if buf_size % element_size != 0:
      raise ValueError(f"Buffer size ({buf_size}) must be a multiple of element size ({element_size}).")

    if buf:
      with locks[socket_idx]:
        tlm_data[socket_idx] = np.frombuffer(buf, dtype=tlm_types)[0]

t_threads = [threading.Thread(target=tlm_read_thread, args=(i,)) for i in range(len(ctr_ports))]
for t in t_threads:
  t.daemon = True
  t.start()

if __name__ == "__main__":
  import argparse
  import tkinter as tk
  from tkinter import ttk

  parser = argparse.ArgumentParser()
  parser.add_argument("--scale", type=float, default=1.0)
  args = parser.parse_args()

  class RobotControlGUI:
    def __init__(self, root, scale):
      self.root = root
      self.root.title("Robot Control Interface")

      # Add keyboard binding for quit
      self.root.bind("<q>", self.quit)
      self.root.bind("<Q>", self.quit)

      # Add number key bindings
      self.root.bind("1", lambda _: self.execute_hard_stop())
      self.root.bind("2", lambda _: self.execute_soft_stop())
      self.root.bind("3", lambda _: self.execute_lie())
      self.root.bind("4", lambda _: self.execute_land())
      self.root.bind("5", lambda _: self.start_logging())
      self.root.bind("6", lambda _: self.stop_logging())
      self.root.bind("7", lambda _: self.execute_stand())
      self.root.bind("8", lambda _: self.execute_walk_idqp())
      self.root.bind("9", lambda _: self.execute_jump())

      # Configure font styles
      base_font_size = 30
      self.title_font = ("Helvetica", int(base_font_size * scale), "bold")
      self.button_font = ("Helvetica", int(base_font_size * scale), "bold")
      self.label_font = ("Helvetica", int(base_font_size * scale), "bold")
      self.joystick_font = ("Helvetica", int(base_font_size * scale), "bold")

      # Configure styles
      style = ttk.Style()
      style.configure("Bold.TLabelframe.Label", font=self.title_font)
      style.configure("Bold.TButton", font=self.button_font)
      style.configure("Bold.TLabel", font=self.label_font)

      # Create main frame with padding
      self.main_frame = ttk.Frame(root, padding="10")
      self.main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

      # Stop Commands Frame
      self.stop_frame = ttk.LabelFrame(self.main_frame, text="Stop Commands", padding="5", style="Bold.TLabelframe")
      self.stop_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
      tk.Button(
        self.stop_frame,
        text="[1] Hard Stop",
        command=self.execute_hard_stop,
        font=self.button_font,
        bg="red",
        fg="white",
      ).grid(row=0, column=0, padx=5, pady=2)
      tk.Button(
        self.stop_frame,
        text="[2] Soft Stop",
        command=self.execute_soft_stop,
        font=self.button_font,
        bg="green",
        fg="white",
      ).grid(row=0, column=1, padx=5, pady=2)

      # Basic Movement Frame
      self.basic_move_frame = ttk.LabelFrame(
        self.main_frame, text="Basic Movement", padding="5", style="Bold.TLabelframe"
      )
      self.basic_move_frame.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
      tk.Button(self.basic_move_frame, text="[3] Lie", command=self.execute_lie, font=self.button_font).grid(
        row=0, column=0, padx=5, pady=2
      )
      tk.Button(self.basic_move_frame, text="[4] Land", command=self.execute_land, font=self.button_font).grid(
        row=0, column=1, padx=5, pady=2
      )

      # Logging Frame
      self.logging_frame = ttk.LabelFrame(
        self.main_frame, text="Logging Controls", padding="5", style="Bold.TLabelframe"
      )
      self.logging_frame.grid(row=2, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))
      tk.Button(self.logging_frame, text="[5] Start Logging", command=self.start_logging, font=self.button_font).grid(
        row=0, column=0, padx=5, pady=2
      )
      tk.Button(self.logging_frame, text="[6] Stop Logging", command=self.stop_logging, font=self.button_font).grid(
        row=0, column=1, padx=5, pady=2
      )

      # Stand IDQP Frame
      self.stand_frame = ttk.LabelFrame(self.main_frame, text="Stand Control", padding="5", style="Bold.TLabelframe")
      self.stand_frame.grid(row=3, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))

      self.h_var = tk.StringVar(value="0.25")
      self.rx_var = tk.StringVar(value="0")
      self.ry_var = tk.StringVar(value="0")
      self.rz_var = tk.StringVar(value="0")

      tk.Button(self.stand_frame, text="[7] Stand", command=self.execute_stand, font=self.button_font).grid(
        row=0, column=0, padx=5, pady=2
      )
      ttk.Label(self.stand_frame, text="Height:", style="Bold.TLabel").grid(row=0, column=1, padx=5)
      tk.Entry(self.stand_frame, textvariable=self.h_var, width=10, font=self.label_font).grid(row=0, column=2, padx=5)
      ttk.Label(self.stand_frame, text="Roll:", style="Bold.TLabel").grid(row=0, column=3, padx=5)
      tk.Entry(self.stand_frame, textvariable=self.rx_var, width=10, font=self.label_font).grid(row=0, column=4, padx=5)
      ttk.Label(self.stand_frame, text="Pitch:", style="Bold.TLabel").grid(row=0, column=5, padx=5)
      tk.Entry(self.stand_frame, textvariable=self.ry_var, width=10, font=self.label_font).grid(row=0, column=6, padx=5)
      ttk.Label(self.stand_frame, text="Yaw:", style="Bold.TLabel").grid(row=0, column=7, padx=5)
      tk.Entry(self.stand_frame, textvariable=self.rz_var, width=10, font=self.label_font).grid(row=0, column=8, padx=5)

      # Walk Frame
      self.walk_frame = ttk.LabelFrame(self.main_frame, text="Walk Control", padding="5", style="Bold.TLabelframe")
      self.walk_frame.grid(row=4, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))

      self.walk_h_var = tk.StringVar(value="0.25")
      self.walk_vx_var = tk.StringVar(value="0")
      self.walk_vy_var = tk.StringVar(value="0")
      self.walk_vrz_var = tk.StringVar(value="0")

      tk.Button(self.walk_frame, text="[8] Walk", command=self.execute_walk_idqp, font=self.button_font).grid(
        row=0, column=0, padx=5, pady=2
      )
      ttk.Label(self.walk_frame, text="Height:", style="Bold.TLabel").grid(row=0, column=1, padx=5)
      tk.Entry(self.walk_frame, textvariable=self.walk_h_var, width=10, font=self.label_font).grid(
        row=0, column=2, padx=5
      )
      ttk.Label(self.walk_frame, text="Vx:", style="Bold.TLabel").grid(row=0, column=3, padx=5)
      tk.Entry(self.walk_frame, textvariable=self.walk_vx_var, width=10, font=self.label_font).grid(
        row=0, column=4, padx=5
      )
      ttk.Label(self.walk_frame, text="Vy:", style="Bold.TLabel").grid(row=0, column=5, padx=5)
      tk.Entry(self.walk_frame, textvariable=self.walk_vy_var, width=10, font=self.label_font).grid(
        row=0, column=6, padx=5
      )
      ttk.Label(self.walk_frame, text="Vrz:", style="Bold.TLabel").grid(row=0, column=7, padx=5)
      tk.Entry(self.walk_frame, textvariable=self.walk_vrz_var, width=10, font=self.label_font).grid(
        row=0, column=8, padx=5
      )

      # Jump Frame
      self.jump_frame = ttk.LabelFrame(self.main_frame, text="Jump Control", padding="5", style="Bold.TLabelframe")
      self.jump_frame.grid(row=5, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))

      self.jump_x_var = tk.StringVar(value="0")
      self.jump_y_var = tk.StringVar(value="0")
      self.jump_z_var = tk.StringVar(value="2")

      tk.Button(self.jump_frame, text="[9] Jump", command=self.execute_jump, font=self.button_font).grid(
        row=0, column=0, padx=5, pady=2
      )
      ttk.Label(self.jump_frame, text="X Vel:", style="Bold.TLabel").grid(row=0, column=1, padx=5)
      tk.Entry(self.jump_frame, textvariable=self.jump_x_var, width=10, font=self.label_font).grid(
        row=0, column=2, padx=5
      )
      ttk.Label(self.jump_frame, text="Y Vel:", style="Bold.TLabel").grid(row=0, column=3, padx=5)
      tk.Entry(self.jump_frame, textvariable=self.jump_y_var, width=10, font=self.label_font).grid(
        row=0, column=4, padx=5
      )
      ttk.Label(self.jump_frame, text="Z Vel:", style="Bold.TLabel").grid(row=0, column=5, padx=5)
      tk.Entry(self.jump_frame, textvariable=self.jump_z_var, width=10, font=self.label_font).grid(
        row=0, column=6, padx=5
      )

      # Virtual Joystick Frame
      self.joystick_frame = ttk.LabelFrame(
        self.main_frame, text="Virtual Joysticks", padding="5", style="Bold.TLabelframe"
      )
      self.joystick_frame.grid(row=6, column=0, padx=5, pady=5, sticky=(tk.W, tk.E))

      # Create frame for both joysticks
      self.joysticks_container = ttk.Frame(self.joystick_frame)
      self.joysticks_container.grid(row=0, column=0, padx=5, pady=5)

      # Left Joystick
      self.left_stick_frame = ttk.LabelFrame(
        self.joysticks_container, text="Local vx and wz", padding="30", style="Bold.TLabelframe"
      )
      self.left_stick_frame.grid(row=0, column=0, padx=20, pady=5)

      # Create canvas for left joystick
      self.canvas_size = 300
      self.left_canvas = tk.Canvas(
        self.left_stick_frame,
        width=self.canvas_size,
        height=self.canvas_size,
        bg="white",
        highlightthickness=2,
        highlightbackground="black",
      )
      self.left_canvas.grid(row=0, column=0, padx=5, pady=5)

      # Draw circular boundary and crosshairs for left joystick
      self.left_canvas.create_oval(10, 10, self.canvas_size - 10, self.canvas_size - 10, outline="gray", width=2)
      self.left_canvas.create_line(
        0, self.canvas_size / 2, self.canvas_size, self.canvas_size / 2, fill="gray", dash=(4, 4)
      )
      self.left_canvas.create_line(
        self.canvas_size / 2, 0, self.canvas_size / 2, self.canvas_size, fill="gray", dash=(4, 4)
      )

      # Create handle for left joystick
      self.handle_size = 20
      self.left_handle = self.left_canvas.create_oval(
        self.canvas_size / 2 - self.handle_size / 2,
        self.canvas_size / 2 - self.handle_size / 2,
        self.canvas_size / 2 + self.handle_size / 2,
        self.canvas_size / 2 + self.handle_size / 2,
        fill="red",
      )

      # Right Joystick
      self.right_stick_frame = ttk.LabelFrame(
        self.joysticks_container, text="Local vx and vy", padding="30", style="Bold.TLabelframe"
      )
      self.right_stick_frame.grid(row=0, column=1, padx=20, pady=5)

      # Create canvas for right joystick
      self.right_canvas = tk.Canvas(
        self.right_stick_frame,
        width=self.canvas_size,
        height=self.canvas_size,
        bg="white",
        highlightthickness=2,
        highlightbackground="black",
      )
      self.right_canvas.grid(row=0, column=0, padx=5, pady=5)

      # Draw circular boundary and crosshairs for right joystick
      self.right_canvas.create_oval(10, 10, self.canvas_size - 10, self.canvas_size - 10, outline="gray", width=2)
      self.right_canvas.create_line(
        0, self.canvas_size / 2, self.canvas_size, self.canvas_size / 2, fill="gray", dash=(4, 4)
      )
      self.right_canvas.create_line(
        self.canvas_size / 2, 0, self.canvas_size / 2, self.canvas_size, fill="gray", dash=(4, 4)
      )

      # Create handle for right joystick
      self.right_handle = self.right_canvas.create_oval(
        self.canvas_size / 2 - self.handle_size / 2,
        self.canvas_size / 2 - self.handle_size / 2,
        self.canvas_size / 2 + self.handle_size / 2,
        self.canvas_size / 2 + self.handle_size / 2,
        fill="blue",
      )

      # Create labels for joystick values
      self.left_value_frame = ttk.Frame(self.left_stick_frame)
      self.left_value_frame.grid(row=1, column=0, pady=5)

      self.left_x_label = tk.Label(self.left_value_frame, text="X: 0.00", font=self.joystick_font)
      self.left_x_label.pack(side=tk.LEFT, padx=10)
      self.left_y_label = tk.Label(self.left_value_frame, text="Y: 0.00", font=self.joystick_font)
      self.left_y_label.pack(side=tk.LEFT, padx=10)

      self.right_value_frame = ttk.Frame(self.right_stick_frame)
      self.right_value_frame.grid(row=1, column=0, pady=5)

      self.right_x_label = tk.Label(self.right_value_frame, text="X: 0.00", font=self.joystick_font)
      self.right_x_label.pack(side=tk.LEFT, padx=10)
      self.right_y_label = tk.Label(self.right_value_frame, text="Y: 0.00", font=self.joystick_font)
      self.right_y_label.pack(side=tk.LEFT, padx=10)

      # Bind mouse events
      self.left_canvas.bind("<Button-1>", lambda e: self.start_move(e, "left"))
      self.left_canvas.bind("<B1-Motion>", lambda e: self.move(e, "left"))
      self.left_canvas.bind("<ButtonRelease-1>", lambda e: self.stop_move(e, "left"))

      self.right_canvas.bind("<Button-1>", lambda e: self.start_move(e, "right"))
      self.right_canvas.bind("<B1-Motion>", lambda e: self.move(e, "right"))
      self.right_canvas.bind("<ButtonRelease-1>", lambda e: self.stop_move(e, "right"))

      # Initialize joystick states
      self.left_active = False
      self.right_active = False
      self.left_x_val = 0.0
      self.left_y_val = 0.0
      self.right_x_val = 0.0
      self.right_y_val = 0.0
      self.left_target_x = 0.0
      self.left_target_y = 0.0
      self.right_target_x = 0.0
      self.right_target_y = 0.0

      self.left_x_val_prev = 0.0
      self.left_y_val_prev = 0.0
      self.right_x_val_prev = 0.0
      self.right_y_val_prev = 0.0

      # Start update loop for smooth motion
      self.update_smooth_motion()

    def execute_hard_stop(self):
      print("Hard Stop")
      hard_stop()

    def execute_soft_stop(self):
      print("Soft Stop")
      soft_stop()

    def execute_lie(self):
      print("Lie")
      lie()

    def execute_land(self):
      print("Land")
      land()

    def execute_stand(self):
      try:
        h = float(self.h_var.get())
        rx = float(self.rx_var.get())
        ry = float(self.ry_var.get())
        rz = float(self.rz_var.get())
        print(f"Standing with height: {h}, roll: {rx}, pitch: {ry}, yaw: {rz}")
        stand_idqp(h=h, rx=rx, ry=ry, rz=rz)
      except ValueError:
        print("Invalid input values for stand command")

    def execute_walk_idqp(self):
      try:
        h = float(self.walk_h_var.get())
        vx = float(self.walk_vx_var.get())
        vy = float(self.walk_vy_var.get())
        vrz = float(self.walk_vrz_var.get())
        print(f"Walking with height: {h}, vx: {vx}, vy: {vy}, vrz: {vrz}")
        walk_idqp(h=h, vx=vx, vy=vy, vrz=vrz)
      except ValueError:
        print("Invalid input values for walk IDQP command")

    def execute_walk_quasi_idqp(self):
      try:
        h = float(self.walk_h_var.get())
        vx = float(self.walk_vx_var.get())
        vy = float(self.walk_vy_var.get())
        vrz = float(self.walk_vrz_var.get())
        print(f"Walking with height: {h}, vx: {vx}, vy: {vy}, vrz: {vrz}")
        walk_quasi_idqp(h=h, vx=vx, vy=vy, vrz=vrz)
      except ValueError:
        print("Invalid input values for walk quasi IDQP command")

    def execute_jump(self):
      try:
        x_vel = float(self.jump_x_var.get())
        y_vel = float(self.jump_y_var.get())
        z_vel = float(self.jump_z_var.get())
        print(f"Jumping with x_vel: {x_vel}, y_vel: {y_vel}, z_vel: {z_vel}")
        jump(x_vel=x_vel, y_vel=y_vel, z_vel=z_vel)
      except ValueError:
        print("Invalid input values for jump command")

    def start_logging(self):
      if has_rerun:
        start_logger()
        print("Logging started")
      else:
        print("Rerun package not available - logging not possible")

    def stop_logging(self):
      if has_rerun:
        stop_logger()
        print("Logging stopped")
      else:
        print("Rerun package not available - logging not possible")

    def quit(self, event=None):
      print("Exiting application...")
      soft_stop()  # Ensure robot stops safely before exit
      self.root.quit()

    def start_move(self, event, stick):
      # Calculate target position
      center_x = self.canvas_size / 2
      center_y = self.canvas_size / 2
      dx = event.x - center_x
      dy = event.y - center_y
      distance = (dx * dx + dy * dy) ** 0.5

      # Normalize to circle boundary
      max_radius = (self.canvas_size - 20) / 2  # Account for border
      if distance > max_radius:
        dx = dx * max_radius / distance
        dy = dy * max_radius / distance

      # Update target values
      target_x = dx / max_radius
      target_y = -dy / max_radius

      if stick == "left":
        self.left_active = True
        self.left_target_x = target_x
        self.left_target_y = target_y
      else:
        self.right_target_x = target_x
        self.right_target_y = target_y

    def move(self, event, stick):
      # Calculate target position
      center_x = self.canvas_size / 2
      center_y = self.canvas_size / 2
      dx = event.x - center_x
      dy = event.y - center_y
      distance = (dx * dx + dy * dy) ** 0.5

      # Normalize to circle boundary
      max_radius = (self.canvas_size - 20) / 2  # Account for border
      if distance > max_radius:
        dx = dx * max_radius / distance
        dy = dy * max_radius / distance

      # Update target values
      target_x = dx / max_radius
      target_y = -dy / max_radius

      if stick == "left":
        self.left_target_x = target_x
        self.left_target_y = target_y
      else:
        self.right_target_x = target_x
        self.right_target_y = target_y

    def stop_move(self, event, stick):
      if stick == "left":
        self.left_active = False
        self.left_target_x = 0.0
        self.left_target_y = 0.0
      else:
        self.right_target_x = 0.0
        self.right_target_y = 0.0

    def update_smooth_motion(self):
      # Smooth motion parameters
      smoothing = 0.2  # Adjust this value to change motion speed (0-1)

      # Update left joystick position
      dx = self.left_target_x - self.left_x_val
      dy = self.left_target_y - self.left_y_val

      if abs(dx) < 0.01:
        self.left_x_val = self.left_target_x
      else:
        self.left_x_val += dx * smoothing
      if abs(dy) < 0.01:
        self.left_y_val = self.left_target_y
      else:
        self.left_y_val += dy * smoothing

      # Update left joystick visual position
      center_x = self.canvas_size / 2
      center_y = self.canvas_size / 2
      max_radius = (self.canvas_size - 20) / 2

      canvas_x = center_x + self.left_x_val * max_radius
      canvas_y = center_y - self.left_y_val * max_radius

      self.left_canvas.coords(
        self.left_handle,
        canvas_x - self.handle_size / 2,
        canvas_y - self.handle_size / 2,
        canvas_x + self.handle_size / 2,
        canvas_y + self.handle_size / 2,
      )

      # Update left labels
      self.left_x_label.config(text=f"X: {self.left_x_val:.2f}")
      self.left_y_label.config(text=f"Y: {self.left_y_val:.2f}")

      if abs(self.left_x_val) > 0.01 or abs(self.left_y_val) > 0.01:
        print(f"Left Joystick: X={self.left_x_val:.2f}, Y={self.left_y_val:.2f}")

      if self.left_x_val != self.left_x_val_prev or self.left_y_val != self.left_y_val_prev:
        print(f"Left Joystick: X={self.left_x_val:.2f}, Y={self.left_y_val:.2f}")
        walk_idqp(h=0.25, vx=self.left_y_val, vy=0, vrz=self.left_x_val)

      dx = self.right_target_x - self.right_x_val
      if abs(dx) < 0.01:
        self.right_x_val = self.right_target_x
      else:
        self.right_x_val += dx * smoothing
      dy = self.right_target_y - self.right_y_val
      if abs(dy) < 0.01:
        self.right_y_val = self.right_target_y
      else:
        self.right_y_val += dy * smoothing

      # Update right joystick visual position
      canvas_x = center_x + self.right_x_val * max_radius
      canvas_y = center_y - self.right_y_val * max_radius

      self.right_canvas.coords(
        self.right_handle,
        canvas_x - self.handle_size / 2,
        canvas_y - self.handle_size / 2,
        canvas_x + self.handle_size / 2,
        canvas_y + self.handle_size / 2,
      )

      # Update right labels
      self.right_x_label.config(text=f"X: {self.right_x_val:.2f}")
      self.right_y_label.config(text=f"Y: {self.right_y_val:.2f}")
      if abs(self.right_x_val) > 0.01 or abs(self.right_y_val) > 0.01:
        print(f"Right Joystick: X={self.right_x_val:.2f}, Y={self.right_y_val:.2f}")

      if self.right_x_val != self.right_x_val_prev or self.right_y_val != self.right_y_val_prev:
        walk_idqp(h=0.25, vx=self.right_y_val, vy=self.right_x_val, vrz=0)

      self.left_x_val_prev = self.left_x_val
      self.left_y_val_prev = self.left_y_val
      self.right_x_val_prev = self.right_x_val
      self.right_y_val_prev = self.right_y_val
      self.root.after(20, self.update_smooth_motion)

  root = tk.Tk()
  app = RobotControlGUI(root, args.scale)
  print("Press 'q' to exit the application")
  root.resizable(False, False)  # Disable both horizontal and vertical resizing
  root.mainloop()

  print("Exiting application...")
