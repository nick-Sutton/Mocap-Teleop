#ifndef CTRL_MODES_H
#define CTRL_MODES_H

#define FIXED_MODES(ctrl_mode) \
  ctrl_mode(HARD_STOP, hard_stop) \
  ctrl_mode(SOFT_STOP, soft_stop) \
  ctrl_mode(LIE, lie) \
  ctrl_mode(SIT, sit) \
  ctrl_mode(STAND, stand) \
  ctrl_mode(STAND_IDQP, stand_idqp) \
  ctrl_mode(TRAJ_TRACK, traj_track) \

#define PERIODIC_MODES(ctrl_mode) \
  ctrl_mode(WALK_TRAJ, walk_traj) \
  ctrl_mode(WALK_IDQP, walk_idqp) \
  ctrl_mode(WALK_QUASI_IDQP, walk_quasi_idqp) \
  ctrl_mode(WALK_QUASI_PLANNED, walk_quasi_planned) \
  ctrl_mode(BOUND, bound) \

#define TRANSIENT_MODES(ctrl_mode) \
  ctrl_mode(JUMP, jump) \
  ctrl_mode(LAND, land) \
  ctrl_mode(CALIBRATE, calibrate) \

#define CTRL_MODES(ctrl_mode) \
  FIXED_MODES(ctrl_mode) \
  PERIODIC_MODES(ctrl_mode) \
  TRANSIENT_MODES(ctrl_mode) \

#endif //CTRL_MODES_H

