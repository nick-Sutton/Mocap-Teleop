#ifndef JOINT_POS_PD_H
#define JOINT_POS_PD_H

#include "robot_types.h"
#include <eigen3/Eigen/Dense>
#include "ctrl/ctrl_utils/traj_gen.h"

namespace joint_pos_pd {

typedef struct { 
  double t = 0;
  traj_gen::CubicPoly traj[NUM_U];
} JointPosPD;

void init(JointPosPD &joint_pos_pd,
          const InputVec &q,
          const InputVec &qd,
          const InputVec &q_des,
          const InputVec &qd_des,
          double duration);
bool execute(JointPosPD &joint_pos_pd,
             ActuatorCmds &act_cmds,
             const StateVec &q,
             const StateVec &qd);
}

#endif //JOINT_POS_PD_H
