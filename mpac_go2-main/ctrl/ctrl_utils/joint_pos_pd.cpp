#include "robot.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <stdio.h>

#include "ctrl_core.h"
#include "ctrl/ctrl_utils/traj_gen.h"
#include "ctrl/ctrl_utils/joint_pos_pd.h"

namespace joint_pos_pd {

void init(JointPosPD &joint_pos_pd,
          const InputVec &q,
          const InputVec &qd,
          const InputVec &q_des,
          const InputVec &qd_des,
          double duration) {
  joint_pos_pd.t = 0;
  double norm_dist = (q - q_des).norm();
  if (duration <= 0) {
    duration = 0.2*norm_dist;
  }
  for (int i = 0; i < NUM_U; ++i) {
    joint_pos_pd.traj[i] = traj_gen::cubic_poly_gen(q[i],qd[i],q_des[i],qd_des[i],duration);
  }
}

bool execute(JointPosPD &joint_pos_pd,
             ActuatorCmds &act_cmds,
             const StateVec &q,
             const StateVec &qd) {
  joint_pos_pd.t += CTRL_LOOP_DURATION;

  double q_cmd, qd_cmd;

  bool finished = true;

  for (int i = 0; i < NUM_U; ++i) {
    finished &= traj_gen::cubic_poly_eval(joint_pos_pd.traj[i], joint_pos_pd.t, q_cmd, qd_cmd);

    act_cmds.mode[i] = CMD_MODE_TAU_VEL_POS;
    act_cmds.u[i] = 0;
    act_cmds.q[i] = q_cmd;
    act_cmds.qd[i] = qd_cmd;
    act_cmds.kp[i] = 50;
    act_cmds.kd[i] = 5;
  }
  return finished;
}

}
