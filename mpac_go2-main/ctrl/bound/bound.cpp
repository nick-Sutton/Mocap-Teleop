#include "robot.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "math.h"

#include "ctrl_core.h"
#include "bound.h"
#include "bound_args.h"
#include "full_model_fd.h"
#include "fk.h"
#include "ik.h"
#include "ctrl/ctrl_utils/traj_gen.h"
#include "math_utils.h"
#include "ctrl_mode_class.h"

extern Robot robot;

namespace bound {

typedef struct {
  double t;
  double foot_pos_init[4][3];
  double i_err;
  bool swing_legs[4];
  double qd_filt;
  double q_rz_init;
  
} Data;

double step_time = 0.1;
double dwell_time = 0.05;

class Bound: public PrimitiveBehavior {

Data data;

public:
ArgAttributes get_arg_attributes() {
  return (ArgAttributes) {NUM_DISC_ARGS, NUM_CONT_ARGS,
                          disc_min, disc_max,
                          cont_min, cont_max,
                                 0,        0};
}
void init(const StateVec &q,
          const StateVec &qd,
          const ContactState c_s,
          const Args &args) {
  data.t = 0;
  InputVec q_des;
  InputVec qd_des;

  data.q_rz_init = q[Q_RZ];
  data.i_err = 0;
  data.qd_filt = 0;

  StateVec q_rz_zero;
  StateVec qd_rz_zero;
  rz_zero_frame(data.q_rz_init, q, qd, q_rz_zero,  qd_rz_zero);
  for (int i = 0; i < 4; ++i) {
    fk(q_rz_zero, data.foot_pos_init[i], (Frame)(i+F_FL_EE));
    data.swing_legs[i]=false;
  }
}

void execute(const StateVec &q,
             const StateVec &qd,
             const ContactState c_s,
             const Args &args, 
             ActuatorCmds &act_cmds) {
  data.t += CTRL_LOOP_DURATION;
  if (data.t > 2*step_time+2*dwell_time) {
    data.t = 0;
  }
  gait_state_machine(data.t, q, c_s, data.swing_legs, data.foot_pos_init);

  InputVec q_des;
  InputVec qd_des;

  compute_joint_pos(data.t, q, qd, c_s, q_des, qd_des, args);

  for (int i = 0; i < NUM_U; ++i) {
    act_cmds.mode[i] = CMD_MODE_TAU_VEL_POS;
    act_cmds.u[i] = 0;
    act_cmds.q[i] = q_des[i];
    act_cmds.qd[i] = 0;//qd_cmd;
    act_cmds.kp[i] = 500;
    act_cmds.kd[i] = 15;
  }
}

/* Note for softstop, we always want to be able to transition in case of emergency */
void setpoint(const StateVec &q,
              const StateVec &qd,
              const ContactState c_s,
              const Args &args,
              const double delta_t,
              StateVec &q_des,
              StateVec &qd_des,
              ContactState &c_s_des) const {
  //TODO better handling of "don't care" for q_des
  q_des = q;
  qd_des = qd;//VectorXd::Zero((int)NUM_Q);
  memcpy(&c_s_des, c_s, sizeof(ContactState));
}

MatrixXd setpoint_jac(const StateVec &q,
                           const StateVec &qd,
                           const ContactState c_s,
                           const Args &args,
                           const double delta_t) const {
  return MatrixXd::Zero((int)NUM_Q+NUM_Q, NUM_CONT_ARGS+1);
}

bool in_sroa_underestimate(const StateVec &q_in,
                           const StateVec &qd_in,
                           const ContactState c_s,
                           const Args &args,
                           const double delta_t) const {
    return false;
  if (!in_safe_set(q_in,qd_in,c_s,args,delta_t)) {
    return false;
  }

  StateVec q;
  StateVec qd;
  rz_zero_frame(q_in[Q_RZ], q_in, qd_in, q,  qd);

  if (fabs(q[Q_Z] - 0.25) < 0.15 ){// &&
     //(qd[Q_X] > 0.2) &&
     //(fabs(qd[Q_Y]) < 0.2)) {
    return true;
  }
}
bool in_sroa_overestimate(const StateVec &q_in,
                          const StateVec &qd_in,
                          const ContactState c_s,
                          const Args &args,
                          const double delta_t) const {
    return false;
  if (!in_safe_set(q_in,qd_in,c_s,args,delta_t)) {
    return false;
  }

  StateVec q;
  StateVec qd;
  rz_zero_frame(q_in[Q_RZ], q_in, qd_in, q,  qd);

  if (fabs(q[Q_Z] - 0.25) < 0.15 ){// &&
     //(qd[Q_X] > 0.2) &&
     //(fabs(qd[Q_Y]) < 0.2)) {
    return true;
  }
  return false;
}
bool in_safe_set(const StateVec &q,
                 const StateVec &qd,
                 const ContactState c_s,
                 const Args &args,
                 const double delta_t) const {
  //return in_vel_limits(qd) && in_pos_limits(q);
  return true;
}

private: 

void rz_zero_frame(const double q_rz_init, const StateVec &q, const StateVec &qd,
                               StateVec &q_rz_zero,  StateVec &qd_rz_zero) const {
  Matrix3d R;
  Matrix3d R_curr;
  R = AngleAxisd(0, Vector3d::UnitX()) *
      AngleAxisd(0, Vector3d::UnitY()) *
      AngleAxisd(-q_rz_init, Vector3d::UnitZ()); 
  R_curr = AngleAxisd(q(Q_RX), Vector3d::UnitX()) *
           AngleAxisd(q(Q_RY), Vector3d::UnitY()) *
           AngleAxisd(q(Q_RZ), Vector3d::UnitZ()); 
  q_rz_zero = q;
  qd_rz_zero = qd;
  Vector3d euler = euler_wrap((R*R_curr).eulerAngles(0, 1, 2));
  q_rz_zero[Q_RX] = euler[0];
  q_rz_zero[Q_RY] = euler[1];
  q_rz_zero[Q_RZ] = euler[2];
  q_rz_zero.head(3) = R*q_rz_zero.head(3);
  qd_rz_zero.head(3) = R*qd_rz_zero.head(3);
}

void compute_joint_pos(double t,
                       const StateVec &q_in, const StateVec &qd_in, const ContactState c_s,
                       InputVec &q_des, InputVec &qd_des, const Args &args) {
  StateVec q;
  StateVec qd;
  rz_zero_frame(data.q_rz_init, q_in, qd_in, q,  qd);
  StateVec q_ground = VectorXd::Zero(NUM_Q);
  StateVec q_early = q;
  q_ground[Q_X] = q[Q_X] + 25*args.cont[ARG_VX]*CTRL_LOOP_DURATION;
  q_early[Q_X] = q_ground[Q_X];
  q_ground[Q_Y] = q[Q_Y];
  q_ground[Q_Z] = 0.25;
  q_ground[Q_RZ] = 0;
  q_ground[Q_RY] = 0.5*q[Q_RY];
  double foot_pos_des[4][3] = {};
  double joint_des[NUM_U] = {};

  double step_height = 0.125;
  double x_goal;
  double y_goal;

  double Ki = 0.000005;
  double antiwindup = 0.05;
  data.i_err += Ki*(qd[Q_X] - args.cont[ARG_VX]);
  data.i_err = fmin(antiwindup, fmax(-antiwindup, data.i_err));
  data.qd_filt += 0.5*(qd[Q_X] - data.qd_filt);
  double x_delta = 0.5*(step_time+dwell_time)*data.qd_filt;// + 0.01*(data.qd_filt - args.cont[ARG_VX]);//-data.i_err;

  double gamma = 1/(step_time+dwell_time)*(fmod(data.t, step_time+dwell_time));

  for (int i = 0; i < 4; ++i) {
    //make x negative for back feet
    //make y negative for right feet
    x_goal = (i == 0 || i == 1) ? 0.15 : -0.17;
    y_goal = (i == 0 || i == 2) ? 0.12 : -0.12;
    //x_goal += data.i_err;

    if (data.swing_legs[i]) {
      foot_pos_des[i][2] = -((gamma-0.5)*(gamma-0.5)-0.25)*4*step_height;
      if (c_s[i]) {
        foot_pos_des[i][0] = data.foot_pos_init[i][0];
        foot_pos_des[i][1] = data.foot_pos_init[i][1];
        ik(robot.contacts[i].frame, q_early, foot_pos_des[i], &joint_des[3*i]);
      } else {
        foot_pos_des[i][0] = (1-gamma)*data.foot_pos_init[i][0] + gamma*(q[Q_X]+x_goal+x_delta);
        foot_pos_des[i][1] = (1-gamma)*data.foot_pos_init[i][1] + gamma*(q[Q_Y]+y_goal);
        ik(robot.contacts[i].frame, q, foot_pos_des[i], &joint_des[3*i]);
      }
    } else {
      foot_pos_des[i][2] = 0;
      foot_pos_des[i][0] = data.foot_pos_init[i][0];
      foot_pos_des[i][1] = data.foot_pos_init[i][1];
      ik(robot.contacts[i].frame, q_ground, foot_pos_des[i], &joint_des[3*i]);
    }
  }

  //for (int foot = 0; foot < 4; ++foot) {
  //  ik(robot.contacts[foot].frame, q_ground, foot_pos_des[foot], &joint_des[3*foot]);
  //}

  for (int i=0; i<12; ++i) {
    q_des[i] = joint_des[i];
  }
}

void gait_state_machine(double t, const StateVec &q_in, const ContactState c_s, bool swing_legs[4], double foot_pos_init[4][3]) {
  StateVec q;
  StateVec qd;
  StateVec qd_in = VectorXd::Zero((int)NUM_Q);
  rz_zero_frame(data.q_rz_init, q_in, qd_in, q,  qd);

  double duration = 2*(dwell_time+step_time);
  double t_phase = fmod(t,duration);
  if (t_phase < dwell_time) {
    for (int i = 0; i < 4; ++i) {
      fk(q, foot_pos_init[i], (Frame)(i+F_FL_EE));
      swing_legs[i]=false;
    }
  } else if (dwell_time <= t_phase && t_phase < step_time + dwell_time) {
    swing_legs[0] = true;
    swing_legs[1] = true;
  } else if (step_time + dwell_time <= t_phase && t_phase < step_time + 2*dwell_time) {
    for (int i = 0; i < 4; ++i) {
      fk(q, foot_pos_init[i], (Frame)(i+F_FL_EE));
      swing_legs[i]=false;
    }
  } else if (step_time + 2*dwell_time <= t_phase && t_phase < 2*step_time + 2*dwell_time) {
    swing_legs[2] = true;
    swing_legs[3] = true;
  }
  for (int i = 0; i < 4; ++i) {
    if (!c_s[i]) {
      fk(q, foot_pos_init[i], (Frame)(i+F_FL_EE));
    }
  }
}

};

PrimitiveBehavior* create() {
  return new Bound;
}
void destroy(PrimitiveBehavior* prim) {
  delete prim;
}

}
