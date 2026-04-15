#include "robot.h"
#include <eigen3/Eigen/Dense>
#include <iostream>

#include "math_utils.h"
#include "ctrl_core.h"
#include "ctrl/ctrl_utils/traj_gen.h"
#include "ik.h"
#include "fk.h"
#include "stand.h"
#include "stand_args.h"

#include "ctrl_mode_class.h"

extern Robot robot;

namespace stand {

typedef struct {
double t = 0;
double foot_pos_init[4][3];
traj_gen::CubicPoly traj[6]; //x,y,z,rx,ry,rz

} Data;

class Stand: public PrimitiveBehavior {

Data data;

public:
ArgAttributes get_arg_attributes() {
  return (ArgAttributes) {NUM_DISC_ARGS, NUM_CONT_ARGS,
                          disc_min, disc_max,
                          cont_min, cont_max,
                                 0,        0};
}
void init(const StateVec &q_in,
          const StateVec &qd,
          const ContactState c_s,
          const Args &args) {
  data.t = 0;
  double centroid[2] = {};
  double relative_z = 0;
  bool relative_z_valid = contact_feet_relative_z(q_in, c_s, relative_z);
  StateVec q = q_in;
  q[Q_Z] = relative_z;
  for (int i = 0; i < 4; ++i) {
    fk(q, data.foot_pos_init[i], (Frame)(i+F_FL_EE));
    //foot_pos_init[i][2] =0;
    centroid[0] += 0.25*data.foot_pos_init[i][0];
    centroid[1] += 0.25*data.foot_pos_init[i][1];
  }

  double fx = 0.5*(data.foot_pos_init[0][0] + data.foot_pos_init[1][0]);
  double fy = 0.5*(data.foot_pos_init[0][1] + data.foot_pos_init[1][1]);
  double bx = 0.5*(data.foot_pos_init[2][0] + data.foot_pos_init[3][0]);
  double by = 0.5*(data.foot_pos_init[2][1] + data.foot_pos_init[3][1]);

  double rz = atan2(fy-by, fx-bx);

  double q_des[6] = {centroid[0]+0.012731, //TODO pull actual CoM
                     centroid[1],
                     args.cont[ARG_H],
                     args.cont[ARG_RX],
                     args.cont[ARG_RY],
                     args.cont[ARG_RZ] + rz};

  double duration = 1;
  for (int i = 0; i < 3; ++i) {
    data.traj[i] = traj_gen::cubic_poly_gen(q[i+Q_X],qd[i+Q_X],q_des[i],0,duration);
  }
  for (int i = 3; i < 6; ++i) {
    q_des[i] = q[i+Q_X] + min_angle_diff(q_des[i],q[i+Q_X]);
    data.traj[i] = traj_gen::cubic_poly_gen(q[i+Q_X],qd[i+Q_X],q_des[i],0,duration);
  }
}

void execute(const StateVec &q,
             const StateVec &qd,
             const ContactState c_s,
             const Args &args, 
             ActuatorCmds &act_cmds) {
  StateVec q_curr = VectorXd::Zero((int)NUM_Q);
  StateVec qd_curr = VectorXd::Zero((int)NUM_Q);
  InputVec q_cmd = VectorXd::Zero((int)NUM_U);
  InputVec qd_cmd = VectorXd::Zero((int)NUM_U);
  bool finished = true;

  data.t += CTRL_LOOP_DURATION;
  for (int i = 0; i < 6; ++i) {
    finished &= traj_gen::cubic_poly_eval(data.traj[i], data.t, q_curr[i+Q_X], qd_curr[i+Q_X]);
  }

  for (int i=0; i<4;++i) {
    ik((Frame)(i+F_FL_EE), q_curr, data.foot_pos_init[i], &q_cmd[3*i]);
  }

  for (int i = 0; i < NUM_U; ++i) {
    act_cmds.mode[i] = CMD_MODE_TAU_VEL_POS;
    act_cmds.u[i] = 0;
    act_cmds.q[i] = q_cmd[i];
    act_cmds.qd[i] = 0;
    act_cmds.kp[i] = 150;
    act_cmds.kd[i] = 5;
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
  //TODO fill in
  q_des = q;
  qd_des = qd;
  memcpy(&c_s_des, c_s, sizeof(ContactState));
}

MatrixXd setpoint_jac(const StateVec &q,
                           const StateVec &qd,
                           const ContactState c_s,
                           const Args &args,
                           const double delta_t) const {
  MatrixXd jac = MatrixXd::Zero((int)NUM_Q+NUM_Q, NUM_CONT_ARGS+1);
  jac(Q_Z, 0) = 1;
  jac(Q_RX, 1) = 1;
  jac(Q_RY, 2) = 1;
  jac(Q_RZ, 3) = 1;

  return jac;
}

bool in_sroa_underestimate(const StateVec &q,
                           const StateVec &qd,
                           const ContactState c_s,
                           const Args &args,
                           const double delta_t) const {
  if (!in_safe_set(q,qd,c_s,args,delta_t)) {
    return false;
  }
  if ((fabs(q[Q_Z]-args.cont[ARG_H]) < 0.2) &&
      (q[Q_Z] > 0) &&
      (fabs(q[Q_RX]-args.cont[ARG_RX]) < 0.5) &&
      (fabs(q[Q_RY]-args.cont[ARG_RY]) < 0.5) &&
      //(fabs(q[Q_RZ]-args.cont[ARG_RZ]) < 0.5) &&
      (qd.squaredNorm() < 100)) {
    return true;
  }
  return false;
}
bool in_sroa_overestimate(const StateVec &q,
                          const StateVec &qd,
                          const ContactState c_s,
                          const Args &args,
                          const double delta_t) const {
  if (!in_safe_set(q,qd,c_s,args,delta_t)) {
    return false;
  }
  if ((fabs(q[Q_Z]-args.cont[ARG_H]) < 0.2) &&
      (q[Q_Z] > 0) &&
      (fabs(q[Q_RX]-args.cont[ARG_RX]) < 0.5) &&
      (fabs(q[Q_RY]-args.cont[ARG_RY]) < 0.5) &&
      //(fabs(q[Q_RZ]-args.cont[ARG_RZ]) < 0.5) &&
      (qd.squaredNorm() < 100)) {
    return true;
  }
  return false;
}
bool in_safe_set(const StateVec &q,
                 const StateVec &qd,
                 const ContactState c_s,
                 const Args &args,
                 const double delta_t) const {
  if (!(in_vel_limits(qd) && in_pos_limits(q))) {
    return false;
  }
  
  // all feet should be in contact
  if (c_s[0] && c_s[1] && c_s[2] && c_s[3]) {
    return true;
  } else {
    return false;
  }
}

};

PrimitiveBehavior* create() {
  return new Stand;
}
void destroy(PrimitiveBehavior* prim) {
  delete prim;
}

}
