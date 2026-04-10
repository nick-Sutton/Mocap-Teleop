#include "robot.h"
#include <eigen3/Eigen/Dense>
#include <iostream>

#include "ctrl/ctrl_utils/joint_pos_pd.h"
#include "sit.h"
#include "ctrl_mode_class.h"

namespace sit {

class Sit: public PrimitiveBehavior {

joint_pos_pd::JointPosPD jppd;

public:
void init(const StateVec &q,
          const StateVec &qd,
          const ContactState c_s,
          const Args &args) {
  InputVec q_des;
  q_des << 0, 1.095, -1.57,
           0, 1.095, -1.57,
           0, 2, -2.5,
           0, 2, -2.5;
  joint_pos_pd::init(jppd, q.tail(NUM_U), qd.tail(NUM_U), q_des, VectorXd::Zero(NUM_U), -1);
}

void execute(const StateVec &q,
             const StateVec &qd,
             const ContactState c_s,
             const Args &args, 
             ActuatorCmds &act_cmds) {
  joint_pos_pd::execute(jppd, act_cmds, q, qd);
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
  //TODO compute proper body height/angle
  q_des << q.head(6),
           0, 1.095, -1.57,
           0, 1.095, -1.57,
           0, 2, -2.5,
           0, 2, -2.5;
  qd_des = VectorXd::Zero(NUM_Q);
}

MatrixXd setpoint_jac(const StateVec &q,
                           const StateVec &qd,
                           const ContactState c_s,
                           const Args &args,
                           const double delta_t) const {
  return MatrixXd::Zero((int)NUM_Q+NUM_Q, 1);
}

bool in_sroa_underestimate(const StateVec &q,
                           const StateVec &qd,
                           const ContactState c_s,
                           const Args &args,
                           const double delta_t) const {
  return false;
  if (!in_safe_set(q,qd,c_s,args,delta_t)) {
    return false;
  }
  return true;
}
bool in_sroa_overestimate(const StateVec &q,
                          const StateVec &qd,
                          const ContactState c_s,
                          const Args &args,
                          const double delta_t) const {
  return false;
  if (!in_safe_set(q,qd,c_s,args,delta_t)) {
    return false;
  }
  return true;
}
bool in_safe_set(const StateVec &q,
                 const StateVec &qd,
                 const ContactState c_s,
                 const Args &args,
                 const double delta_t) const {
  return in_vel_limits(qd) && in_pos_limits(q);
}

};

PrimitiveBehavior* create() {
  return new Sit;
}
void destroy(PrimitiveBehavior* prim) {
  delete prim;
}

}
