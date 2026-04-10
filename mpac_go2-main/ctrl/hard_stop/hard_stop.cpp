#include "robot.h"
#include <eigen3/Eigen/Dense>
#include <iostream>

#include "hard_stop.h"
#include "ctrl_mode_class.h"

class HardStop: public PrimitiveBehavior {

public:
void init(const StateVec &q,
          const StateVec &qd,
          const ContactState c_s,
          const Args &args){}

void execute(const StateVec &q,
             const StateVec &qd,
             const ContactState c_s,
             const Args &args, 
             ActuatorCmds &act_cmds) {
  for (int i = 0; i < NUM_U; ++i) {
    act_cmds.mode[i] = CMD_MODE_TAU;
    act_cmds.u[i] = 0;
  }
}

/* Note for hardstop, we always want to be able to transition in case of emergency */
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
  return MatrixXd::Zero((int)NUM_Q+NUM_Q, 1);
}

bool in_sroa_underestimate(const StateVec &q,
                           const StateVec &qd,
                           const ContactState c_s,
                           const Args &args,
                           const double delta_t) const {
  return true;
}
bool in_sroa_overestimate(const StateVec &q,
                          const StateVec &qd,
                          const ContactState c_s,
                          const Args &args,
                          const double delta_t) const {
  return true;
}
bool in_safe_set(const StateVec &q,
                 const StateVec &qd,
                 const ContactState c_s,
                 const Args &args,
                 const double delta_t) const {
  return true;
}

};

namespace hard_stop {

PrimitiveBehavior* create() {
  return new HardStop;
}
void destroy(PrimitiveBehavior* prim) {
  delete prim;
}

}
