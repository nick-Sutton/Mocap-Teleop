#include "robot.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <stdio.h>

#include "ctrl_core.h"
#include "ctrl/ctrl_utils/traj_gen.h"
#include "ctrl/ctrl_utils/joint_pos_pd.h"
#include "ctrl_mode_class.h"
#include "lie.h"
#include "lie_tlm.h"
#include "fk.h"

extern Robot robot;

namespace lie {

class Lie: public PrimitiveBehavior {
  joint_pos_pd::JointPosPD jppd;
  LieTelemetry lie_tlm;

public:
Lie() {
  tlm_ptr = (char*) &lie_tlm;
  tlm_size = sizeof lie_tlm;
}
void init(const StateVec &q,
          const StateVec &qd,
          const ContactState c_s,
          const Args &args) {
  InputVec q_des;
  q_des << 0, 1.57, -2.5,
           0, 1.57, -2.5,
           0, 1.57, -2.5,
           0, 1.57, -2.5;
  joint_pos_pd::init(jppd, q.tail(NUM_U), qd.tail(NUM_U), q_des, VectorXd::Zero(NUM_U), -1);
}

void execute(const StateVec &q,
             const StateVec &qd,
             const ContactState c_s,
             const Args &args, 
             ActuatorCmds &act_cmds) {
  joint_pos_pd::execute(jppd, act_cmds, q, qd);
  for (int i = 0; i < NUM_U; ++i) {
    lie_tlm.q_err[i] = q[i+Q_FL1] - act_cmds.q[i];
  }
}

bool has_tlm() {return true;};

void create_attributes_type(hid_t &attr_type) {
  hid_t str32 = H5Tcopy (H5T_C_S1);
  size_t size = 32 * sizeof(char);
  H5Tset_size(str32, size);

  hid_t single_attr_type = H5Tcreate (H5T_COMPOUND, sizeof (TelemetryAttribute));
  H5Tinsert(single_attr_type, "name", HOFFSET (TelemetryAttribute, name), str32);
  H5Tinsert(single_attr_type, "units", HOFFSET (TelemetryAttribute, units), str32);
  H5Tinsert(single_attr_type, "crit_lo", HOFFSET (TelemetryAttribute, crit_lo), H5T_NATIVE_DOUBLE);
  H5Tinsert(single_attr_type, "warn_lo", HOFFSET (TelemetryAttribute, warn_lo), H5T_NATIVE_DOUBLE);
  H5Tinsert(single_attr_type, "warn_hi", HOFFSET (TelemetryAttribute, warn_hi), H5T_NATIVE_DOUBLE);
  H5Tinsert(single_attr_type, "crit_hi", HOFFSET (TelemetryAttribute, crit_hi), H5T_NATIVE_DOUBLE);

  hsize_t dims_input[1] = {NUM_U};
  hid_t input_vec = H5Tarray_create(single_attr_type, 1, dims_input);

  attr_type = H5Tcreate (H5T_COMPOUND, sizeof (TelemetryAttributes));
  H5Tinsert(attr_type, "q_err", HOFFSET (TelemetryAttributes, q_err), input_vec);
};


void fill_attributes(std::vector<TelemetryAttribute> &attributes) {
  TelemetryAttributes attr;

  Robot robot;
  robot_params_init(robot);

  memset(&attr, 0, sizeof(TelemetryAttributes));

  for (int i = 0; i < NUM_U; ++i) {
    sprintf(attr.q_err[i].name, "%s_err", robot.joints[Q_FL1+i].name);
    strcpy(attr.q_err[i].units, robot.joints[i].type == REVOLUTE ? "rad":"m");
    attr.q_err[i].crit_lo = -0.1;
    attr.q_err[i].warn_lo = -0.05;
    attr.q_err[i].crit_hi =  0.05;
    attr.q_err[i].warn_hi =  0.1;
  }

  //TODO probably a cleaner way...
  TelemetryAttribute *attr_array = (TelemetryAttribute*) &attr;
  for (int i=0;i < sizeof(TelemetryAttributes)/sizeof(TelemetryAttribute); ++i) {
    attributes.push_back(attr_array[i]);
  }
}

void create_timeseries_type(hid_t &timeseries_type) {
  hsize_t dims_input[1] = {NUM_U};
  hid_t input_vec = H5Tarray_create(H5T_NATIVE_DOUBLE, 1, dims_input);

  timeseries_type = H5Tcreate (H5T_COMPOUND, sizeof (LieTelemetry));
  H5Tinsert(timeseries_type, "q_err", HOFFSET (LieTelemetry, q_err), input_vec);
};

/* Note for softstop, we always want to be able to transition in case of emergency */
void setpoint(const StateVec &q,
              const StateVec &qd,
              const ContactState c_s,
              const Args &args,
              const double delta_t,
              StateVec &q_des,
              StateVec &qd_des,
              ContactState &c_s_des) const {
  q_des << q(Q_X), q(Q_Y), 0.119727,
           0,    0, q(Q_RZ),
           0, 1.57, -2.5,
           0, 1.57, -2.5,
           0, 1.57, -2.5,
           0, 1.57, -2.5;
  qd_des = VectorXd::Zero(NUM_Q);
  memset(&c_s_des, true, sizeof(ContactState));
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
  //TODO lie usually is the first move in sim, where joint position limits aren't enforced,
  // so don't include joint position limits as safety for now
  if (!in_vel_limits(qd)) {
    return false;
  }

  // at least one foot should be in contact
  if (c_s[0] || c_s[1] || c_s[2] || c_s[3]) {
    return true;
  } else {
    return false;
  }
}

};

PrimitiveBehavior* create() {
  return new Lie;
}
void destroy(PrimitiveBehavior* prim) {
  delete prim;
}

}
