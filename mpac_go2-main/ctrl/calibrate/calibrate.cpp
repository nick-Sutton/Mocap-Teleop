#include "robot.h"
#include "estim/state_estim.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <stdio.h>

#include "calibrate.h"
#include "ctrl_mode_class.h"
#include "io/io_handler.h"

extern Robot robot;

namespace calibrate {

#define NUM_SAMPLES 5000

typedef struct {
  int samples;
  double foot_force_sum[4];
  double accel_sum[3];
  joint_pos_pd::JointPosPD jppd;

} Data;

class Calibrate: public PrimitiveBehavior {

Data data;

public:
void init(const StateVec &q,
          const StateVec &qd,
          const ContactState c_s,
          const Args &args) {
  InputVec q_des;
  q_des << 0.7, 1.57, -2.5,
          -0.7, 1.57, -2.5,
           0.7, 1.57, -2.5,
          -0.7, 1.57, -2.5;

  joint_pos_pd::init(data.jppd, q.tail(NUM_U), qd.tail(NUM_U), q_des, VectorXd::Zero(NUM_U), 3);
  data.samples = 0;
  memset(&data.foot_force_sum, 0, 4*sizeof(double));
  memset(&data.accel_sum, 0, 3*sizeof(double));
}

void execute(const StateVec &q,
             const StateVec &qd,
             const ContactState c_s,
             const Args &args, 
             ActuatorCmds &act_cmds) {
  if (joint_pos_pd::execute(data.jppd, act_cmds, q, qd)) {
    if (1000 < data.samples <= NUM_SAMPLES+1000) {
      data.samples++;
      for (int i=0;i<4;++i) {
        data.foot_force_sum[i] += robot.y[F_FRC_FL_EE+i];
      }
      for (int i=0;i<3;++i) {
        data.accel_sum[i] += robot.y[S_IMU_ACC_X+i];
      }
    }
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
  //TODO better handling of "never pick this in the plan"
  q_des = q;
  qd_des = qd;
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

void set_next_ctrl_des(const StateVec &q, const StateVec &qd, const ContactState c_s,
                       CtrlState &ctrl_des) {
  if (data.samples > NUM_SAMPLES) {
    ctrl_des.mode = C_LIE;

    FILE *fp;
    fp = fopen(robot.io->mode() == IO_handler::MODE_SIM ? "sim_calibration.txt" : "calibration.txt", "w");

    for (int i=0;i<4;++i) {
      if (robot.io->mode() == IO_handler::MODE_SIM) {
        fprintf(fp, "%lf, ", 0.0);
      } else {
        fprintf(fp, "%lf, ", data.foot_force_sum[i]/data.samples);
      }
    }
    for (int i=0;i<3;++i) {
      if (robot.io->mode() == IO_handler::MODE_SIM) {
        fprintf(fp, "%lf, ", i == 2 ? 9.81 : 0.0);
      } else {
        fprintf(fp, "%lf, ", data.accel_sum[i]/data.samples);
      }
    }
    fclose(fp);
    state_estim_read_cal_file(robot.estim);
  }
}

};

PrimitiveBehavior* create() {
  return new Calibrate;
}
void destroy(PrimitiveBehavior* prim) {
  delete prim;
}

}
