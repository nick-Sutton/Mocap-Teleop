#include "robot.h"

#include "sim_core.h"
#include "io/sim_io.h"
#include "fk.h"
#include "full_model_fd.h"
#include "math_utils.h"

extern Robot robot;

namespace sim_io {

static const int buff_size = 3;

void init(SimIo &sim_io, char* args) {
  sim_io.q_buffer = std::vector<StateVec>(buff_size);
  sim_io.qd_buffer = std::vector<StateVec>(buff_size);
  sim_io.init = false;

  sim_io.use_ground_truth = false;
  if (args) {
    sim_io.use_ground_truth = strcmp(args, "ground_truth") == 0;
  }
}

void read(SimIo &sim_io, OutputVec &y,
          StateVec &q_sim, StateVec &qd_sim,
          StateVec &qdd_sim, ContactVec &c_sim,
          InputVec &u_sim) {
  // probably should make a seperate init function
  if (!sim_io.init) {
    for (int i=0; i<buff_size; ++i) {
      fill(sim_io.q_buffer.begin(), sim_io.q_buffer.end(), q_sim);
      fill(sim_io.qd_buffer.begin(), sim_io.qd_buffer.end(), qd_sim);
    }
    sim_io.init = true;
  }

  //TODO better ways to do a ring buffer
  sim_io.q_buffer.pop_back();
  sim_io.q_buffer.insert(sim_io.q_buffer.begin(),q_sim);
  sim_io.qd_buffer.pop_back();
  sim_io.qd_buffer.insert(sim_io.qd_buffer.begin(),qd_sim);
  // IMU
  for (int i = 0; i < 3; ++i) {
    y(i + S_IMU_ACC_X)   = qdd_sim(i);
    y(i + S_IMU_GYRO_RX) = sim_io.qd_buffer.back()(i+Q_RX);
  }
  y(S_IMU_ACC_Z) += 9.81; //adding gravity to sim qdd to match what accelerometer reads

  Matrix3d m;
  m = AngleAxisd(q_sim(Q_RX), Vector3d::UnitX())
    * AngleAxisd(q_sim(Q_RY), Vector3d::UnitY())
    * AngleAxisd(q_sim(Q_RZ), Vector3d::UnitZ());
  Vector3d euler = euler_wrap(m.eulerAngles(0, 1, 2));
  y(S_IMU_FUSED_RX) = euler[0];
  y(S_IMU_FUSED_RY) = euler[1];
  y(S_IMU_FUSED_RZ) = euler[2];

  // joints
  for (int i = 0; i < NUM_Q-Q_FL1; ++i) {
    y(i + S_Q_FL1) = sim_io.q_buffer.back()(i + Q_FL1);
    y(i + S_QD_FL1) = sim_io.qd_buffer.back()(i + Q_FL1);
    y(i + S_U_FL1) = u_sim[i];
  }

  // foot forces (actual robot force sensor is single axis, as magnitude)
  for (int contact = C_FL_EE; contact < C_BR_EE+1; ++contact) {
    VectorXd contact_force = compute_contact_force(q_sim, qd_sim, c_sim, contact, false);
    y(F_FRC_FL_EE + contact) = contact_force.norm();
  } 

  if (sim_io.use_ground_truth) {
    y(S_GROUND_TRUTH_X) = q_sim[Q_X];
    y(S_GROUND_TRUTH_Y) = q_sim[Q_Y];
    y(S_GROUND_TRUTH_Z) = q_sim[Q_Z];
    y(S_GROUND_TRUTH_RX) = q_sim[Q_RX];
    y(S_GROUND_TRUTH_RY) = q_sim[Q_RY];
    y(S_GROUND_TRUTH_RZ) = q_sim[Q_RZ];
  } else {
    y(S_GROUND_TRUTH_X) = std::numeric_limits<double>::quiet_NaN();
    y(S_GROUND_TRUTH_Y) = std::numeric_limits<double>::quiet_NaN();
    y(S_GROUND_TRUTH_Z) = std::numeric_limits<double>::quiet_NaN();
    y(S_GROUND_TRUTH_RX) = std::numeric_limits<double>::quiet_NaN();
    y(S_GROUND_TRUTH_RY) = std::numeric_limits<double>::quiet_NaN();
    y(S_GROUND_TRUTH_RZ) = std::numeric_limits<double>::quiet_NaN();
  }

}

void write(const ActuatorCmds &act_cmds,
           StateVec &q_sim, StateVec &qd_sim, StateVec &qdd_sim,
           ContactVec &c_sim, InputVec &u_sim) {
  sim(act_cmds,q_sim,qd_sim, qdd_sim, c_sim, u_sim);

}

void finish(SimIo &sim_io) {

}

}
