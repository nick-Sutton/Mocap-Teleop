#include "stdio.h"
#include "robot.h"
#include "math_utils.h"
#include "ctrl_core.h"
#include "estim/state_estim.h"
#include "fk.h"
#include "io/io_handler.h"

extern Robot robot;

void state_estim_init(StateEstimator &estim) {
  estim.qd_prev = VectorXd::Zero((int)NUM_Q);
  memset(&estim.foot_force_filt, 0, 4*sizeof(double));
  memset(&estim.foot_force_bias, 0, 4*sizeof(double));
  memset(&estim.accel_bias, 0, 3*sizeof(double));
  memset(&estim.foot_contact_pos, 0, 12*sizeof(double));
  memset(&estim.contact, 0, NUM_C*sizeof(bool));
  estim.accel_bias[2] = 9.81; //initialize with gravity
  state_estim_read_cal_file(estim);
  estim.heading_bias = 0;
}

void state_estim_read_cal_file(StateEstimator &estim) {
  FILE *fp;

  if (fp = fopen(robot.io->mode() == IO_handler::MODE_HARDWARE ? "calibration.txt" : "sim_calibration.txt", "r")) {
    for (int i=0;i<4;++i) {
      fscanf(fp, "%lf,", &estim.foot_force_bias[i]);
      printf("foot_force_bias %d: %f\n", i, estim.foot_force_bias[i]);
    }
    for (int i=0;i<3;++i) {
      fscanf(fp, "%lf,", &estim.accel_bias[i]);
      printf("accel_bias %d: %f\n", i, estim.accel_bias[i]);
    }
    fclose(fp);
  }
}

void state_estim(StateEstimator &estim, const OutputVec &y, StateVec &q, StateVec &qd, ContactState c_s) {
  // joints
  for (int i = 0; i < NUM_Q-Q_FL1; ++i) {
    q(i+Q_FL1) = y(i + S_Q_FL1);
    qd(i+Q_FL1) = y(i + S_QD_FL1);
  }

  // Body Rotation
  for (int i = 0; i < 3; ++i) {
    q(i+Q_RX) = y(i + S_IMU_FUSED_RX);
    qd(i+Q_RX) = y(i + S_IMU_GYRO_RX);
  }
  Eigen::Matrix3d R_offset, R_meas, R;
  R_offset = Eigen::AngleAxisd(estim.heading_bias, Eigen::Vector3d::UnitZ()); 
  R_meas  = Eigen::AngleAxisd(q[Q_RX], Eigen::Vector3d::UnitX()) *
            Eigen::AngleAxisd(q[Q_RY], Eigen::Vector3d::UnitY()) *
            Eigen::AngleAxisd(q[Q_RZ], Eigen::Vector3d::UnitZ()); 
  R = R_offset*R_meas;
  Vector3d euler = euler_wrap(R.eulerAngles(0, 1, 2));
  q.segment<3>(Q_RX) = euler;
  qd.segment<3>(Q_RX) = R_offset*qd.segment<3>(Q_RX);

  // filter the foot force to debounce contact detection
  for (int contact = C_FL_EE; contact < C_BR_EE+1; ++contact) {
    estim.foot_force_filt[contact-C_FL_EE] += 0.5*(y(F_FRC_FL_EE + contact) -
                                                   estim.foot_force_filt[contact-C_FL_EE]);
  }
  for (int contact = C_FL_EE; contact < C_BR_EE+1; ++contact) {
    if (estim.foot_force_filt[contact-C_FL_EE] - estim.foot_force_bias[contact-C_FL_EE] > 5) {
      c_s[contact] = true;
    } else {
      c_s[contact] = false;
    }
  }

  // body velocity
  VectorXd foot_vel = VectorXd::Zero(3);
  int contact_feet = 0;
  double z_sum = 0;
  StateVec qd_vel = qd;
  qd_vel(Q_X) = 0;
  qd_vel(Q_Y) = 0;
  qd_vel(Q_Z) = 0;
  /* we are assuming contact feet do not move. While reasonable, it's definitely not true.
     The soft contact model in sim can cause issues with fast collisions (i.e. when falling)
     where the contact feet are noticably moving. Additionally, the squishy feet of the A1 move
     when in contact on the actual hardware. Improvements can be made, but seems good enough for now   */
  for (int contact = C_FL_EE; contact < C_BR_EE+1; ++contact) {
    if (c_s[contact]) {
      MatrixXd jac = fk_jac(q, robot.contacts[contact].frame);
      foot_vel += jac*qd_vel;
      contact_feet++;
      fk(q, estim.foot_contact_pos[contact], robot.contacts[contact].frame);
      if (estim.contact[contact] == false) {
        fk(q, estim.foot_contact_pos[contact], robot.contacts[contact].frame);
        estim.contact[contact] = true;
      }
      z_sum += estim.foot_contact_pos[contact][2];
    } else {
      estim.contact[contact] = false;
    }
  } 
  // TODO: instead of conditional, kalman filter?
  if (contact_feet > 0) {
    for (int i = 0; i < 3; ++i) {
      qd(Q_X+i) += 0.25*(-foot_vel(i)/contact_feet - qd(Q_X+i));
    }
  } else {
    qd(Q_X) = estim.qd_prev(Q_X) + (y(S_IMU_ACC_X)-estim.accel_bias[0])*CTRL_LOOP_DURATION;
    qd(Q_Y) = estim.qd_prev(Q_Y) + (y(S_IMU_ACC_Y)-estim.accel_bias[1])*CTRL_LOOP_DURATION;
    qd(Q_Z) = estim.qd_prev(Q_Z) + (y(S_IMU_ACC_Z)-estim.accel_bias[2])*CTRL_LOOP_DURATION;
  }
  
  /* using terrain map to determine z height. could cause problems w.r.t. x,y drift
     really need a localization without drift (e.g. perception, mocap) when operating on terrain */
  //double relative_z = 0;
  //bool relative_z_valid = contact_feet_relative_z(q, c_s, relative_z);
  double terrain_z = 0;
  bool terrain_z_valid = contact_feet_terrain_z(q, c_s, terrain_z);

  if (contact_feet == 0) {
    q(Q_Z) += 0.5*(qd(Q_Z)+estim.qd_prev(Q_Z))*CTRL_LOOP_DURATION;
  } else {
    //q(Q_Z) = relative_z + z_sum/contact_feet;
    q(Q_Z) = terrain_z;
  }

  // body x,y. finite integral of velocity for now
  q(Q_X) += 0.5*(qd(Q_X)+estim.qd_prev(Q_X))*CTRL_LOOP_DURATION;
  q(Q_Y) += 0.5*(qd(Q_Y)+estim.qd_prev(Q_Y))*CTRL_LOOP_DURATION;

  if (!isnan(y(S_GROUND_TRUTH_X)) &&
      !isnan(y(S_GROUND_TRUTH_Y)) &&
      !isnan(y(S_GROUND_TRUTH_Z)) &&
      !isnan(y(S_GROUND_TRUTH_RX)) &&
      !isnan(y(S_GROUND_TRUTH_RY)) &&
      !isnan(y(S_GROUND_TRUTH_RZ))) {
    q(Q_X) += 0.01*(y(S_GROUND_TRUTH_X) - q(Q_X));
    q(Q_Y) += 0.01*(y(S_GROUND_TRUTH_Y) - q(Q_Y));
    if (contact_feet == 0) {
      q(Q_Z) += 0.01*(y(S_GROUND_TRUTH_Z) - q(Q_Z));
    }
    estim.heading_bias += 0.01*min_angle_diff(y(S_GROUND_TRUTH_RZ), q(Q_RZ));
  }

  estim.qd_prev = qd;

}

bool contact_feet_relative_z(const StateVec &q, const ContactState c_s, double &relative_z) {
  double ee_pos[3];
  double z_sum = 0;
  StateVec q_fk = q;
  q_fk(Q_Z) = 0;
  int contact_feet = 0;
  for (int contact = C_FL_EE; contact < C_BR_EE+1; ++contact) {
    if (c_s[contact]){
      fk(q_fk, ee_pos, robot.contacts[contact].frame);
      z_sum += ee_pos[2];
      contact_feet++;
    }
  }

  if (contact_feet != 0) {
    relative_z = -z_sum/contact_feet;
    return true;
  } else {
    relative_z = q[Q_Z]; //TODO this needs update, probably just have to compute from terrain map
    return false;
  }
}

bool contact_feet_terrain_z(const StateVec &q, const ContactState c_s, double &terrain_z) {
  double ee_pos[3];
  double z_sum = 0;
  StateVec q_fk = q;
  q_fk(Q_Z) = 0;
  int contact_feet = 0;
  for (int contact = C_FL_EE; contact < C_BR_EE+1; ++contact) {
    if (c_s[contact]){
      fk(q_fk, ee_pos, robot.contacts[contact].frame);
      z_sum += ee_pos[2];
      contact_feet++;
    
      Vector3d ee_pos_vec;
      ee_pos_vec << ee_pos[0], ee_pos[1], ee_pos[2]+q[Q_Z];
      Vector3d normal;
      Vector3d contact_pos;
      terrain_map(ee_pos_vec,
                  contact_pos,
                  normal);
      //std::cout << contact_pos << std::endl;
      z_sum -= contact_pos[2];
    }
  }

  if (contact_feet != 0) {
    terrain_z = -z_sum/contact_feet;
    return true;
  } else {
    terrain_z = 0;
    return false;
  }
}
