#ifndef STATE_ESTIM_H
#define STATE_ESTIM_H

#include "robot.h"
#include "pinocchio/multibody/data.hpp"

typedef struct {
  StateVec qd_prev;
  double foot_force_filt[4];
  double foot_force_bias[4];
  double accel_bias[3];
  double foot_contact_pos[4][3];
  bool contact[4];
  double heading_bias;

} StateEstimator;

void state_estim_init(StateEstimator &estim);
void state_estim_read_cal_file(StateEstimator &estim);
void state_estim(StateEstimator &estim, const OutputVec &y, StateVec &q, StateVec &qd, ContactState c_s);

bool contact_feet_relative_z(const StateVec &q,
                             const ContactState c_s,
                             double &relative_z);
bool contact_feet_terrain_z(const StateVec &q,
                            const ContactState c_s,
                            double &relative_z);
#endif //STATE_ESTIM_H


