#include <stdio.h>
#include <time.h>
#include <math.h>

#include "robot.h"
#include "tlm.h"
#include "ctrl_modes_core.h"
#include <iostream>

#include "hdf5.h"

void fill_tlm(Robot &robot, Telemetry *tlm) {
  for (int i = 0; i < NUM_Q; ++i) {
    tlm->q[i] = robot.q[i];
    tlm->qd[i] = robot.qd[i];
    tlm->qdd_sim[i] = robot.qdd_sim[i];
  }

  for (int i = 0; i < NUM_U; ++i) {
    tlm->u_des[i] = robot.act_cmds.u[i];
    tlm->act_mode[i] = robot.act_cmds.mode[i];
  }
  for (int i = 0; i < NUM_U; ++i) {
    tlm->q_des[i] = robot.act_cmds.q[i];
    tlm->qd_des[i] = robot.act_cmds.qd[i];
  }

  for (int i = 0; i < 4; ++i) {
    tlm->f[i] = robot.y[F_FRC_FL_EE + i] - robot.estim.foot_force_bias[i];
 
    // tlm->f[i] = robot.act_cmds.fc(i,0) - robot.estim.foot_force_bias[i];   
    // std::cout<<"contact: "<<robot.act_cmds.fc <<std::endl;
  }

  for (int i = 0; i < NUM_U; ++i) {
    tlm->u[i] = robot.y[S_U_FL1+i];
    tlm->temp[i] = robot.y[S_TEMP_FL1+i];
  }

  tlm->cycle_cnt = robot.cycle_cnt;
  tlm->cycle_duration = 1e-6*(1e9*(robot.cycle_start_t.tv_sec - robot.cycle_start_t_prev.tv_sec) +
                              robot.cycle_start_t.tv_nsec - robot.cycle_start_t_prev.tv_nsec);
  tlm->compute_duration = 1e-6*(1e9*(robot.compute_end_t.tv_sec - robot.cycle_start_t.tv_sec) +
                                robot.compute_end_t.tv_nsec - robot.cycle_start_t.tv_nsec);
  tlm->tictoc = 1e-6*(1e9*(robot.toc.tv_sec - robot.tic.tv_sec) +
                      robot.toc.tv_nsec - robot.tic.tv_nsec);

  tlm->cycle_start_time.tv_sec = robot.cycle_start_t.tv_sec + robot.clock_offset.tv_sec;
  tlm->cycle_start_time.tv_nsec = robot.cycle_start_t.tv_nsec + robot.clock_offset.tv_nsec;
  if (tlm->cycle_start_time.tv_nsec >= 1e9) {
    tlm->cycle_start_time.tv_nsec -= 1e9;
    tlm->cycle_start_time.tv_sec++;
  }

  tlm->path_compute_duration = robot.path_compute_duration;

  tlm->ctrl_curr = robot.ctrl_curr;
  tlm->ctrl_next = robot.ctrl_next;
  tlm->ctrl_des = robot.ctrl_des;
  
  tlm->prim_path_len = std::min(8,(int)(robot.prim_path.size()-1));
  for (int i = 0; i < tlm->prim_path_len; ++i) {
    tlm->prim_path[i] = robot.prim_path.at(i+1).action_from_parent;
  }

}

void fill_derived_tlm(DerivedTelemetry *derived_tlm, Telemetry *tlm) {
  derived_tlm->tlm = *tlm;

  char buff[24];
  strftime(buff, sizeof buff, "%F", localtime(&tlm->cycle_start_time.tv_sec));
  sprintf(derived_tlm->local_date, "%s", buff);
  strftime(buff, sizeof buff, "%T", localtime(&tlm->cycle_start_time.tv_sec));
  sprintf(derived_tlm->local_time, "%s.%03d", buff, (int) (1e-6*tlm->cycle_start_time.tv_nsec));

  ctrl_mode_to_string(derived_tlm->ctrl_mode_curr, tlm->ctrl_curr.mode);
  strcpy(derived_tlm->ctrl_mode_curr + strlen(derived_tlm->ctrl_mode_curr), "(");
  ctrl_args_to_string(derived_tlm->ctrl_mode_curr + strlen(derived_tlm->ctrl_mode_curr), tlm->ctrl_curr);
  strcpy(derived_tlm->ctrl_mode_curr + strlen(derived_tlm->ctrl_mode_curr), ")");

  ctrl_mode_to_string(derived_tlm->ctrl_mode_next, tlm->ctrl_next.mode);
  strcpy(derived_tlm->ctrl_mode_next + strlen(derived_tlm->ctrl_mode_next), "(");
  ctrl_args_to_string(derived_tlm->ctrl_mode_next + strlen(derived_tlm->ctrl_mode_next), tlm->ctrl_next);
  strcpy(derived_tlm->ctrl_mode_next + strlen(derived_tlm->ctrl_mode_next), ")");

  ctrl_mode_to_string(derived_tlm->ctrl_mode_des, tlm->ctrl_des.mode);
  strcpy(derived_tlm->ctrl_mode_des + strlen(derived_tlm->ctrl_mode_des), "(");
  ctrl_args_to_string(derived_tlm->ctrl_mode_des + strlen(derived_tlm->ctrl_mode_des), tlm->ctrl_des);
  strcpy(derived_tlm->ctrl_mode_des + strlen(derived_tlm->ctrl_mode_des), ")");

  for (int i = 0; i < 8; ++i) {
    strcpy(derived_tlm->prim_path[i], "");
  }
  for (int i = 0; i < tlm->prim_path_len; ++i) {
    ctrl_mode_to_string(derived_tlm->prim_path[i], tlm->prim_path[i].mode);
    strcpy(derived_tlm->prim_path[i] + strlen(derived_tlm->prim_path[i]), "(");
    ctrl_args_to_string(derived_tlm->prim_path[i] + strlen(derived_tlm->prim_path[i]), tlm->prim_path[i]);
    strcpy(derived_tlm->prim_path[i] + strlen(derived_tlm->prim_path[i]), ")");
  }

  derived_tlm->epoch_time = tlm->cycle_start_time.tv_sec + 1e-9*tlm->cycle_start_time.tv_nsec;

}

void create_common_timeseries_type(hid_t &timeseries_type) {

  hid_t str32 = H5Tcopy (H5T_C_S1);
  size_t size32 = 32 * sizeof(char);
  H5Tset_size(str32, size32);
  hid_t str128 = H5Tcopy (H5T_C_S1);
  size_t size128 = 128 * sizeof(char);
  H5Tset_size(str128, size128);

  hsize_t dims_state[1] = {NUM_Q};
  hid_t state_vec = H5Tarray_create(H5T_NATIVE_DOUBLE, 1, dims_state);
  hsize_t dims_input[1] = {NUM_U};
  hid_t input_vec = H5Tarray_create(H5T_NATIVE_DOUBLE, 1, dims_input);
  hid_t input_mode_vec = H5Tarray_create(H5T_NATIVE_INT, 1, dims_input);
  hsize_t dims_force[1] = {4};
  hid_t force_vec = H5Tarray_create(H5T_NATIVE_DOUBLE, 1, dims_force);
  hsize_t dims_path[1] = {8};
  hid_t path_vec = H5Tarray_create(str128, 1, dims_path);

  timeseries_type = H5Tcreate (H5T_COMPOUND, sizeof (DerivedTelemetry));
  H5Tinsert(timeseries_type, "local_date", HOFFSET (DerivedTelemetry, local_date), str32);
  H5Tinsert(timeseries_type, "local_time", HOFFSET (DerivedTelemetry, local_time), str32);
  H5Tinsert(timeseries_type, "epoch_time", HOFFSET (DerivedTelemetry, epoch_time), H5T_NATIVE_LDOUBLE);
  H5Tinsert(timeseries_type, "cycle_count", HOFFSET (Telemetry, cycle_cnt), H5T_NATIVE_ULONG);
  H5Tinsert(timeseries_type, "cycle_duration", HOFFSET (DerivedTelemetry, tlm.cycle_duration), H5T_NATIVE_DOUBLE);
  H5Tinsert(timeseries_type, "compute_duration", HOFFSET (DerivedTelemetry, tlm.compute_duration), H5T_NATIVE_DOUBLE);
  H5Tinsert(timeseries_type, "tictoc", HOFFSET (DerivedTelemetry, tlm.tictoc), H5T_NATIVE_DOUBLE);
  H5Tinsert(timeseries_type, "path_compute_duration", HOFFSET (DerivedTelemetry, tlm.path_compute_duration), H5T_NATIVE_DOUBLE);
  H5Tinsert(timeseries_type, "q", HOFFSET (DerivedTelemetry, tlm.q), state_vec);
  H5Tinsert(timeseries_type, "qd", HOFFSET (DerivedTelemetry, tlm.qd), state_vec);
  H5Tinsert(timeseries_type, "u", HOFFSET (DerivedTelemetry, tlm.u), input_vec);
  H5Tinsert(timeseries_type, "temp", HOFFSET (DerivedTelemetry, tlm.temp), input_vec);
  H5Tinsert(timeseries_type, "act_mode", HOFFSET (DerivedTelemetry, tlm.act_mode), input_mode_vec);
  H5Tinsert(timeseries_type, "q_des", HOFFSET (DerivedTelemetry, tlm.q_des), input_vec);
  H5Tinsert(timeseries_type, "qd_des", HOFFSET (DerivedTelemetry, tlm.qd_des), input_vec);
  H5Tinsert(timeseries_type, "u_des", HOFFSET (DerivedTelemetry, tlm.u_des), input_vec);
  H5Tinsert(timeseries_type, "f", HOFFSET (DerivedTelemetry, tlm.f), force_vec);
  H5Tinsert(timeseries_type, "ctrl_mode_curr", HOFFSET (DerivedTelemetry, ctrl_mode_curr), str128);
  H5Tinsert(timeseries_type, "ctrl_mode_next", HOFFSET (DerivedTelemetry, ctrl_mode_next), str128);
  H5Tinsert(timeseries_type, "ctrl_mode_des", HOFFSET (DerivedTelemetry, ctrl_mode_des), str128);
  H5Tinsert(timeseries_type, "prim_path", HOFFSET (DerivedTelemetry, prim_path), path_vec);
  H5Tinsert(timeseries_type, "prim_tlm_index", HOFFSET (DerivedTelemetry, prim_tlm_index), H5T_NATIVE_ULONG);

}

void fill_attributes(std::vector<TelemetryAttribute> &attributes) {
  TelemetryAttributes attr;

  Robot robot;
  robot_params_init(robot);

  memset(&attr, 0, sizeof(TelemetryAttributes));
  attr.epoch_time       = (TelemetryAttribute) {"", "sec"};
  attr.cycle_duration   = (TelemetryAttribute) {"", "ms", 0.5, 0.9, 1.1, 1.5};
  attr.compute_duration = (TelemetryAttribute) {"", "ms", 0, 0, 0.8, 1.0};
  attr.tictoc           = (TelemetryAttribute) {"", "ms"};
  attr.path_compute_duration = (TelemetryAttribute) {"", "ms", 0, 0, 0, 10.0};

  for (int i = 0; i < NUM_Q; ++i) {
    strcpy(attr.q[i].name, robot.joints[i].name);
    strcpy(attr.q[i].units, robot.joints[i].type == REVOLUTE ? "rad":"m");
    attr.q[i].crit_lo = robot.joints[i].lower_pos_lim;
    attr.q[i].warn_lo = robot.joints[i].lower_pos_lim+0.05;
    attr.q[i].crit_hi = robot.joints[i].upper_pos_lim;
    attr.q[i].warn_hi = robot.joints[i].upper_pos_lim-0.05;
  }

  for (int i = 0; i < NUM_Q; ++i) {
    sprintf(attr.qd[i].name, "QD%s", &robot.joints[i].name[1]);
    strcpy(attr.qd[i].units, robot.joints[i].type == REVOLUTE ? "rad/s":"m/s");
    attr.qd[i].crit_lo = -robot.joints[i].vel_lim;
    attr.qd[i].warn_lo = -robot.joints[i].vel_lim+2;
    attr.qd[i].crit_hi = robot.joints[i].vel_lim;
    attr.qd[i].warn_hi = robot.joints[i].vel_lim-2;
  }

  for (int i = 0; i < NUM_U; ++i) {
    sprintf(attr.u[i].name, "U%s", &robot.joints[NUM_Q-NUM_U+i].name[1]);
    strcpy(attr.u[i].units, robot.joints[NUM_Q-NUM_U+i].type == REVOLUTE ? "Nm":"N");
    attr.u[i].crit_lo = -robot.joints[NUM_Q-NUM_U+i].effort_lim;
    attr.u[i].warn_lo = -robot.joints[NUM_Q-NUM_U+i].effort_lim+5;
    attr.u[i].crit_hi = robot.joints[NUM_Q-NUM_U+i].effort_lim;
    attr.u[i].warn_hi = robot.joints[NUM_Q-NUM_U+i].effort_lim-5;
  }

  for (int i = 0; i < NUM_U; ++i) {
    sprintf(attr.act_mode[i].name, "%s_act_mode", &robot.joints[NUM_Q-NUM_U+i].name[2]);
  }

  memcpy(&attr.q_des, &attr.q[NUM_Q-NUM_U], sizeof(attr.q_des));
  for (int i = 0; i < NUM_U; ++i) {
    sprintf(attr.q_des[i].name, "%s_des", robot.joints[NUM_Q-NUM_U+i].name);
  }

  memcpy(&attr.qd_des, &attr.qd[NUM_Q-NUM_U], sizeof(attr.qd_des));
  for (int i = 0; i < NUM_U; ++i) {
    sprintf(attr.qd_des[i].name, "QD%s_des", &robot.joints[NUM_Q-NUM_U+i].name[1]);
  }

  memcpy(&attr.u_des, &attr.u, sizeof(attr.u));
  for (int i = 0; i < NUM_U; ++i) {
    sprintf(attr.u_des[i].name, "U%s_des", &robot.joints[NUM_Q-NUM_U+i].name[1]);
  }

  attr.f[0] = (TelemetryAttribute) {"F_FL", "N", -150,-100,100,150};
  attr.f[1] = (TelemetryAttribute) {"F_FR", "N", -150,-100,100,150};
  attr.f[2] = (TelemetryAttribute) {"F_BL", "N", -150,-100,100,150};
  attr.f[3] = (TelemetryAttribute) {"F_BR", "N", -150,-100,100,150};

  attr.temp[U_FL1] = (TelemetryAttribute) {"TEMP_FL1", "degC", -10,0,40,50};
  attr.temp[U_FL2] = (TelemetryAttribute) {"TEMP_FL2", "degC", -10,0,40,50};
  attr.temp[U_FL3] = (TelemetryAttribute) {"TEMP_FL3", "degC", -10,0,40,50};
  attr.temp[U_FR1] = (TelemetryAttribute) {"TEMP_FR1", "degC", -10,0,40,50};
  attr.temp[U_FR2] = (TelemetryAttribute) {"TEMP_FR2", "degC", -10,0,40,50};
  attr.temp[U_FR3] = (TelemetryAttribute) {"TEMP_FR3", "degC", -10,0,40,50};
  attr.temp[U_BL1] = (TelemetryAttribute) {"TEMP_BL1", "degC", -10,0,40,50};
  attr.temp[U_BL2] = (TelemetryAttribute) {"TEMP_BL2", "degC", -10,0,40,50};
  attr.temp[U_BL3] = (TelemetryAttribute) {"TEMP_BL3", "degC", -10,0,40,50};
  attr.temp[U_BR1] = (TelemetryAttribute) {"TEMP_BR1", "degC", -10,0,40,50};
  attr.temp[U_BR2] = (TelemetryAttribute) {"TEMP_BR2", "degC", -10,0,40,50};
  attr.temp[U_BR3] = (TelemetryAttribute) {"TEMP_BR3", "degC", -10,0,40,50};


  //TODO probably a cleaner way...
  TelemetryAttribute *attr_array = (TelemetryAttribute*) &attr;
  for (int i=0;i < sizeof(TelemetryAttributes)/sizeof(TelemetryAttribute); ++i) {
    attributes.push_back(attr_array[i]);
  }
}

void create_common_attribute_type(hid_t &attr_type) {

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

  hsize_t dims_state[1] = {NUM_Q};
  hid_t state_vec = H5Tarray_create(single_attr_type, 1, dims_state);
  hsize_t dims_input[1] = {NUM_U};
  hid_t input_vec = H5Tarray_create(single_attr_type, 1, dims_input);
  hsize_t dims_force[1] = {4};
  hid_t force_vec = H5Tarray_create(single_attr_type, 1, dims_force);

  attr_type = H5Tcreate (H5T_COMPOUND, sizeof (TelemetryAttributes));
  H5Tinsert(attr_type, "epoch_time", HOFFSET (TelemetryAttributes, epoch_time), single_attr_type);
  H5Tinsert(attr_type, "cycle_duration", HOFFSET (TelemetryAttributes, cycle_duration), single_attr_type);
  H5Tinsert(attr_type, "compute_duration", HOFFSET (TelemetryAttributes, compute_duration), single_attr_type);
  H5Tinsert(attr_type, "tictoc", HOFFSET (TelemetryAttributes, tictoc), single_attr_type);
  H5Tinsert(attr_type, "path_compute_duration", HOFFSET (TelemetryAttributes, path_compute_duration), single_attr_type);
  H5Tinsert(attr_type, "q", HOFFSET (TelemetryAttributes, q), state_vec);
  H5Tinsert(attr_type, "qd", HOFFSET (TelemetryAttributes, qd), state_vec);
  H5Tinsert(attr_type, "u", HOFFSET (TelemetryAttributes, u), input_vec);
  H5Tinsert(attr_type, "act_mode", HOFFSET (TelemetryAttributes, act_mode), input_vec);
  H5Tinsert(attr_type, "q_des", HOFFSET (TelemetryAttributes, q_des), input_vec);
  H5Tinsert(attr_type, "qd_des", HOFFSET (TelemetryAttributes, qd_des), input_vec);
  H5Tinsert(attr_type, "u_des", HOFFSET (TelemetryAttributes, u_des), input_vec);
  H5Tinsert(attr_type, "f", HOFFSET (TelemetryAttributes, f), force_vec);
  H5Tinsert(attr_type, "temp", HOFFSET (TelemetryAttributes, temp), input_vec);

}
