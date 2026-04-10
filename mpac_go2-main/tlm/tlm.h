#ifndef TLM_H
#define TLM_H

#include <sys/time.h>
#include "robot.h"
#include "hdf5.h"

typedef struct {

  struct timespec cycle_start_time;
  unsigned long cycle_cnt;
  double cycle_duration;
  double compute_duration;
  double tictoc;
  double path_compute_duration;
  double q[NUM_Q];
  double qd[NUM_Q];
  double qdd_sim[NUM_Q];
  double u[NUM_U];
  ActuatorCmdMode act_mode[NUM_U];
  double u_des[NUM_U];
  double q_des[NUM_U];
  double qd_des[NUM_U];
  double f[4];
  double temp[NUM_U];
  CtrlState ctrl_curr;
  CtrlState ctrl_next;
  CtrlState ctrl_des;
  int prim_path_len;
  CtrlState prim_path[8];

} Telemetry;

typedef struct {
  Telemetry tlm;

  long double epoch_time;
  char local_date[32];
  char local_time[32];
  char ctrl_mode_curr[128];
  char ctrl_mode_next[128];
  char ctrl_mode_des[128];
  char prim_path[8][128];
  unsigned long prim_tlm_index;

} DerivedTelemetry;

typedef struct {
  TelemetryAttribute epoch_time;
  TelemetryAttribute cycle_duration;
  TelemetryAttribute compute_duration;
  TelemetryAttribute tictoc;
  TelemetryAttribute path_compute_duration;
  TelemetryAttribute q[NUM_Q];
  TelemetryAttribute qd[NUM_Q];
  TelemetryAttribute u[NUM_U];
  TelemetryAttribute act_mode[NUM_U];
  TelemetryAttribute u_des[NUM_U];
  TelemetryAttribute q_des[NUM_U];
  TelemetryAttribute qd_des[NUM_U];
  TelemetryAttribute f[12];
  TelemetryAttribute temp[NUM_U];

} TelemetryAttributes;

void fill_tlm(Robot &robot, Telemetry *tlm);
void fill_derived_tlm(DerivedTelemetry* derived_tlm, Telemetry *tlm);
void fill_attributes(std::vector<TelemetryAttribute> &attributes);
void create_common_timeseries_type(hid_t &timeseries_type);
void create_common_attribute_type(hid_t &attribute_type);

#endif //TLM_H
