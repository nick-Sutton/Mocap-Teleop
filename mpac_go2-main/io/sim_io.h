#ifndef SIM_IO_H
#define SIM_IO_H

#include "robot.h"

namespace sim_io {

typedef struct {
  std::vector<StateVec> q_buffer;
  std::vector<StateVec> qd_buffer;
  bool init;
  bool use_ground_truth;

} SimIo;

void init(SimIo &sim_io, char* args);
void finish(SimIo &sim_io);

void read(SimIo &sim_io, OutputVec &y,
          StateVec &q_sim, StateVec &qd_sim,
          StateVec &qdd_sim, ContactVec &c_sim,
          InputVec &u_sim);
void write(const ActuatorCmds &act_cmds,
                  StateVec &q_sim, StateVec &qd_sim, StateVec &qdd_sim,
                  ContactVec &c_sim, InputVec &u_sim);

}

#endif //SIM_IO_H

