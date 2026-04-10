#include "io_class.h"
#include "io/io_handler.h"
#include "io/sim_io.h"
#include "io/mujoco_io.h"
// #include "io/hardware_io.h"

namespace IO_handler {

class IO_handler: public IO {

IOMode io_mode;
sim_io::SimIo sim_io;
Mujoco_io mujoco_io;

int mode() {
  return io_mode;
};

void init(char* io_mode_str, char* io_mode_args) {
  if (strcmp(io_mode_str, "simulation") == 0) {
    io_mode = MODE_SIM;
    sim_io::init(sim_io, io_mode_args);
  }
  if (strcmp(io_mode_str, "hardware") == 0) {
    io_mode = MODE_HARDWARE;
    // hardware_io::init(io_mode_args);
    mujoco_io.init(true);
  }
  if (strcmp(io_mode_str, "mujoco") == 0) {
    io_mode = MODE_MUJOCO_SIM;
    mujoco_io.init(false);
  }
}

void read(OutputVec &y,
          StateVec &q_sim,
          StateVec &qd_sim,
          StateVec &qdd_sim,
          ContactVec &c_sim,
          InputVec &u_sim) {
  switch(io_mode) {
    case MODE_SIM:
      sim_io::read(sim_io,
                   y,
                   q_sim,
                   qd_sim,
                   qdd_sim,
                   c_sim,
                   u_sim);
      break;
    case MODE_HARDWARE:
      mujoco_io.read(y);
      break;
    case MODE_MUJOCO_SIM:
      mujoco_io.read(y);
      break;
    default:
      printf("Unhandled io mode.\n");
      break;
  }
}

void write(const ActuatorCmds &act_cmds,
           StateVec &q_sim,
           StateVec &qd_sim,
           StateVec &qdd_sim,
           ContactVec &c_sim,
           InputVec &u_sim) {
  switch(io_mode) {
    case MODE_SIM:
      sim_io::write(act_cmds,
                    q_sim,
                    qd_sim,
                    qdd_sim,
                    c_sim,
                    u_sim);
      break;
    case MODE_HARDWARE:
      mujoco_io.update(act_cmds);
      mujoco_io.write();
      break;
    //   hardware_io::write(act_cmds);
    //   break;
    case MODE_MUJOCO_SIM:
      mujoco_io.update(act_cmds);
      mujoco_io.write();
      break;
    default:
      printf("Unhandled io mode.\n");
      break;
  }
}

void finish() {
  switch(io_mode) {
    case MODE_SIM:
      sim_io::finish(sim_io);
      break;
    case MODE_HARDWARE:
      mujoco_io.finish();
      break;
    //   hardware_io::finish();
    //   break;
    case MODE_MUJOCO_SIM:
      mujoco_io.finish();
      break;
    default:
      printf("Unhandled io mode.\n");
      break;
  }
}


};

IO* create() {
  return new IO_handler;
}
void destroy(IO* io) {
  delete io;
}

}
