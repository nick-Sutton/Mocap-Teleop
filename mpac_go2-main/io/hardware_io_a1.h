#ifndef HARDWARE_IO_A1_H
#define HARDWARE_IO_A1_H

#include "robot.h"

namespace hardware_io_a1 {

void init(char* args);
void read(OutputVec &y);
void write(const ActuatorCmds &act_cmds);
void finish();

}

#endif //HARDWARE_IO_H


