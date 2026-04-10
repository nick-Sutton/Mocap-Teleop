#ifndef HARDWARE_IO_H
#define HARDWARE_IO_H

#include "robot.h"

namespace hardware_io {

void init(char* args);
void read(OutputVec &y);
void write(const ActuatorCmds &act_cmds);
void finish();

}

#endif //HARDWARE_IO_H


