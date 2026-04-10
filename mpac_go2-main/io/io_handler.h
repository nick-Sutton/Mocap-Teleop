#ifndef IO_HANDLER_H
#define IO_HANDLER_H

#include "robot.h"

namespace IO_handler {

typedef enum {
  MODE_SIM = 0,
  MODE_HARDWARE,
  MODE_MUJOCO_SIM,
} IOMode;

IO* create();
void destroy(IO*);

}

#endif //IO_HANDLER_H



