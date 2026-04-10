#ifndef HARD_STOP_H
#define HARD_STOP_H

#include "robot_types.h"
#include <eigen3/Eigen/Dense>

#include "ctrl_mode_class.h"

namespace hard_stop {

PrimitiveBehavior* create();
void destroy(PrimitiveBehavior*);

}

#endif //HARD_STOP_H
