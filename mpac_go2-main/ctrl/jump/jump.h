#ifndef JUMP_H
#define JUMP_H

#include "robot_types.h"
#include "ctrl_modes_core.h"
#include <eigen3/Eigen/Dense>

#include "ctrl_mode_class.h"

namespace jump {

PrimitiveBehavior* create();
void destroy(PrimitiveBehavior*);

}

#endif //JUMP_H
