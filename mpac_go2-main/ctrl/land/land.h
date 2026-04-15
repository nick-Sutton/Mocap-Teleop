#ifndef LAND_H
#define LAND_H

#include "robot_types.h"
#include "ctrl_modes_core.h"
#include <eigen3/Eigen/Dense>
#include "ctrl_mode_class.h"

namespace land {

PrimitiveBehavior* create();
void destroy(PrimitiveBehavior*);

}

#endif //LAND_H
