#ifndef LIE_H
#define LIE_H

#include "robot_types.h"
#include <eigen3/Eigen/Dense>
#include "ctrl/ctrl_utils/joint_pos_pd.h"

namespace lie {

PrimitiveBehavior* create();
void destroy(PrimitiveBehavior*);

}

#endif //LIE_H
