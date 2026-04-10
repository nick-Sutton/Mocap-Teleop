#ifndef SOFT_STOP_H
#define SOFT_STOP_H

#include "ctrl_mode_class.h"

namespace soft_stop {

PrimitiveBehavior* create();
void destroy(PrimitiveBehavior*);

}

#endif //SOFT_STOP_H
