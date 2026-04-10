#ifndef JUMP_ARGS_H
#define JUMP_ARGS_H

namespace jump {

enum {
  NUM_DISC_ARGS = 0
};

enum {
  ARG_X_VEL = 0,
  ARG_Y_VEL = 1,
  ARG_Z_VEL = 2,
  NUM_CONT_ARGS
};

const int disc_min[] = {};
const int disc_max[] = {};
const double cont_min[] = {0};
const double cont_max[] = {2.5};

}

#endif //STAND_ARGS_H
