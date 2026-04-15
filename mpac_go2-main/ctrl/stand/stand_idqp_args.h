#ifndef STAND_IDQP_ARGS_H
#define STAND_IDQP_ARGS_H

namespace stand_idqp {

enum {
  NUM_DISC_ARGS = 0
};

enum {
  ARG_H = 0,
  ARG_RX,
  ARG_RY,
  ARG_RZ,
  NUM_CONT_ARGS
};

const int disc_min[NUM_DISC_ARGS] = {};
const int disc_max[NUM_DISC_ARGS] = {};
const double cont_min[NUM_CONT_ARGS] = {0.1,-0.5,-0.5,-0.5};
const double cont_max[NUM_CONT_ARGS] = {0.3,0.5,0.5,0.5};

}

#endif //STAND_IDQP_ARGS_H
