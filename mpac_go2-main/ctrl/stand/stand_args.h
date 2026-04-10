#ifndef STAND_ARGS_H
#define STAND_ARGS_H

namespace stand {

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

const int disc_min[] = {};
const int disc_max[] = {};
const double cont_min[] = {0.1,-0.5,-0.5,-0.5};
const double cont_max[] = {0.3,0.5,0.5,0.5};

}

#endif //STAND_ARGS_H
