#ifndef WALK_QUASI_PLANNED_ARGS_H
#define WALK_QUASI_PLANNED_ARGS_H

namespace walk_quasi_planned {

enum {
  ARG_MODE = 0,
  NUM_DISC_ARGS
};

enum {
  ARG_H = 0,
  ARG_VX,
  ARG_VY,
  ARG_VRZ,
  ARG_X,
  ARG_Y,
  ARG_RZ,
  NUM_CONT_ARGS
};

const int disc_min[] = {0};
const int disc_max[] = {1};
const double cont_min[] = {0.1,-0.05,-0.02,-0.1};
const double cont_max[] = {0.3,0.05,0.02,0.1};

}

#endif //WALK_QUASI_PLANNED_ARGS_H
