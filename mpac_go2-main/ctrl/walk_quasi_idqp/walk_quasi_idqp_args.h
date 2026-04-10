#ifndef WALK_QUASI_IDQP_ARGS_H
#define WALK_QUASI_IDQP_ARGS_H

namespace walk_quasi_idqp {

enum {
  NUM_DISC_ARGS = 0
};

enum {
  ARG_H = 0,
  ARG_VX,
  ARG_VY,
  ARG_VRZ,
  NUM_CONT_ARGS
};

const int disc_min[] = {};
const int disc_max[] = {};
const double cont_min[] = {0.1,-0.05,-0.02,-0.1};
const double cont_max[] = {0.3,0.05,0.02,0.1};

}

#endif //WALK_QUASI_IDQP_ARGS_H
