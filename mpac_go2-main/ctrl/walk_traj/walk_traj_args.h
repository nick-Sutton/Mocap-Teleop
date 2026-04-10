#ifndef WALK_TRAJ_ARGS_H
#define WALK_TRAJ_ARGS_H

namespace walk_traj {

enum {
  ARG_GAIT = 0,
  NUM_DISC_ARGS
};

enum {
  NUM_CONT_ARGS = 0
};

const int disc_min[] = {0};
const int disc_max[] = {3};
const double cont_min[] = {};
const double cont_max[] = {};

}

#endif //WALK_TRAJ_ARGS_H
