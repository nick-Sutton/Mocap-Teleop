#include "robot.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <stdio.h>
#include <fstream>

#include "ctrl_core.h"
#include "ctrl/ctrl_utils/traj_gen.h"
#include "ctrl/ctrl_utils/joint_pos_pd.h"
#include "traj_track.h"
#include "traj_track_args.h"
#include "fk.h"
#include "ctrl_mode_class.h"

#include <string>

extern Robot robot;

namespace traj_track {

typedef struct {
  double t;
  InputVec q;
  InputVec qd;
  InputVec u;
  
} Waypoint;

typedef struct {
  joint_pos_pd::JointPosPD jppd;
  traj_gen::CubicPoly traj[NUM_U];
  double t;
  double traj_start_t;
  std::vector<Waypoint> waypoints;
  int curr_waypoint;
  int next_waypoint;

} Data;

class TrajTrack: public PrimitiveBehavior {

Data data;

int disc_min[NUM_DISC_ARGS] = {0};
int disc_max[NUM_DISC_ARGS] = {0};
double cont_min[NUM_CONT_ARGS] = {};
double cont_max[NUM_CONT_ARGS] = {};

public:
ArgAttributes get_arg_attributes() {
  return (ArgAttributes) {NUM_DISC_ARGS, NUM_CONT_ARGS,
                          disc_min, disc_max,
                          cont_min, cont_max,
                                 0,        0};
}
void init(const StateVec &q,
          const StateVec &qd,
          const ContactState c_s,
          const Args &args) {
  char filename[256];
  filename_from_arg(args, filename);
  read_traj(data, filename);
  InputVec q_des = data.waypoints.at(0).q;
  InputVec qd_des = data.waypoints.at(0).qd;
  joint_pos_pd::init(data.jppd, q.tail(NUM_U), qd.tail(NUM_U), q_des, qd_des, -1);
  data.t = 0;
  data.curr_waypoint = 0;
  data.next_waypoint = 1;
  gen_cubic_traj(data);
}

void execute(const StateVec &q,
             const StateVec &qd,
             const ContactState c_s,
             const Args &args, 
             ActuatorCmds &act_cmds) {
  if (joint_pos_pd::execute(data.jppd, act_cmds, q, qd)) {
    if (data.t > data.waypoints.at(data.next_waypoint).t) {
      data.curr_waypoint = data.next_waypoint;
      //data.next_waypoint = (data.curr_waypoint+1) % (data.waypoints.size()-1);
      if (data.curr_waypoint+1 < data.waypoints.size()) {
        data.next_waypoint = data.curr_waypoint+1;
        gen_cubic_traj(data);
      }
    }
    for (int i = 0; i < NUM_U; ++i) {
      double q_cmd, qd_cmd;
      traj_gen::cubic_poly_eval(data.traj[i],
                                data.t - data.traj_start_t,
                                q_cmd, qd_cmd);
      act_cmds.mode[i] = CMD_MODE_TAU_VEL_POS;
      act_cmds.u[i] = data.waypoints.at(data.curr_waypoint).u[i];
      act_cmds.qd[i] = qd_cmd;
      act_cmds.q[i] = q_cmd;
      act_cmds.kp[i] = 200;
      act_cmds.kd[i] = 5;
    }

    if (data.t < data.waypoints.back().t) {
      data.t += CTRL_LOOP_DURATION;
      //data.t = fmod(data.t, data.waypoints.back().t);
    }
  }
}

/* Note for softstop, we always want to be able to transition in case of emergency */
void setpoint(const StateVec &q,
              const StateVec &qd,
              const ContactState c_s,
              const Args &args,
              const double delta_t,
              StateVec &q_des,
              StateVec &qd_des,
              ContactState &c_s_des) const {
  //TODO fill in
  q_des = q;
  qd_des = qd;
  memcpy(&c_s_des, c_s, sizeof(ContactState));
}

MatrixXd setpoint_jac(const StateVec &q,
                           const StateVec &qd,
                           const ContactState c_s,
                           const Args &args,
                           const double delta_t) const {
  return MatrixXd::Zero((int)NUM_Q+NUM_Q, 1);
}

bool in_sroa_underestimate(const StateVec &q,
                           const StateVec &qd,
                           const ContactState c_s,
                           const Args &args,
                           const double delta_t) const {
  return false;
  //return true;
  if (!in_safe_set(q,qd,c_s,args,delta_t)) {
    return false;
  }
  return true;
}
bool in_sroa_overestimate(const StateVec &q,
                          const StateVec &qd,
                          const ContactState c_s,
                          const Args &args,
                          const double delta_t) const {
  return false;
  //return true;
  if (!in_safe_set(q,qd,c_s,args,delta_t)) {
    return false;
  }
  return true;
}
bool in_safe_set(const StateVec &q,
                 const StateVec &qd,
                 const ContactState c_s,
                 const Args &args,
                 const double delta_t) const {
  return in_vel_limits(qd) && in_pos_limits(q);
}

private:

void read_traj(Data &data, char* filename) {
  std::ifstream myfile (filename);
  std::string line;
  if (myfile.is_open())
  {
    while (getline(myfile,line, ',')) {
      Waypoint waypoint;
      waypoint.t = std::stod(line);

      for (int i = 0; i < NUM_U; ++i) {
        getline(myfile,line, ',');
        waypoint.q[i] = std::stod(line);
      }
      for (int i = 0; i < NUM_U; ++i) {
        getline(myfile,line, ',');
        waypoint.qd[i] = std::stod(line);
      }
      for (int i = 0; i < NUM_U; ++i) {
        if (i == NUM_U-1){
          getline(myfile,line);
        } else {
          getline(myfile,line, ',');
        }
        waypoint.u[i] = std::stod(line);
      }

      data.waypoints.push_back(waypoint);
    }

    myfile.close();
  } else {
    printf("file: %s not found.\n",filename);
  }

  //printf("num_waypoints: %d\n", data.waypoints.size());
  //for (int i = 0; i < data.waypoints.size(); ++i) {
  //  printf("%f, ", data.waypoints.at(i).t);
  //    for (int j = 0; j < NUM_U; ++j) {
  //      printf("%f, ", data.waypoints.at(i).q[j]);
  //    }
  //    for (int j = 0; j < NUM_U; ++j) {
  //      printf("%f, ", data.waypoints.at(i).qd[j]);
  //    }
  //    for (int j = 0; j < NUM_U; ++j) {
  //      printf("%f, ", data.waypoints.at(i).u[j]);
  //    }
  //    printf("\n");
  //}

}

void gen_cubic_traj(Data &data) {
  double duration = data.waypoints.at(data.next_waypoint).t -
                    data.waypoints.at(data.curr_waypoint).t; //need some mod for periodic here
  for (int i = 0; i < NUM_U; ++i) {
    data.traj[i] = traj_gen::cubic_poly_gen(data.waypoints.at(data.curr_waypoint).q[i],
                                            data.waypoints.at(data.curr_waypoint).qd[i],
                                            data.waypoints.at(data.next_waypoint).q[i],
                                            data.waypoints.at(data.next_waypoint).qd[i],
                                            duration);
  }
  data.traj_start_t = data.t;

}

void filename_from_arg(const Args &args, char* filename) {
  switch(args.disc[ARG_TRAJ]) {
    case 0:
      sprintf(filename, "../ctrl/traj_track/default.csv");
      break;
    default:
      sprintf(filename, "../ctrl/traj_track/default.csv");
      break;
  }
}

};

PrimitiveBehavior* create() {
  return new TrajTrack;
}
void destroy(PrimitiveBehavior* prim) {
  delete prim;
}

}
