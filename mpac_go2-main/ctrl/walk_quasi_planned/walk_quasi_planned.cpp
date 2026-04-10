#include "robot.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <thread>
#include <future>

#include "osqp.h"
#include "OsqpEigen/OsqpEigen.h"
#include <eigen3/Eigen/Dense>

#include "math_utils.h"
#include "ctrl_core.h"
#include "full_model_fd.h"
#include "fk.h"
#include "ik.h"
#include "walk_quasi_planned.h"
#include "walk_quasi_planned_args.h"
#include "ctrl/ctrl_utils/traj_gen.h"
#include "pinocchio/algorithm/kinematics.hpp"
#include "pinocchio/algorithm/frames.hpp"

extern Robot robot;

using namespace Eigen;

namespace walk_quasi_planned {

const double step_height = 0.075;//0.3
const double step_time = 1;//0.3
const double dwell_time = 1;

typedef struct {
  Contact swing_foot;
  double foot_start_pos[4][3];
  double foot_end_pos[4][3];
  double com_pos[3];
  double heading;

  double cost;

  StateVec q_start;
  StateVec q_end;

} Step;

typedef struct {
  int prnt_idx; //parent node index
  Step step;
  int step_num;
  double cost;
  bool infeasible[2][4];
  int depth;
} Node;


typedef struct {
  double foot_pos_init[4][3];
  double t;
  std::deque<Step> steps;
  traj_gen::CubicPoly traj[6]; //x,y,z,rx,ry,rz
  traj_gen::CubicPoly swing_traj[4]; //x,y,z1,z2

  double centroid_init[3];
  double heading_init[3];
  double global_z_offset; //TODO maybe better to plan in global since we eventually use global for terrain...

  int step_num;
  Contact prev_foot;

  std::future<std::deque<Step>> plan_future;
  bool planning;
  
} Data;

class WalkQuasiPlanned: public PrimitiveBehavior {

Data data;

public:
ArgAttributes get_arg_attributes() {
  return (ArgAttributes) {NUM_DISC_ARGS, NUM_CONT_ARGS,
                          disc_min, disc_max,
                          cont_min, cont_max,
                                 0,        0};
}
void init(const StateVec &q_in,
          const StateVec &qd,
          const ContactState c_s,
          const Args &args) { 
  double relative_z = 0;
  bool relative_z_valid = contact_feet_relative_z(q_in, c_s, relative_z);
  StateVec q = q_in;
  q[Q_Z] = relative_z;
  data.global_z_offset = q_in[Q_Z] - relative_z;

  data.prev_foot = (Contact) -1;
  compute_foot_pos(q, data.foot_pos_init);

  data.centroid_init[0] = 0;
  data.centroid_init[1] = 0;
  data.centroid_init[2] = 0.22;
  for (int i=0; i<4; ++i) {
    data.centroid_init[0] += data.foot_pos_init[i][0]/4.0;
    data.centroid_init[1] += data.foot_pos_init[i][1]/4.0;
  }
  data.heading_init[2] = q[Q_RZ];

  data.step_num = 0;
  data.steps = plan_steps(q, args, data.prev_foot);
  compute_com_traj(q, qd);
  compute_foot_traj(q, qd, data.steps.at(0));

  data.t = 0;
  data.planning = false;
}

void execute(const StateVec &q_in,
             const StateVec &qd,
             const ContactState c_s,
             const Args &args, 
             ActuatorCmds &act_cmds) {
  double relative_z = 0;
  bool relative_z_valid = contact_feet_relative_z(q_in, c_s, relative_z);
  StateVec q = q_in;
  q[Q_Z] = relative_z;

  StateVec q_curr = VectorXd::Zero((int)NUM_Q);
  StateVec qd_curr = VectorXd::Zero((int)NUM_Q);
  InputVec q_cmd = VectorXd::Zero((int)NUM_U);
  InputVec qd_cmd = VectorXd::Zero((int)NUM_U);
  bool finished = true;
  bool contacts[4] = {1,1,1,1};

  data.t += CTRL_LOOP_DURATION;

  if (data.t > dwell_time+step_time) {
    if (!data.planning) {
      compute_foot_pos(q, data.foot_pos_init);
      data.global_z_offset = q_in[Q_Z] - relative_z;
      data.plan_future = std::async(std::launch::async, &walk_quasi_planned::WalkQuasiPlanned::plan_steps, this,  q, args, data.prev_foot);
      data.planning = true;
    }

    if (data.planning && data.plan_future.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready)
    {
      data.steps =  data.plan_future.get();
      if (data.steps.size() > 0) {
        compute_com_traj(q, qd);
        compute_foot_traj(q, qd, data.steps.at(0));
        data.prev_foot = data.steps.at(0).swing_foot;
        data.t = 0;
        data.step_num++;
        data.planning = false;
      }
    }
  }

  double com_pos[3];
  double com_pos_des[3];
  fk_com(q, com_pos);

  for (int i = 0; i < 2; ++i) {
    traj_gen::cubic_poly_eval(data.traj[i], data.t, com_pos_des[i], qd_curr[i+Q_X]);
    q_curr[i+Q_X] = com_pos_des[i] - com_pos[i] + q[i];
  }
  for (int i = 2; i < 6; ++i) {
    traj_gen::cubic_poly_eval(data.traj[i], data.t, q_curr[i+Q_X], qd_curr[i+Q_X]);
  }

  if (q_curr[Q_RZ] > M_PI) {
    q_curr[Q_RZ] -= 2*M_PI;
  }
  if (q_curr[Q_RZ] < -M_PI) {
    q_curr[Q_RZ] += 2*M_PI;
  }

  if (data.t < dwell_time || data.planning) {
    for (int i=0; i<4;++i) {
      ik((Frame)(i+F_FL_EE), q_curr, data.foot_pos_init[i], &q_cmd[3*i]);
    }
  } else if (data.t >= dwell_time && data.t <= dwell_time+step_time) {
    double foot_pos[3] = {};
    double foot_vel[3] = {};
    for (int i=0; i<2;++i) {
      finished &= traj_gen::cubic_poly_eval(data.swing_traj[i],
                                            data.t-dwell_time,
                                            foot_pos[i],
                                            foot_vel[i]);
    }
    if (data.t - dwell_time < step_time/2) {
      finished &= traj_gen::cubic_poly_eval(data.swing_traj[2],
                                            data.t-dwell_time,
                                            foot_pos[2],
                                            foot_vel[2]);
    } else {
      finished &= traj_gen::cubic_poly_eval(data.swing_traj[3],
                                            data.t-dwell_time-step_time/2,
                                            foot_pos[2],
                                            foot_vel[2]);
    }
    for (int i=0; i<4;++i) {
      if (i == data.steps.at(0).swing_foot) {
        ik((Frame)(i+F_FL_EE), q_curr, foot_pos, &q_cmd[3*i]);
        contacts[i] = 0;
      } else {
        ik((Frame)(i+F_FL_EE), q_curr, data.foot_pos_init[i], &q_cmd[3*i]);
      }
    }
  }

  // compute gravity offset
  InputVec grav = gravity_offset(q, com_pos, data.foot_pos_init, contacts);

  for (int i = 0; i < NUM_U; ++i) {
    act_cmds.mode[i] = CMD_MODE_TAU_VEL_POS;
    act_cmds.u[i] = grav[i];
    act_cmds.q[i] = q_cmd[i];
    act_cmds.qd[i] = 0;
    act_cmds.kp[i] = 500;
    act_cmds.kd[i] = 5;
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
  return MatrixXd::Zero((int)NUM_Q+NUM_Q, NUM_CONT_ARGS+1);
}

bool in_sroa_underestimate(const StateVec &q,
                           const StateVec &qd,
                           const ContactState c_s,
                           const Args &args,
                           const double delta_t) const {
  //if (!in_safe_set(q,qd,c_s,args,delta_t)) {
  //  return false;
  //}
  double relative_z = 0;
  bool relative_z_valid = contact_feet_relative_z(q, c_s, relative_z);
  if (fabs(relative_z - args.cont[ARG_H]) < 0.05) {
    return true;
  }
  return false;
}
bool in_sroa_overestimate(const StateVec &q,
                          const StateVec &qd,
                          const ContactState c_s,
                          const Args &args,
                          const double delta_t) const {
  //if (!in_safe_set(q,qd,c_s,args,delta_t)) {
  //  return false;
  //}
  double relative_z = 0;
  bool relative_z_valid = contact_feet_relative_z(q, c_s, relative_z);
  if (fabs(relative_z - args.cont[ARG_H]) < 0.075) {
    return true;
  }
  return false;
}
bool in_safe_set(const StateVec &q,
                 const StateVec &qd,
                 const ContactState c_s,
                 const Args &args,
                 const double delta_t) const {
  if (!(in_vel_limits(qd) && in_pos_limits(q))) {
    return false;
  }

  // at least 2 feet should be in contact
  if (c_s[0] + c_s[3] + c_s[1] + c_s[2] >= 2) {
    return true;
  } else {
    return false;
  }
}

private:

InputVec gravity_offset(const StateVec &q,
                        const double com_pos[3], 
                        const double foot_pos[4][3], 
                        const bool contacts[4]) {
  StateVec F = VectorXd::Zero(NUM_Q);
  
  int num_contacts = 0;
  for (int i = 0; i < 4; ++i) {
    if (contacts[i]) num_contacts++;
  }

  /* A*w = c
     where columns of A are feet x,y,1
     w is the weight proportion on each foot
     c is com_x, com_y, 1
     last row of 1s in A and c enforce |w| = 1
  */

  MatrixXd A(3, num_contacts);
  VectorXd c(3);
  c << com_pos[0], com_pos[1], 1;

  int idx = 0;
  for (int i = 0; i < 4; ++i) {
    if (contacts[i]) {
      A(0,idx) = foot_pos[i][0];
      A(1,idx) = foot_pos[i][1];
      A(2,idx) = 1;
      idx++;
    }
  }

  /* using right pseudoinverse */
  VectorXd w = A.transpose()*(A*A.transpose()).llt().solve(c);

  Vector3d contact_force = Vector3d::Zero();
  idx = 0;
  for (int i = 0; i < 4; ++i) {
    if (contacts[i]) {
      MatrixXd jac = fk_jac(q, robot.contacts[i].frame);
      contact_force[2] = -w[idx]*robot.pin_data_pool[0].mass[0]*9.81;
      F += jac.transpose()*contact_force;
      idx++;
    }
  }

  return F.tail(NUM_U);

}

void compute_com_traj(const StateVec &q,
                      const StateVec &qd) {
  double q_des[6] = {data.steps.at(0).com_pos[0],
                     data.steps.at(0).com_pos[1],
                     data.steps.at(0).com_pos[2],
                     0,
                     0,
                     data.steps.at(0).heading};
                     //args.cont[ARG_RZ] + rz};

  double duration = 1;
  double com_pos[3];
  double com_vel[3] = {qd[Q_X], qd[Q_Y], qd[Q_Z]}; //TODO implement com jacobian to get actual com velocity
  fk_com(q, com_pos);
  for (int i = 0; i < 2; ++i) {
    data.traj[i] = traj_gen::cubic_poly_gen(com_pos[i+Q_X],com_vel[i+Q_X],q_des[i],0,duration);
  }
  data.traj[Q_Z] = traj_gen::cubic_poly_gen(q[Q_Z],qd[Q_Z],q_des[2],0,duration);
  for (int i = 3; i < 6; ++i) {
    q_des[i] = q[i+Q_X] + min_angle_diff(q_des[i],q[i+Q_X]);
    data.traj[i] = traj_gen::cubic_poly_gen(q[i+Q_X],qd[i+Q_X],q_des[i],0,duration);
  }
}

void compute_foot_traj(const StateVec &q,
                       const StateVec &qd,
                       Step &step) {
  for (int i = 0; i < 2; ++i) {
    data.swing_traj[i] = traj_gen::cubic_poly_gen(step.foot_start_pos[step.swing_foot][i], 0,
                                                  step.foot_end_pos[step.swing_foot][i],   0,
                                                  step_time);
  }
  data.swing_traj[2] = traj_gen::cubic_poly_gen(step.foot_start_pos[step.swing_foot][2], 0,
                                                step.foot_start_pos[step.swing_foot][2]+step_height, 0,
                                                step_time/2.0);
  data.swing_traj[3] = traj_gen::cubic_poly_gen(step.foot_start_pos[step.swing_foot][2]+step_height, 0,
                                                step.foot_end_pos[step.swing_foot][2], 0,
                                                step_time/2.0);
}

void compute_goal_pos(const double centroid_init[2],
                      const double heading_init,
                      const Args &args,
                      const int step_num,
                      double centroid_goal_pos[2],
                      double &heading_goal_pos) {
  if (args.disc[ARG_MODE] == 1) {
    centroid_goal_pos[0] = centroid_init[0] +
                           fmax(fmin(args.cont[ARG_X] - centroid_init[0],
                                     fabs(args.cont[ARG_VX]*(dwell_time+step_time)*(step_num))),
                                    -fabs(args.cont[ARG_VX]*(dwell_time+step_time)*(step_num)));
    centroid_goal_pos[1] = centroid_init[1] +
                           fmax(fmin(args.cont[ARG_Y] - centroid_init[1],
                                     fabs(args.cont[ARG_VY]*(dwell_time+step_time)*(step_num))),
                                    -fabs(args.cont[ARG_VY]*(dwell_time+step_time)*(step_num)));
    heading_goal_pos = heading_init +
                       fmax(fmin(args.cont[ARG_RZ] - heading_init,
                                 fabs(args.cont[ARG_VRZ]*(dwell_time+step_time)*(step_num))),
                                -fabs(args.cont[ARG_VRZ]*(dwell_time+step_time)*(step_num)));
  } else {
    centroid_goal_pos[0] = centroid_init[0] + 
                           args.cont[ARG_VX]*(dwell_time+step_time)*(step_num);
    centroid_goal_pos[1] = centroid_init[1] + 
                           args.cont[ARG_VY]*(dwell_time+step_time)*(step_num);
    heading_goal_pos = heading_init + 
                       args.cont[ARG_VRZ]*(dwell_time+step_time)*(step_num);
  }
}

std::deque<Step> plan_steps(const StateVec &q,
                            const Args &args,
                            const Contact foot_prev) {
  std::deque<Step> best_plan;
  std::vector<Node> nodes;

  int cheapest_node_idx = 0;
  double cheapest_node_cost = 1e9;
  int cheapest_node_at_depth_idx = 0;
  double cheapest_node_at_depth_cost = 1e9;

  int depth = 0;
  int max_depth = 5;

  int max_depth_node_idx = 0;


  Step dummy_start_step = {};
  dummy_start_step.q_end = q;
  dummy_start_step.swing_foot = foot_prev;
  compute_foot_pos(q, dummy_start_step.foot_end_pos);

  Node start_node = {};
  start_node.prnt_idx = -1;
  start_node.cost = 0;
  start_node.step = dummy_start_step;
  start_node.step_num = data.step_num;
  start_node.depth = 0;

  /* mark double stepping with the same foot as infeasible */
  if (foot_prev >=0 && foot_prev < 4) {
    start_node.infeasible[0][foot_prev] = true;
    start_node.infeasible[1][foot_prev] = true;
  }

  nodes.push_back(start_node);

  size_t base_node_idx = 0;
  int iter = 0;
  while (base_node_idx < nodes.size()) {
    if (nodes.at(base_node_idx).depth >= max_depth) {
      base_node_idx++;
      continue;
    }

    Node samp_node;
    samp_node.prnt_idx = base_node_idx;
    samp_node.step_num = nodes.at(base_node_idx).step_num+1;
    
    double centroid_goal_pos[2];
    double heading_goal_pos;
    compute_goal_pos(data.centroid_init,
                     data.heading_init[2],
                     args,
                     samp_node.step_num,
                     centroid_goal_pos,
                     heading_goal_pos);

    for (int swing_foot = 0; swing_foot < 4; ++swing_foot) {
      if (swing_foot == nodes.at(base_node_idx).step.swing_foot) {
        continue;
      }
      if (!nominal_step_sample(nodes.at(base_node_idx).step.q_end,
                               nodes.at(base_node_idx).step.foot_end_pos,
                               centroid_goal_pos,
                               heading_goal_pos,
                               (Contact) swing_foot,
                               samp_node.step)) {
        continue;
      }
      samp_node.cost = nodes.at(base_node_idx).cost + samp_node.step.cost;
      samp_node.depth = nodes.at(base_node_idx).depth + 1;
      if (samp_node.cost < cheapest_node_cost) {
        cheapest_node_idx = nodes.size();
        cheapest_node_cost = samp_node.cost;
      }
      if (depth < samp_node.depth) {
        max_depth_node_idx = nodes.size();
        depth = samp_node.depth;
      }
      if (samp_node.depth == max_depth && samp_node.cost < cheapest_node_at_depth_cost) {
        cheapest_node_at_depth_idx = nodes.size();
        cheapest_node_at_depth_cost = samp_node.cost;
      }
      nodes.push_back(samp_node);
    }
    base_node_idx++;
  }

  int path_node_idx = cheapest_node_at_depth_idx;
  while (nodes.at(path_node_idx).prnt_idx != -1) {
    best_plan.push_front(nodes.at(path_node_idx).step);
    path_node_idx = nodes.at(path_node_idx).prnt_idx;
  }

  return best_plan;
}

bool nominal_step_sample(const StateVec &q,
                         double foot_start_pos[4][3],
                         const double centroid_goal_pos[2],
                         const double goal_heading,
                         const Contact swing_foot,
                         Step &step) {
  memcpy(step.foot_start_pos, foot_start_pos, 12*sizeof(double));
  memcpy(step.foot_end_pos, foot_start_pos, 12*sizeof(double));

  step.heading = goal_heading;
  step.swing_foot = swing_foot;
  com_over_support_polygon_nominal(step.foot_start_pos,
                                   step.swing_foot,
                                   step.com_pos);
  step.q_start = compute_q_step(step.foot_start_pos,
                                step.com_pos,
                                step.heading);
  if (!check_pose_feasibility(step.q_start,
                              step.foot_start_pos,
                              step.com_pos,
                              step.heading,
                              step.swing_foot)) {
    return false;
  }
  compute_nominal_foot_step(step.swing_foot,
                            step.q_start,
                            centroid_goal_pos,
                            goal_heading,
                            step.foot_end_pos[step.swing_foot]);
  double neutral_pos[3] = {};
  compute_foot_neutral_pos(step.swing_foot,
                           step.q_start,
                           neutral_pos);
  descend_foot_pos(step, neutral_pos);
  step.q_end = compute_q_step(step.foot_end_pos,
                              step.com_pos,
                              step.heading);
  if (!check_footfall_feasibility(step.q_end,
                                  step.foot_end_pos,
                                  step.com_pos,
                                  step.heading,
                                  step.swing_foot)) {
    return false;
  }

  step.cost = step_cost(step, centroid_goal_pos, goal_heading);
  return true;

}

void descend_foot_pos(Step &step, double neutral_pos[3]) {
  double k = 0.02;
  int descent_cnt = 0;
  step.q_end = compute_q_step(step.foot_end_pos,
                              step.com_pos,
                              step.heading);

  double foot_end_pos[4][3];
  memcpy(foot_end_pos, step.foot_end_pos, 12*sizeof(double));

  for (double r=0; r<0.3; r+=0.01) {
    for (double th=0;th<2*M_PI;th+=0.01/(r+0.1)) {
      step.q_end = compute_q_step(foot_end_pos,
                                  step.com_pos,
                                  step.heading);
      if (!check_footfall_feasibility(step.q_end,
                                      foot_end_pos,
                                      step.com_pos,
                                      step.heading,
                                      step.swing_foot)) {
        foot_end_pos[step.swing_foot][0] = step.foot_end_pos[step.swing_foot][0] + r*cos(th);
        foot_end_pos[step.swing_foot][1] = step.foot_end_pos[step.swing_foot][1] + r*sin(th);
      } else {
        memcpy(step.foot_end_pos, foot_end_pos, 12*sizeof(double));
        return;
      }
    }
  }
}

double step_cost(const Step &step,
                 const double centroid_goal_pos[2],
                 const double goal_heading) {
  double centroid_end[2] = {};
  for (int i=0; i<4; ++i) {
    centroid_end[0] += step.foot_end_pos[i][0]/4.0;
    centroid_end[1] += step.foot_end_pos[i][1]/4.0;
  }

  double cost = (step.heading-goal_heading)*(step.heading-goal_heading);

  cost += (centroid_end[0]-centroid_goal_pos[0])*(centroid_end[0]-centroid_goal_pos[0]) +
          (centroid_end[1]-centroid_goal_pos[1])*(centroid_end[1]-centroid_goal_pos[1]);
  for (int i=0; i<4;++i) {
    double nominal_footfall[3] = {};
    compute_nominal_foot_step((Contact) i,
                              step.q_end,
                              centroid_goal_pos,
                              goal_heading,
                              nominal_footfall);

    cost += (nominal_footfall[0]-step.foot_end_pos[i][0])*
            (nominal_footfall[0]-step.foot_end_pos[i][0])+
            (nominal_footfall[1]-step.foot_end_pos[i][1])*
            (nominal_footfall[1]-step.foot_end_pos[i][1]);
  }
  return cost;
}

void compute_foot_neutral_pos(const Contact foot,
                              const StateVec &q,
                              double neutral_pos[3]) {
    StateVec q_neutral = VectorXd::Zero(NUM_Q);
    Frame neutral_frame;
    switch(robot.contacts[foot].frame) {
      case F_FL_EE:
        neutral_frame = F_Q_FL2;
        break;
      case F_FR_EE:
        neutral_frame = F_Q_FR2;
        break;
      case F_BL_EE:
        neutral_frame = F_Q_BL2;
        break;
      case F_BR_EE:
        neutral_frame = F_Q_BR2;
        break;
      default:
        neutral_frame = robot.contacts[foot].frame;
        break;
    }
    q_neutral << q.head(3),0,0,q[Q_RZ],
                  0.1,0,0,
                 -0.1,0,0,
                  0.1,0,0,
                 -0.1,0,0;
    fk(q_neutral, neutral_pos, robot.contacts[foot].frame);
    neutral_pos[2] = 0;
}

void compute_foot_pos(const StateVec &q,
                      double foot_pos[4][3]) {
  for (int i = 0; i < 4; ++i) {
    fk(q, foot_pos[i], (Frame)(i+F_FL_EE));
  }

}

void compute_nominal_foot_step(Contact swing_foot,
                               const StateVec &q,
                               const double centroid_goal_pos[2],
                               const double goal_heading,
                               double nominal_foot_step[3]) {
  double r = 0.25;
  double th = 0;
  switch(swing_foot) {
    case C_FL_EE:
      th = M_PI/6;
      break;
    case C_FR_EE:
      th = -M_PI/6;
      break;
    case C_BL_EE:
      th = 5*M_PI/6;
      break;
    case C_BR_EE:
      th = -5*M_PI/6;
      break;
    default:
      break;
  }

  double neutral_pos[3] = {};
  compute_foot_neutral_pos(swing_foot,
                           q,
                           neutral_pos);

  nominal_foot_step[0] = centroid_goal_pos[0] + r*cos(th+goal_heading);
  nominal_foot_step[1] = centroid_goal_pos[1] + r*sin(th+goal_heading);
}


StateVec compute_q_step(const double foot_pos[4][3],
                        const double com_pose[3],
                        const double heading) {
  // TODO assuming CoM lies at origin for now
  StateVec q = VectorXd::Zero(NUM_Q);
  q[Q_X] = com_pose[0];
  q[Q_Y] = com_pose[1];
  q[Q_Z] = com_pose[2];
  q[Q_RZ] = heading;
  double joint_des[3];
  for (int foot = 0; foot < 4; ++foot) {
    ik(robot.contacts[foot].frame, q, foot_pos[foot], joint_des);
    q[Q_FL1+3*foot] = joint_des[0];
    q[Q_FL1+3*foot+1] = joint_des[1];
    q[Q_FL1+3*foot+2] = joint_des[2];
  }

  return q;
}

bool check_pose_feasibility(const StateVec &q,
                            const double foot_pos[4][3],
                            const double com_pose[3],
                            const double heading,
                            Contact swing_foot) {
  if (!in_pos_limits(q)) {
    return false;
  }

  /* minimum distance to edge of support polygon */
  for (int i = 0; i < 4; ++i) {
    if (i == swing_foot) continue;
    for (int j = i+1; j < 4; ++j) {
      if (j == swing_foot) continue;
      double dist = point_to_line_dist(com_pose[0],
                                       com_pose[1],
                                       foot_pos[i][0],
                                       foot_pos[i][1],
                                       foot_pos[j][0],
                                       foot_pos[j][1]);
      if (dist < 0.02) {
        return false;
      }
    }
  }
  return true;
}

bool valid_foot_placement(const double foot_pos[4][3], Contact swing_foot) {
  Vector3d ee_pos_vec;
  ee_pos_vec << foot_pos[swing_foot][0],
                foot_pos[swing_foot][1],
                foot_pos[swing_foot][2] + data.global_z_offset;
  Vector3d normal;
  Vector3d contact_pos;
  double r = 0.01; // margin to check circle around contact point. TODO maybe there is a better way...
  for (double th = 0; th < 2*M_PI + 2*M_PI/4.; th += 2*M_PI/4.) {
    /* this returns the closest terrain (in z) surface point as contact_pos*/
    terrain_map(ee_pos_vec,
                contact_pos,
                normal);
    if (fabs(ee_pos_vec[0] - contact_pos[0]) > 1e-6 || // allow only for float precision error in x
        fabs(ee_pos_vec[1] - contact_pos[1]) > 1e-6 || // allow only for float precision error in x
        fabs(ee_pos_vec[2] - contact_pos[2]) > 0.1) {  // allow for some error in z
      return false;
    }
    ee_pos_vec[0] = foot_pos[swing_foot][0] + r*cos(th);
    ee_pos_vec[1] = foot_pos[swing_foot][1] + r*sin(th);
  }
  return true;
}

bool check_footfall_feasibility(const StateVec &q,
                                const double foot_pos[4][3],
                                const double com_pose[3],
                                const double heading,
                                Contact swing_foot) {

  if (!valid_foot_placement(foot_pos, swing_foot)) {
    return false;
  }

  if (!check_pose_feasibility(q, foot_pos, com_pose, heading, swing_foot)) {
    return false;
  }

  /* feet shouldn't be too close */
  double margin = 0.1;
  for (int i=0;i<4;++i) {
    if (i != swing_foot) {
      if ((foot_pos[i][0]-foot_pos[swing_foot][0])*(foot_pos[i][0]-foot_pos[swing_foot][0]) +
          (foot_pos[i][1]-foot_pos[swing_foot][1])*(foot_pos[i][1]-foot_pos[swing_foot][1]) <
          margin*margin) {
        return false;
      }
    }
  }

  return true;

}

double point_to_line_dist(double px,
                          double py,
                          double lx1,
                          double ly1,
                          double lx2,
                          double ly2) {
  return fabs((lx2-lx1)*(ly1-py) - (lx1-px)*(ly2-ly1))/
           sqrt((lx2-lx1)*(lx2-lx1) + (ly2-ly1)*(ly2-ly1));
}

void com_over_support_polygon_nominal(double foot_pos[4][3],
                                      Contact swing_foot,
                                      double com_sample[3]) {

  double margin = 0.2;//333;
  double weights[4] = {};
  double opposite_weight = margin;
  double neighbor_weight = (1-margin)/2.;
  int opposite_foot[4] = {3, 2, 1, 0};

  for (int i=0; i<4; ++i) {
    if (i != swing_foot) {
      if (i == opposite_foot[swing_foot]) {
        weights[i] = opposite_weight;
      } else {
        weights[i] = neighbor_weight;
      }
    }
  }

  com_sample[0] = 0;
  com_sample[1] = 0;
  com_sample[2] = 0;
  for (int i = 0; i < 4; ++i) {
    if (i != swing_foot) {
      com_sample[0] += weights[i]*foot_pos[i][0];
      com_sample[1] += weights[i]*foot_pos[i][1];
      com_sample[2] += weights[i]*foot_pos[i][2];
    }
  }
  com_sample[2] = 0.22;
}

};

PrimitiveBehavior* create() {
  return new WalkQuasiPlanned;
}
void destroy(PrimitiveBehavior* prim) {
  delete prim;
}

}
