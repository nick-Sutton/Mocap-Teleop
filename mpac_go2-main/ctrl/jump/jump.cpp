#include "robot.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "math.h"

#include "math_utils.h"
#include "ctrl_core.h"
#include "ctrl/ctrl_utils/joint_pos_pd.h"
#include "ctrl/ctrl_utils/traj_gen.h"
#include "ik.h"
#include "fk.h"
#include "jump.h"
#include "jump_args.h"

#include <fstream>

extern Robot robot;

namespace jump {

typedef struct {
  double t;
  double t_delay;
  bool takeoff;
  double foot_pos_init[4][3];
  traj_gen::CubicPoly traj[6]; //x,y,z,rx,ry,rz
  std::ofstream outFile;
  
} Data;

class Jump: public PrimitiveBehavior {

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
  data.t = 0;
  data.t_delay = 0;
  data.takeoff = false;

  double centroid[2] = {};
  double relative_z = 0;
  bool relative_z_valid = contact_feet_relative_z(q_in, c_s, relative_z);
  StateVec q = q_in;
  q[Q_Z] = relative_z;

  for (int i = 0; i < 4; ++i) {
    fk(q, data.foot_pos_init[i], (Frame)(i+F_FL_EE));
    //foot_pos_init[i][2] =0;
    centroid[0] += 0.25*data.foot_pos_init[i][0];
    centroid[1] += 0.25*data.foot_pos_init[i][1];
  }

  double fx = 0.5*(data.foot_pos_init[0][0] + data.foot_pos_init[1][0]);
  double fy = 0.5*(data.foot_pos_init[0][1] + data.foot_pos_init[1][1]);
  double bx = 0.5*(data.foot_pos_init[2][0] + data.foot_pos_init[3][0]);
  double by = 0.5*(data.foot_pos_init[2][1] + data.foot_pos_init[3][1]);
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
  double rz = atan2(fy-by, fx-bx);

  double duration = 0.5;

  double q_des[6] = {centroid[0] + args.cont[ARG_X_VEL]*duration, //TODO pull actual CoM
                    //  centroid[0]+0.012731 + args.cont[ARG_X_VEL]*duration,
                     centroid[1] +  args.cont[ARG_Y_VEL]*duration,
                     0.25,
                     0,
                     0,
                     q[Q_RZ]};
  double qd_des[6] = {args.cont[ARG_X_VEL],
                      args.cont[ARG_Y_VEL],
                      args.cont[ARG_Z_VEL],
                      0,
                      0,
                      0};

  for (int i = 0; i < 3; ++i) {
    data.traj[i] = traj_gen::cubic_poly_gen(q[i+Q_X],qd[i+Q_X],q_des[i],0,duration);
    if (i+Q_X == Q_Z) {
      data.traj[i] = traj_gen::cubic_poly_gen(q[i+Q_X],qd[i+Q_X],q_des[i],qd_des[i],duration+0.1);
    }
  }
  for (int i = 3; i < 6; ++i) {
    q_des[i] = q[i+Q_X] + min_angle_diff(q_des[i],q[i+Q_X]);
    data.traj[i] = traj_gen::cubic_poly_gen(q[i+Q_X],qd[i+Q_X],q_des[i],0,duration);
  }

  data.outFile.open("/home/jlee267/mpac/mpac_a1/data/jump_traj.txt",std::ios::app);
}

void execute(const StateVec &q,
             const StateVec &qd,
             const ContactState c_s,
             const Args &args, 
             ActuatorCmds &act_cmds) {
  StateVec q_curr = VectorXd::Zero((int)NUM_Q);
  StateVec qd_curr = VectorXd::Zero((int)NUM_Q);
  InputVec q_cmd = VectorXd::Zero((int)NUM_U);
  InputVec qd_cmd = VectorXd::Zero((int)NUM_U);
  bool finished = true;

  data.t += CTRL_LOOP_DURATION;
  for (int i = 0; i < 6; ++i) {
    finished &= traj_gen::cubic_poly_eval(data.traj[i], data.t, q_curr[i+Q_X], qd_curr[i+Q_X]);
  }

  if (data.outFile.is_open()) {
      // Iterate through the vector and write each value to the file
      for (size_t i = 0; i < 7; ++i) {
        if (i ==0){
          data.outFile << data.t;
        }
        else{
          data.outFile << q_curr[i-1];
        }
          if (i != 6) {
              data.outFile << ", ";  // Add a comma after each value except the last one
          }
        }
       data.outFile << "\n";  
  }

  // std::cout<<"traj[x]: "<<q_curr[0] <<std::endl;
  // std::cout<<"traj[y]: "<<q_curr[1] <<std::endl;
  // std::cout<<"traj[z]: "<<q_curr[2] <<std::endl;

  if (!data.takeoff && finished) {
    if (c_s[0] || c_s[1] || c_s[2] || c_s[3]) {
      data.takeoff = true;
    }
  }

  if (!data.takeoff) {
    for (int i=0; i<4;++i) {
      ik((Frame)(i+F_FL_EE), q_curr, data.foot_pos_init[i], &q_cmd[3*i]);
    }

    MatrixXd J = MatrixXd::Zero(6*NUM_C, (int)NUM_Q);
    for (int i = 0; i < NUM_C; ++i) {
      J.block(6*i,0,6,(int)NUM_Q) << fk_jac_full(q,robot.contacts[i].frame);
    }

    VectorXd qd_curr_stacked = VectorXd::Zero(30);
    qd_curr_stacked << VectorXd::Zero(24), qd_curr.head(6);
    //J = J.block(0,6,24,(int)NUM_U);
    MatrixXd A = MatrixXd::Zero(6*NUM_C+6, (int)NUM_Q);
    A << J, MatrixXd::Identity(6, 6), MatrixXd::Zero(6, (int)NUM_Q - 6);
    //std::cout << J.rows() << J.cols() << " qd " <<qd_curr_stacked.rows() << qd_curr_stacked.cols() <<std::endl;

    //std::cout << A.rows() << A.cols() << " qd " <<qd_curr_stacked.rows() << qd_curr_stacked.cols() <<std::endl;
    //std::cout << std::fixed << std::setprecision(3) << A << std::endl;
    //std::cout << qd_curr_stacked << std::endl;
    //InputVec qd_cmd2 = (J.transpose()*J).llt().solve(J.transpose()*qd_curr_stacked);
    // Least squares error to achieve body velocity
    StateVec b = (A.transpose()*A).llt().solve(A.transpose()*qd_curr_stacked);
    //StateVec b = A.transpose()*(A*A.transpose()).llt().solve(qd_curr_stacked);
    qd_cmd = b.tail(NUM_U);

    //std::cout << "res: " << A*b - qd_curr_stacked << std::endl;
    //std::cout << "res2: " << b << std::endl;
    //std::cout << "qd_cmd" << J*qd_cmd2 - qd_curr_stacked << std::endl;
    //std::cout << "qd_first" << qd_curr(0) - qd_cmd2(0) << std::endl;

    double com_pos[3];
    fk_com(q, com_pos);
    InputVec grav = gravity_offset(q, com_pos, data.foot_pos_init, c_s);

    for (int i = 0; i < NUM_U; ++i) {
      act_cmds.mode[i] = CMD_MODE_TAU_VEL_POS;
      act_cmds.u[i] = grav[i];
      act_cmds.q[i] = q_cmd[i];
      act_cmds.qd[i] = qd_cmd[i];
      act_cmds.kp[i] = 200;
      act_cmds.kd[i] = 5;
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
  //TODO better handling of "don't care" for q_des
  q_des = q;
  qd_des = qd;//VectorXd::Zero((int)NUM_Q);
  memcpy(&c_s_des, c_s, sizeof(ContactState));
}

MatrixXd setpoint_jac(const StateVec &q,
                           const StateVec &qd,
                           const ContactState c_s,
                           const Args &args,
                           const double delta_t) const {
  //TODO fill in
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
  //if (fabs(q[Q_Z] - 0.25) < 0.05) {
  double relative_z = 0;
  bool relative_z_valid = contact_feet_relative_z(q, c_s, relative_z);
  if (relative_z > 0.05) {
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
  //if (fabs(q[Q_Z] - 0.25) < 0.05) {
  double relative_z = 0;
  bool relative_z_valid = contact_feet_relative_z(q, c_s, relative_z);
  if (relative_z > 0.05) {
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
  
  // feet should be in contact
  if (c_s[0] && c_s[1] && c_s[2] && c_s[3]) {
    return true;
  } else {
    return false;
  }
}

void set_next_ctrl_des(const StateVec &q, const StateVec &qd, const ContactState c_s,
                       CtrlState &ctrl_des) {
  if (data.takeoff) { //TODO read foot sensors
    ctrl_des.mode = C_LAND;
  }
}

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

};

PrimitiveBehavior* create() {
  return new Jump;
}
void destroy(PrimitiveBehavior* prim) {
  delete prim;
}

}
