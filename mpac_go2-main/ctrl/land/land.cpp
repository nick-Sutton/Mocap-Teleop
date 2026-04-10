#include "robot.h"
#include <eigen3/Eigen/Dense>
#include <iostream>
#include "math.h"

#include "ctrl_core.h"
#include "land.h"
#include "full_model_fd.h"
#include "fk.h"
#include "ik.h"
#include "ctrl/ctrl_utils/traj_gen.h"
#include "ctrl_mode_class.h"

extern Robot robot;

typedef struct {
  int contact_cnt;
  InputVec ground_joints;
  Vector3d qd_filtered;
  
} Data;

class Land: public PrimitiveBehavior {

Data data;

void compute_aerial_foot_pos_des(const StateVec &q, const StateVec &qd, double foot_pos_des[4][3]);
void compute_ground_foot_pos_des(const StateVec &q, double foot_pos_des[4][3]);
void compute_ground_joint_des(const StateVec &q);
VectorXd compute_foot_torques(Contact foot, const StateVec &q, const StateVec &qd, const ContactState c_s, double foot_pos_des[3]);


public:
void init(const StateVec &q,
          const StateVec &qd,
          const ContactState c_s,
          const Args &args) { 
  data.contact_cnt = 0;
  compute_ground_joint_des(data, q);
  data.qd_filtered = Vector3d::Zero();
}

void execute(const StateVec &q,
             const StateVec &qd,
             const ContactState c_s,
             const Args &args, 
             ActuatorCmds &act_cmds) {
  double foot_pos_des[4][3] = {};
  StateVec q_des = q;

  data.qd_filtered = data.qd_filtered + 0.2*(qd.head(3)-data.qd_filtered);

  if (c_s[0] && c_s[1] && c_s[2] && c_s[3]) {
    if (data.contact_cnt <= 11) {
      data.contact_cnt++;
    }
  } else {
    if (data.contact_cnt > 0) {
      data.contact_cnt--;
    }
  }
  if (data.contact_cnt == 10) {
    compute_ground_joint_des(data, q);
  }
  if (data.contact_cnt > 10) {
    compute_ground_foot_pos_des(data, q, foot_pos_des);
  } else {
    compute_aerial_foot_pos_des(data, q, qd, foot_pos_des);
  }

  InputVec u;
  for (int i=0; i<4;++i) {
    u.segment(3*i,3) = compute_foot_torques(data, (Contact) i, q, qd, c_s, foot_pos_des[i]);
  }
  for (int i = 0; i < NUM_U; ++i) {
    act_cmds.mode[i] = CMD_MODE_TAU_VEL;
    act_cmds.u[i] = u[i];
    act_cmds.qd[i] = 0;
    act_cmds.kd[i] = 2;
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
  q_des << q(Q_X), q(Q_Y), 0.05,
           0,    0, q(Q_RZ),
           0, 1.57, -2.5,
           0, 1.57, -2.5,
           0, 1.57, -2.5,
           0, 1.57, -2.5;
  qd_des = VectorXd::Zero(NUM_Q);
  memset(&c_s_des, true, sizeof(ContactState));
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
  return true;
}
bool in_sroa_overestimate(const StateVec &q,
                          const StateVec &qd,
                          const ContactState c_s,
                          const Args &args,
                          const double delta_t) const {
  return true;
}
bool in_safe_set(const StateVec &q,
                 const StateVec &qd,
                 const ContactState c_s,
                 const Args &args,
                 const double delta_t) const {
  if (!(in_vel_limits(qd) && in_pos_limits(q))) {
    return false;
  }
  return true;
}

private:
void compute_aerial_foot_pos_des(Data &data, const StateVec &q, const StateVec &qd, double foot_pos_des[4][3]) {
  Matrix3d R_curr;
  R_curr = AngleAxisd(q(Q_RX), Vector3d::UnitX()) *
           AngleAxisd(q(Q_RY), Vector3d::UnitY()) *
           AngleAxisd(q(Q_RZ), Vector3d::UnitZ()); 

  Vector2d x_proj = R_curr.block(0,0,2,1);
  double z_ang = acos(R_curr.col(2).transpose()*Vector3d::UnitZ());

  double heading;
  heading = atan2(x_proj[1],x_proj[0]);

  StateVec fallback_q;
  fallback_q << q.head(6),
                0, 1.57, -2.5,
                0, 1.57, -2.5,
                0, 1.57, -2.5,
                0, 1.57, -2.5;
  double foot_pos_fallback[4][3] = {};
  for (int i = 0; i < 4; ++i) {
    fk(fallback_q, foot_pos_fallback[i], (Frame)(i+F_FL_EE));
  }

  Vector3d vel_vec = 0.1*data.qd_filtered;

  if (fabs(vel_vec[Q_X]) > 0.2) {
    vel_vec *= 0.2/fabs(vel_vec[Q_X]);
  }
  if (fabs(vel_vec[Q_Y]) > 0.15) {
    vel_vec *= 0.15/fabs(vel_vec[Q_X]);
  }
  if (vel_vec[Q_Z] < -0.2) {
    vel_vec *= 0.2/fabs(vel_vec[Q_Z]);
  }
  if (vel_vec[Q_Z] > 0) {
    vel_vec *= 0;
  }

  double k_vel = 0.1;
  double x_center = vel_vec[Q_X];
  double y_center = vel_vec[Q_Y];
  double z_center = vel_vec[Q_Z];

  double xf = +0.2 + x_center;
  double xb = -0.2 + x_center;
  double yl = +0.15 + y_center;
  double yr = -0.15 + y_center;
  double z  = 0.2 - z_center;
                 
  double foot_pos_nominal[4][3] = {{q[Q_X] + xf*cos(heading) - yl*sin(heading),
                                    q[Q_Y] + xf*sin(heading) + yl*cos(heading),
                                    q[Q_Z]-z},
                                   {q[Q_X] + xf*cos(heading) - yr*sin(heading),
                                    q[Q_Y] + xf*sin(heading) + yr*cos(heading),
                                    q[Q_Z]-z},
                                   {q[Q_X] + xb*cos(heading) - yl*sin(heading),
                                    q[Q_Y] + xb*sin(heading) + yl*cos(heading),
                                    q[Q_Z]-z},
                                   {q[Q_X] + xb*cos(heading) - yr*sin(heading),
                                    q[Q_Y] + xb*sin(heading) + yr*cos(heading),
                                    q[Q_Z]-z},
                                  };

  double gamma = 0;
  if (fabs(z_ang) < 0.5) {
    gamma = 1;
  } else if (fabs(z_ang) > 0.5 && fabs(z_ang) < 1.4) {
    gamma = (1.4 - fabs(z_ang))/(1.4-0.5);
  } else {
    gamma = 0;
  }
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      foot_pos_des[i][j] = gamma*foot_pos_nominal[i][j] + (1-gamma)*foot_pos_fallback[i][j];
    }
  }
}

void compute_ground_foot_pos_des(Data &data, const StateVec &q, double foot_pos_des[4][3]) {
  StateVec ground_q;
  ground_q << q.head(6), data.ground_joints;
              
  for (int i = 0; i < 4; ++i) {
    fk(ground_q, foot_pos_des[i], (Frame)(i+F_FL_EE));
  }
}

void compute_ground_joint_des(Data &data, const StateVec &q) {
  double foot_pos_init[4][3];
  for (int i = 0; i < 4; ++i) {
    fk(q, foot_pos_init[i], (Frame)(i+F_FL_EE));
  }

  double centroid[2] = {};
  for (int i = 0; i < 4; ++i) {
    centroid[0] += 0.25*foot_pos_init[i][0];
    centroid[1] += 0.25*foot_pos_init[i][1];
  }

  double fx = 0.5*(foot_pos_init[0][0] + foot_pos_init[1][0]);
  double fy = 0.5*(foot_pos_init[0][1] + foot_pos_init[1][1]);
  double bx = 0.5*(foot_pos_init[2][0] + foot_pos_init[3][0]);
  double by = 0.5*(foot_pos_init[2][1] + foot_pos_init[3][1]);

  double rz = atan2(fy-by, fx-bx);

  StateVec q_des = VectorXd::Zero(NUM_Q);
  q_des[0] = centroid[0]+0.012731; //TODO pull actual CoM
  q_des[1] = centroid[1];
  q_des[2] = 0.15;
  q_des[3] = 0;
  q_des[4] = 0;
  q_des[5] = rz;

  double joint_des[12];
  for (int foot = 0; foot < 4; ++foot) {
    ik(robot.contacts[foot].frame, q_des, foot_pos_init[foot], &joint_des[3*foot]);
  }

  for (int i=0; i < 12; ++i) {
    data.ground_joints[i] = joint_des[i];
  }

}

VectorXd compute_foot_torques(Data &data, Contact foot, const StateVec &q, const StateVec &qd, const ContactState c_s, double foot_pos_des[3]) {
  double foot_pos[3] = {};
  Vector3d foot_vel = Vector3d::Zero();
  Vector3d foot_force = Vector3d::Zero();
  fk(q, foot_pos, robot.contacts[foot].frame);
  MatrixXd J = fk_jac(q, robot.contacts[foot].frame).block(0,6+3*foot,3,3);;
  foot_vel = J*qd.segment(6+3*foot,3);
  for (int i = 0; i < 3; ++i) {
    foot_force[i] = 300*(foot_pos_des[i] - foot_pos[i]) - 5*foot_vel[i];
  }

  VectorXd u = J.transpose()*foot_force;
  u = u.cwiseMax(-15);
  u = u.cwiseMin(15);
  return u;
}

};

namespace land {

PrimitiveBehavior* create() {
  return new Land;
}
void destroy(PrimitiveBehavior* prim) {
  delete prim;
}

}
