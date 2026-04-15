#include "robot.h"
#include <eigen3/Eigen/Dense>
#include <iostream>

#include "osqp.h"
#include "OsqpEigen/OsqpEigen.h"
#include <eigen3/Eigen/Dense>

#include "ctrl_core.h"
#include "full_model_fd.h"
#include "fk.h"
#include "ik.h"
#include "walk_quasi_idqp.h"
#include "walk_quasi_idqp_args.h"
#include "ctrl/ctrl_utils/traj_gen.h"

extern Robot robot;

using namespace Eigen;

namespace walk_quasi_idqp {

static const int num_vars =   NUM_U /*inputs*/
                            + NUM_Q /*qdd-qdd_des*/
                            + 3*4   /*ground reaction forces*/;

static const int num_constraints =   NUM_U /*input constraints*/
                                   + NUM_Q /*dynamic constraints*/
                                   + 3*4   /*contact constraints*/
                                   + 4*4+4 /*admissable force constraints*/;

const double step_height = 0.075;//0.3
const double step_time = 0.3;//0.3
const double dwell_time = 0.5;

typedef enum {
  W_FL = 0,
  W_BR,
  W_FR,
  W_BL,
  NUM_WALK_PHASES
} WalkPhase;


typedef struct {
  SparseMatrix<double> B;
  SparseMatrix<double> Kp;
  SparseMatrix<double> Kd;
  SparseMatrix<double> P;
  Matrix<double, num_vars, 1> qo;
 
  SparseMatrix<double> A;
  Matrix<double, num_constraints, 1> lb;
  Matrix<double, num_constraints, 1> ub;
  VectorXd input_lb;
  VectorXd input_ub;
  VectorXd force_lb;
  VectorXd force_ub;
 
  InputVec qd_cmd_prev;
  ContactState c_s_prev;
  WalkPhase phase;

  double foot_pos_init[4][3];
  double t;
  
  bool swing_legs[4];
  
  StateVec q_des_prev;

  OsqpEigen::Solver solver;

} Data;

class WalkQuasiIdqp: public PrimitiveBehavior {

Data data;

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
  data.t = 0;
  data.qd_cmd_prev = qd.tail(NUM_U);
  data.q_des_prev = q;

  data.B = SparseMatrix<double>((int)NUM_Q, (int)NUM_U);
  data.Kp = SparseMatrix<double>((int)NUM_Q, (int)NUM_Q);
  data.Kd = SparseMatrix<double>((int)NUM_Q, (int)NUM_Q);
  data.P = SparseMatrix<double>(num_vars,num_vars);
 
  data.A = SparseMatrix<double>(num_constraints,num_vars);
  data.input_lb = -15*VectorXd::Ones((int)NUM_U);
  data.input_ub =  15*VectorXd::Ones((int)NUM_U);
  data.force_lb = VectorXd::Zero(16+4);
  data.force_ub = VectorXd::Zero(16+4);
 
  data.qd_cmd_prev = VectorXd::Zero(NUM_U);

  data.Kp.coeffRef(Q_X,Q_X) = -200;
  data.Kp.coeffRef(Q_Y,Q_Y) = -200;
  data.Kp.coeffRef(Q_Z,Q_Z) = -200;
  data.Kp.coeffRef(Q_RX,Q_RX) = -300;
  data.Kp.coeffRef(Q_RY,Q_RY) = -300;
  data.Kp.coeffRef(Q_RZ,Q_RZ) = -300;

  data.Kd.coeffRef(Q_X,Q_X) = -15;
  data.Kd.coeffRef(Q_Y,Q_Y) = -15;
  data.Kd.coeffRef(Q_Z,Q_Z) = -15;
  data.Kd.coeffRef(Q_RX,Q_RX) = -15;
  data.Kd.coeffRef(Q_RY,Q_RY) = -15;
  data.Kd.coeffRef(Q_RZ,Q_RZ) = -15;

  for (int i = Q_FL1; i < NUM_Q; ++i) {
    data.Kd.coeffRef(i,i) = -50;
  }

  //Kp*=0.1;
  //Kd*=0.1;

  /* objective function x'Px + qo'x */
  /* 2-norm on body qdd-qdd_des */
  for (int i = NUM_U; i < NUM_U + 6; ++i) {
    data.P.coeffRef(i,i) = 1;
  }

  for (int i = NUM_U+6; i < NUM_U + NUM_Q; ++i) {
    data.P.coeffRef(i,i) = 0.00001;
  }
  data.qo.setZero();

  /* constraints lb < Ax < ub */
  data.lb.setOnes();data.lb*=-OSQP_INFTY;
  data.ub.setOnes();data.ub*=OSQP_INFTY;

  /*input limits*/
  /* lb < u < ub */
  for (int i = 0; i < NUM_U; ++i) {
    data.A.coeffRef(i,i) = 1; //identity
  }

  /* Dynamics Constraints */
  /* D*qdd - B*u -J'F = -H */
  /*-B*u*/
  for (int i = 0; i < NUM_U; ++i) {
    data.A.coeffRef(i+NUM_U+6,i) = -1;
  }

  /*Admissable Forces*/
  int addmis_force_offset = NUM_U+NUM_Q+3*NUM_C;
  int grf_var_offset = NUM_U+NUM_Q;
  for (int i = 0; i < NUM_C; ++i) {
    double mu = 0.75*robot.contacts[i].mu;
    int fx_constr_l = addmis_force_offset+5*i;
    int fx_constr_u = addmis_force_offset+5*i+1;
    int fy_constr_l = addmis_force_offset+5*i+2;
    int fy_constr_u = addmis_force_offset+5*i+3;
    int fz_constr = addmis_force_offset+5*i+4;
    int fx_var = grf_var_offset+3*i;
    int fy_var = grf_var_offset+3*i+1;
    int fz_var = grf_var_offset+3*i+2;
    // x - mu*z < 0
    data.A.coeffRef(fx_constr_l, fx_var) = 1;
    data.A.coeffRef(fx_constr_l, fz_var) = -mu; //mu*z
    data.force_lb(5*i) = -OSQP_INFTY;
    data.force_ub(5*i) = 0;
    // x + mu*z > 0
    data.A.coeffRef(fx_constr_u, fx_var) = 1;
    data.A.coeffRef(fx_constr_u, fz_var) = mu; //mu*z
    data.force_lb(5*i+1) = 0;
    data.force_ub(5*i+1) = OSQP_INFTY;
    // y - mu*z < 0
    data.A.coeffRef(fy_constr_l, fy_var) = 1;
    data.A.coeffRef(fy_constr_l, fz_var) = -mu; //mu*z
    data.force_lb(5*i+2) = -OSQP_INFTY;
    data.force_ub(5*i+2) = 0;
    // y + mu*z > 0
    data.A.coeffRef(fy_constr_u, fy_var) = 1;
    data.A.coeffRef(fy_constr_u, fz_var) = mu; //mu*z
    data.force_lb(5*i+3) = 0;
    data.force_ub(5*i+3) = OSQP_INFTY;
    // z > 0
    data.A.coeffRef(fz_constr, fz_var) = 1;
    data.force_lb(5*i+4) = 0;
    data.force_ub(5*i+4) = OSQP_INFTY;
  }


  //TODO probably need a finish method to deallocate memory from controllers
  //as is now this remains allocated for all time if you run this once
  if (data.solver.isInitialized()) {
    data.solver.updateHessianMatrix(data.P);
    data.solver.updateGradient(data.qo);
    data.solver.updateLinearConstraintsMatrix(data.A);
    data.solver.updateLowerBound(data.lb);
    data.solver.updateUpperBound(data.ub);

  } else {
    data.solver.settings()->setVerbosity(false);
    data.solver.data()->setNumberOfVariables(num_vars);
    data.solver.data()->setNumberOfConstraints(num_constraints);
    data.solver.data()->setHessianMatrix(data.P);
    data.solver.data()->setGradient(data.qo);
    data.solver.data()->setLinearConstraintsMatrix(data.A);
    data.solver.data()->setLowerBound(data.lb);
    data.solver.data()->setUpperBound(data.ub);
    data.solver.initSolver();
  }

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

  data.t += CTRL_LOOP_DURATION;
  bool dwell;
  gait_state_machine(data, data.t, q, data.phase, dwell);
  solve_idqp(data, act_cmds, q, qd, c_s, args, dwell);
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
  if (!in_safe_set(q,qd,c_s,args,delta_t)) {
    return false;
  }
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
  if (!in_safe_set(q,qd,c_s,args,delta_t)) {
    return false;
  }
  double relative_z = 0;
  bool relative_z_valid = contact_feet_relative_z(q, c_s, relative_z);
  if (fabs(relative_z - args.cont[ARG_H]) < 0.05) {
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

void solve_idqp(Data &data, ActuatorCmds &act_cmds, const StateVec &q, const StateVec &qd, const ContactState c_s, const Args &args, bool dwell) {

  StateVec qdd_des = VectorXd::Zero((int)NUM_Q, 1);

  Contact foot;
  Inputs act_u;

  switch (data.phase) {
    case W_FL:
      foot = C_FL_EE;
      act_u = U_FL1;
      break;
    case W_BR:
      foot = C_BR_EE;
      act_u = U_BR1;
      break;
    case W_FR:
      foot = C_FR_EE;
      act_u = U_FR1;
      break;
    case W_BL:
      foot = C_BL_EE;
      act_u = U_BL1;
      break;
    default:
      break;
  }

  MatrixXd D = robot_D(q);
  MatrixXd H = robot_H(q,qd);

  MatrixXd J = MatrixXd::Zero(3*NUM_C, (int)NUM_Q);
  MatrixXd Jd = MatrixXd::Zero(3*NUM_C, (int)NUM_Q);
  for (int i = 0; i < NUM_C; ++i) {
    if (i != foot || dwell) {
      J.block(3*i,0,3,(int)NUM_Q) << fk_jac(q,robot.contacts[i].frame);
      Jd.block(3*i,0,3,(int)NUM_Q) << fk_jac_dot(q,qd,robot.contacts[i].frame);
    }
  }

  StateVec q_des = VectorXd::Zero((int)NUM_Q);
  StateVec qd_des = VectorXd::Zero((int)NUM_Q);

  q_des[Q_Z] = args.cont[ARG_H];

  qd_des[Q_RZ] = args.cont[ARG_VRZ];
  q_des[Q_RZ] = CTRL_LOOP_DURATION*qd_des[Q_RZ] + data.q_des_prev[Q_RZ];

  Matrix3d R_curr;
  R_curr = AngleAxisd(q(Q_RX), Vector3d::UnitX()) *
           AngleAxisd(q(Q_RY), Vector3d::UnitY()) *
           AngleAxisd(q(Q_RZ), Vector3d::UnitZ()); 

  Matrix3d R_des;
  R_des  = AngleAxisd(q_des(Q_RX), Vector3d::UnitX()) *
           AngleAxisd(q_des(Q_RY), Vector3d::UnitY()) *
           AngleAxisd(q_des(Q_RZ), Vector3d::UnitZ()); 

  VectorXd R_err = 0.5*((R_des*Vector3d::UnitX()).cross(R_curr*Vector3d::UnitX()) +
                        (R_des*Vector3d::UnitY()).cross(R_curr*Vector3d::UnitY()) +
                        (R_des*Vector3d::UnitZ()).cross(R_curr*Vector3d::UnitZ()));

  StateVec q_err;
  StateVec qd_err;

  q_err << q.head(3) - q_des.head(3),
           R_err,
           VectorXd::Zero(NUM_U,1);

  compute_planar_body_err(q, qd, foot, q_err, data.t);
  
  qd_err << qd.head(6) - qd_des.head(6), 
           VectorXd::Zero(NUM_U,1);
  
  qdd_des = data.Kp*(q_err) + data.Kd*(qd_err);

  data.q_des_prev = q_des;

  /*Dynamics constraints*/
  /* D*qdd - B*u -J'F = -H */
  /*subtracting D*qdd_des from both sides to match decision var */
  /*D*(qdd-qdd_des) */
  //std::cout << "D: \n" << D << std::endl;
  for (int i = 0; i < NUM_Q; ++i) {
    for (int j = 0; j <= i; ++j) {
      data.A.coeffRef(i+NUM_U,j+NUM_U) = D(j,i);
      data.A.coeffRef(j+NUM_U,i+NUM_U) = D(j,i);
    }
  }
  //std::cout << "A: \n" << A << std::endl;
  /*-J^T*F*/
  for (int i = 0; i < 3*NUM_C; ++i) {
    for (int j = 0; j < NUM_Q; ++j) {
      data.A.coeffRef(j+NUM_U,i+NUM_U+NUM_Q) = -J(i,j);
    }
  }
  /* -H - D*qdd_des*/
  VectorXd dynamics_lb = -H - D*qdd_des;
  VectorXd dynamics_ub = -H - D*qdd_des;

  /*Contact constraints*/
  /*subtracting J*qdd_des from both sides to match decision var */
  /*J*qdd = -Jd*qd*/
  /*J*(qdd-qdd_des) */
  for (int i = 0; i < 3*NUM_C; ++i) {
    for (int j = 0; j < NUM_Q; ++j) {
      data.A.coeffRef(i+NUM_U+NUM_Q,j+NUM_U) = J(i,j);
    }
  }
  /* -Jd*qd - J*qdd_des*/
  VectorXd contact_lb = -Jd*qd - J*qdd_des;
  VectorXd contact_ub = -Jd*qd - J*qdd_des;

  //std::cout << "A: " << A << std::endl;

  data.lb << data.input_lb, dynamics_lb, contact_lb, data.force_lb;
  data.ub << data.input_ub, dynamics_ub, contact_ub, data.force_ub;

  //std::cout << lb << ub << std::endl;

  data.solver.updateLinearConstraintsMatrix(data.A);
  data.solver.updateBounds(data.lb, data.ub);
  bool success = data.solver.solve();
  VectorXd sol = data.solver.getSolution();

  //std::cout << "sol: \n" << sol << std::endl;
  if (success) {
    InputVec u = sol.head(NUM_U);
    StateVec qdd = sol.segment<NUM_Q>(NUM_U);
    double gamma = 0.5;
    InputVec qd_fuse = (1-gamma)*qd.tail(NUM_U) + gamma*data.qd_cmd_prev;
    for (int i = 0; i < NUM_U; ++i) {
      act_cmds.mode[i] = CMD_MODE_TAU_VEL_POS;
      act_cmds.u[i] = u[i];
      act_cmds.qd[i] = qd_fuse[i] + CTRL_LOOP_DURATION*(qdd[i+Q_FL1] + qdd_des[i+Q_FL1]);
      act_cmds.q[i] = q[i+Q_FL1] + CTRL_LOOP_DURATION*qd_fuse[i]+0.5*CTRL_LOOP_DURATION*CTRL_LOOP_DURATION*(qdd[i+Q_FL1] + qdd_des[i+Q_FL1]);
      act_cmds.kp[i] = 200;
      act_cmds.kd[i] = 5;
    }
    data.qd_cmd_prev = act_cmds.qd;

    double joint_des[3];

    double t_phase = 1/step_time*(fmod(data.t, step_time+dwell_time)-dwell_time);
    if (t_phase >= 0) {
      swing_leg_setpoint(data, foot, q, data.t, joint_des, args);

      for (int i = act_u; i < act_u+3; ++i) {
        act_cmds.mode[i] = CMD_MODE_TAU_VEL_POS;
        act_cmds.u[i] = 0;
        act_cmds.qd[i] = 0;
        act_cmds.q[i] = joint_des[i-act_u];
        act_cmds.kp[i] = 200;
        act_cmds.kd[i] = 5;
      }

    }
    
  
  } else {
    /* soft stop */
    printf("failed, soft_stopping\n");
    InputVec Kd = 1*VectorXd::Ones((int)NUM_U);
    InputVec u = Kd.cwiseProduct(-qd.tail(NUM_U));
    for (int i = 0; i < NUM_U; ++i) {
      act_cmds.mode[i] = CMD_MODE_TAU;
      act_cmds.u[i] = u[i];
    }
  }
}

//x, y, rz
void compute_planar_body_err(const StateVec &q, const StateVec &qd, Contact swing_foot, StateVec &q_err, double t) {
  double centroid[2] = {};
  double foot_pos[4][3];
  double support_foot_pos[4][3];

  double support_centroid[2] = {};

  for (int i = 0; i < 4; ++i) {
    fk(q, foot_pos[i], (Frame)(i+F_FL_EE));
    //foot_pos_init[i][2] =0;
    centroid[0] += 0.25*foot_pos[i][0];
    centroid[1] += 0.25*foot_pos[i][1];

    if (i != swing_foot) {
      support_centroid[0] += foot_pos[i][0]/3.0;
      support_centroid[1] += foot_pos[i][1]/3.0;
    }

  }

  double margin_frac = 0.65;
  for (int i = 0; i < 4; ++i) {
    support_foot_pos[i][0] = margin_frac*foot_pos[i][0] + (1-margin_frac)*support_centroid[0];
    support_foot_pos[i][1] = margin_frac*foot_pos[i][1] + (1-margin_frac)*support_centroid[1];
  }
  

  //double fx = 0.5*(foot_pos[0][0] + foot_pos[1][0]);
  //double fy = 0.5*(foot_pos[0][1] + foot_pos[1][1]);
  //double bx = 0.5*(foot_pos[2][0] + foot_pos[3][0]);
  //double by = 0.5*(foot_pos[2][1] + foot_pos[3][1]);

  //double rz = atan2(fy-by, fx-bx);

  double com[3];
  fk_com(q, com);

  double m;
  double b;
  int f1,f2;
  double goal[2];

  switch (swing_foot) {
    case C_FL_EE:
      f1 = C_FR_EE;
      f2 = C_BL_EE;
      break;
    case C_BR_EE:
      f2 = C_FR_EE;
      f1 = C_BL_EE;
      break;
    case C_FR_EE:
      f1 = C_FL_EE;
      f2 = C_BR_EE;
      break;
    case C_BL_EE:
      f2 = C_FL_EE;
      f1 = C_BR_EE;
      break;
    default:
      break;
  }

  m = (support_foot_pos[f1][1]-support_foot_pos[f2][1])
      /(support_foot_pos[f1][0]-support_foot_pos[f2][0]);
  b = support_foot_pos[f1][1] - m*support_foot_pos[f1][0];

  //TODO: handle division by zero
  double m2 = -1/m;
  double b2 = centroid[1] - m2*centroid[0];

  double x_intersection = (b2-b)/(m-m2);
  double y_intersection = m*x_intersection+b;

  double th = atan2(centroid[1]-y_intersection,centroid[0]-x_intersection);

  double margin = 0;

  goal[0] = x_intersection + margin*cos(th);
  goal[1] = y_intersection + margin*sin(th);

  //printf("x: %f, y: %f, centroid: %f %f\n", x_intersection, y_intersection, centroid[0], centroid[1]);

  double max_time = 0.95*(dwell_time+step_time);
  double duration = fmax(0, max_time - fmod(t, dwell_time + step_time));
  //double vel_des_dummy;
  //traj_gen::CubicPoly com_traj[2];
  //com_traj[0] = traj_gen::cubic_poly_gen(0,0,com[0]-goal[0],0,duration);
  //com_traj[1] = traj_gen::cubic_poly_gen(0,0,com[1]-goal[1],0,duration);
  //traj_gen::cubic_poly_eval(com_traj[0], 0.001, q_err[Q_X], vel_des_dummy);
  //traj_gen::cubic_poly_eval(com_traj[1], 0.001, q_err[Q_Y], vel_des_dummy);

  q_err[Q_X] = (1-duration/max_time)*(com[0]-goal[0]);//centroid[0];
  q_err[Q_Y] = (1-duration/max_time)*(com[1]-goal[1]);//centroid[1];
  

}

void gait_state_machine(Data &data, double t, const StateVec &q, WalkPhase &phase, bool &dwell) {
  double duration = 4*(dwell_time+step_time);
  double t_phase = fmod(t,duration);

  dwell = false;
  for (int p = 0; p < 4; ++p) {
    double test_t = t_phase - p*(dwell_time + step_time);
    if (test_t >= 0 && test_t < dwell_time) {
      for (int i = 0; i < 4; ++i) {
        fk(q, data.foot_pos_init[i], (Frame)(i+F_FL_EE));
      }
      dwell = true;
    } 
    if (test_t >= 0 && test_t < dwell_time + step_time) {
      phase = (WalkPhase) p;
    }
  }
}

double goal_pos[3] = {};
int foot_prev = -1;

void swing_leg_setpoint(Data &data, Contact foot, const StateVec &q, double t, double joint_des[3], const Args &args) {
  traj_gen::CubicPoly swing_traj[3];

  double t_phase = fmax(0,1/step_time*(fmod(t, step_time+dwell_time)-dwell_time));
  double foot_pos_des[3];
  double foot_vel_des[3];
  
  if (foot != foot_prev) {
    //double goal_pos[3] = {};
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
    fk(q_neutral, goal_pos, robot.contacts[foot].frame);

    //memcpy(goal_pos, &foot_pos_init[foot][0], 2*sizeof(double));

    //double foot_pos[3];
    //double foot_vel_des[3];
    //fk(foot_pos, robot.contacts[foot].frame);
    double r = sqrt((goal_pos[1]-q[Q_Y])*(goal_pos[1]-q[Q_Y])+
                    (goal_pos[0]-q[Q_X])*(goal_pos[0]-q[Q_X]));

    double th = 0.5*args.cont[ARG_VRZ]*4*(step_time+dwell_time) + atan2(goal_pos[1]-q[Q_Y],goal_pos[0]-q[Q_X]);

    goal_pos[0] = q[Q_X] + r*cos(th);
    goal_pos[1] = q[Q_Y] + r*sin(th);

    goal_pos[0] += (args.cont[ARG_VX]*cos(q[Q_RZ]) - args.cont[ARG_VY]*sin(q[Q_RZ]))*4*(step_time+dwell_time);
    goal_pos[1] += (args.cont[ARG_VX]*sin(q[Q_RZ]) + args.cont[ARG_VY]*cos(q[Q_RZ]))*4*(step_time+dwell_time);

    //printf("pos: %f goal: %f init:%f\n", q[Q_X], goal_pos[0],foot_pos_init[foot][0]);
    if (goal_pos[0] > 0.25 && goal_pos[0] < 1) {
      if (goal_pos[1] > 0) {
        goal_pos[1] = fmax(0.26,goal_pos[1]);
      } else {
        goal_pos[1] = fmin(-0.26,goal_pos[1]);
      }
    }
  }
  foot_prev = foot;


  for (int i = 0; i < 2; ++i) {
    swing_traj[i] = traj_gen::cubic_poly_gen(data.foot_pos_init[foot][i],0,goal_pos[i],0,step_time*0.8);
    traj_gen::cubic_poly_eval(swing_traj[i], step_time*t_phase, foot_pos_des[i], foot_vel_des[i]);
  }

  if (t_phase < 0.5) {
    swing_traj[2] = traj_gen::cubic_poly_gen(0,0,step_height,0,step_time*0.5);
    traj_gen::cubic_poly_eval(swing_traj[2], step_time*t_phase, foot_pos_des[2], foot_vel_des[2]);
  } else {
    swing_traj[2] = traj_gen::cubic_poly_gen(step_height,0,0,0,step_time*0.5);
    traj_gen::cubic_poly_eval(swing_traj[2], step_time*(t_phase-0.5), foot_pos_des[2], foot_vel_des[2]);
  }

  ik(robot.contacts[foot].frame, q, foot_pos_des, joint_des);

}


};

PrimitiveBehavior* create() {
  return new WalkQuasiIdqp;
}
void destroy(PrimitiveBehavior* prim) {
  delete prim;
}

}
