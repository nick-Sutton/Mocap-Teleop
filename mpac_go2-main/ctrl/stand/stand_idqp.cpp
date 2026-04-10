#include "robot.h"
#include <eigen3/Eigen/Dense>
#include <iostream>

#include "osqp.h"
#include "OsqpEigen/OsqpEigen.h"

#include "math_utils.h"
#include "ctrl_core.h"
#include "full_model_fd.h"
#include "fk.h"
#include "ik.h"
#include "stand_idqp.h"
#include "stand_idqp_args.h"
#include "ctrl/ctrl_utils/traj_gen.h"

extern Robot robot;

using namespace Eigen;

namespace stand_idqp {

static const int num_vars =   NUM_U /*inputs*/
                            + NUM_Q /*qdd-qdd_des*/
                            + 3*4   /*ground reaction forces*/;

static const int num_constraints =   NUM_U /*input constraints*/
                                   + NUM_Q /*dynamic constraints*/
                                   + 3*4   /*contact constraints*/
                                   + 4*4+4 /*admissable force constraints*/;

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

  double t;
  double foot_pos_init[4][3];
  traj_gen::CubicPoly traj[6]; //x,y,z,rx,ry,rz
  
  OsqpEigen::Solver solver;

} Data;

class StandIdqp: public PrimitiveBehavior {

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

  double centroid[2] = {};
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

  double q_des[6] = {centroid[0]+0.012731, //TODO pull actual CoM
                     centroid[1],
                     args.cont[ARG_H]+0.02,
                     args.cont[ARG_RX]*cos(rz) -args.cont[ARG_RY]*sin(rz),
                     args.cont[ARG_RX]*sin(rz) +args.cont[ARG_RY]*cos(rz),
                     args.cont[ARG_RZ] + rz};

  data.t = 0;

  for (int i = 3; i < 6; ++i) {
    q_des[i] = q[i+Q_X] + min_angle_diff(q_des[i],q[i+Q_X]);
  }
  double duration_z = fabs(q_des[Q_Z]-q[Q_Z])/0.15;
  double duration_rx = fabs(q_des[Q_RX]-q[Q_RX])/1.0;
  double duration_ry = fabs(q_des[Q_RY]-q[Q_RY])/1.0;
  double duration_rz = fabs(q_des[Q_RZ]-q[Q_RZ])/1.0;
  double duration = fmax(fmax(fmax(duration_z, duration_rx),duration_ry),duration_rz);
  if (duration < 0.1) duration = 0;
  for (int i = 0; i < 3; ++i) {
    data.traj[i] = traj_gen::cubic_poly_gen(q[i+Q_X],qd[i+Q_X],q_des[i],0,duration);
  }
  for (int i = 3; i < 6; ++i) {
    data.traj[i] = traj_gen::cubic_poly_gen(q[i+Q_X],qd[i+Q_X],q_des[i],0,duration);
  }


  data.qd_cmd_prev = qd.tail(NUM_U);

  if (!data.solver.isInitialized()) {
    data.B = SparseMatrix<double>((int)NUM_Q, (int)NUM_U);
    data.Kp = SparseMatrix<double>((int)NUM_Q, (int)NUM_Q);
    data.Kd = SparseMatrix<double>((int)NUM_Q, (int)NUM_Q);
    data.P = SparseMatrix<double>(num_vars,num_vars);
 
    data.A = SparseMatrix<double>(num_constraints,num_vars);
    data.input_lb = -15*VectorXd::Ones((int)NUM_U);
    data.input_ub =  15*VectorXd::Ones((int)NUM_U);
    data.force_lb = VectorXd::Zero(16+4);
    data.force_ub = VectorXd::Zero(16+4);

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

void execute(const StateVec &q,
             const StateVec &qd,
             const ContactState c_s,
             const Args &args, 
             ActuatorCmds &act_cmds) {
  solve_idqp(act_cmds, q, qd, c_s, args);
}

void compute_stance_origin_info(const StateVec &q,
                                double &rz,
                                double centroid[2],
                                double foot_pos_init[4][3]) const {

  for (int i = 0; i < 4; ++i) {
    fk(q, foot_pos_init[i], (Frame)(i+F_FL_EE));
    centroid[0] += 0.25*foot_pos_init[i][0];
    centroid[1] += 0.25*foot_pos_init[i][1];
  }

  double fx = 0.5*(foot_pos_init[0][0] + foot_pos_init[1][0]);
  double fy = 0.5*(foot_pos_init[0][1] + foot_pos_init[1][1]);
  double bx = 0.5*(foot_pos_init[2][0] + foot_pos_init[3][0]);
  double by = 0.5*(foot_pos_init[2][1] + foot_pos_init[3][1]);

  rz = atan2(fy-by, fx-bx);
}

void body_setpoint(const StateVec &q_in,
                   const StateVec &qd,
                   const ContactState c_s,
                   const Args &args,
                   const double delta_t,
                   StateVec &q_des,
                   StateVec &qd_des,
                   double foot_pos_init[4][3]) const {

  //TODO consolidate this with calls in init
  double relative_z = 0;
  bool relative_z_valid = contact_feet_relative_z(q_in, c_s, relative_z);
  StateVec q = q_in;
  q[Q_Z] = relative_z;

  double centroid[2] = {};
  double rz = 0; 
  compute_stance_origin_info(q, rz, centroid, foot_pos_init);

  q_des[0] = centroid[0]+0.012731, //TODO pull actual CoM
  q_des[1] = centroid[1],
  q_des[2] = args.cont[ARG_H],
  q_des[3] = args.cont[ARG_RX]*cos(rz) -args.cont[ARG_RY]*sin(rz),
  q_des[4] = args.cont[ARG_RX]*sin(rz) +args.cont[ARG_RY]*cos(rz),
  q_des[5] = args.cont[ARG_RZ] + rz;

  for (int i = 3; i < 6; ++i) {
    q_des[i] = q[i+Q_X] + min_angle_diff(q_des[i],q[i+Q_X]);
  }

}

void setpoint(const StateVec &q,
              const StateVec &qd,
              const ContactState c_s,
              const Args &args,
              const double delta_t,
              StateVec &q_des,
              StateVec &qd_des,
              ContactState &c_s_des) const {

  double foot_pos_init[4][3] = {};
  body_setpoint(q, qd, c_s, args, delta_t, q_des, qd_des, foot_pos_init);

  for (int i=0; i<4;++i) {
    ik((Frame)(i+F_FL_EE), q_des, foot_pos_init[i], &q_des[6+3*i]);
  }
  qd_des = VectorXd::Zero((int)NUM_Q);
  memset(&c_s_des, true, sizeof(ContactState));
}

MatrixXd setpoint_jac(const StateVec &q,
                           const StateVec &qd,
                           const ContactState c_s,
                           const Args &args,
                           const double delta_t) const {
  double centroid[2] = {};
  double foot_pos_init[4][3] = {};
  double rz = 0; 
  compute_stance_origin_info(q, rz, centroid, foot_pos_init);

  MatrixXd jac = MatrixXd::Zero((int)NUM_Q+NUM_Q, NUM_CONT_ARGS+1);
  jac(Q_Z, 0) = 1;
  jac(Q_RX, 1) =  cos(rz);
  jac(Q_RX, 2) = -sin(rz);
  jac(Q_RY, 1) =  sin(rz);
  jac(Q_RY, 2) =  cos(rz);
  jac(Q_RZ, 3) = 1;

  return jac;
}

bool in_sroa_underestimate(const StateVec &q,
                           const StateVec &qd,
                           const ContactState c_s,
                           const Args &args,
                           const double delta_t) const {
  if (!in_safe_set(q,qd,c_s,args,delta_t)) {
    return false;
  }

  StateVec q_des = VectorXd::Zero((int)NUM_Q);
  StateVec qd_des = VectorXd::Zero((int)NUM_Q);
  double foot_pos_init[4][3] = {};
  body_setpoint(q, qd, c_s, args, delta_t, q_des, qd_des, foot_pos_init);

  if ((q[Q_Z] > 0.05) &&
      (fabs(min_angle_diff(q_des[Q_RX],q[Q_RX])) < 0.5) &&
      (fabs(min_angle_diff(q_des[Q_RY],q[Q_RY])) < 0.5) &&
      (fabs(qd[Q_X]) < 0.1) &&
      (fabs(qd[Q_Y]) < 0.1) &&
      (fabs(qd[Q_Z]) < 0.5) &&
      //(fabs(q[Q_RZ]-args.cont[ARG_RZ]) < 0.5) &&
      (qd.squaredNorm() < 100)) {
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

  StateVec q_des = VectorXd::Zero((int)NUM_Q);
  StateVec qd_des = VectorXd::Zero((int)NUM_Q);
  double foot_pos_init[4][3] = {};
  body_setpoint(q, qd, c_s, args, delta_t, q_des, qd_des, foot_pos_init);

  if ((q[Q_Z] > 0.05) &&
      (fabs(min_angle_diff(q_des[Q_RX],q[Q_RX])) < 0.5) &&
      (fabs(min_angle_diff(q_des[Q_RY],q[Q_RY])) < 0.5) &&
      (fabs(qd[Q_X]) < 0.15) &&
      (fabs(qd[Q_Y]) < 0.15) &&
      (fabs(qd[Q_Z]) < 0.5) &&
      //(fabs(q[Q_RZ]-args.cont[ARG_RZ]) < 0.5) &&
      (qd.squaredNorm() < 100)) {
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
  
  // at least one foot should be in contact
  if (c_s[0] || c_s[1] || c_s[2] || c_s[3]) {
    return true;
  } else {
    return false;
  }
}

private:
void solve_idqp(ActuatorCmds &act_cmds,
                const StateVec &q_in,
                const StateVec &qd,
                const ContactState c_s,
                const Args &args) {
  StateVec qdd_des = VectorXd::Zero((int)NUM_Q, 1);

  double relative_z = 0;
  bool relative_z_valid = contact_feet_relative_z(q_in, c_s, relative_z);
  StateVec q = q_in;
  q[Q_Z] = relative_z;

  MatrixXd D = robot_D(q);
  MatrixXd H = robot_H(q,qd);

  MatrixXd J = MatrixXd::Zero(3*NUM_C, (int)NUM_Q);
  MatrixXd Jd = MatrixXd::Zero(3*NUM_C, (int)NUM_Q);
  for (int i = 0; i < NUM_C; ++i) {
    J.block(3*i,0,3,(int)NUM_Q) << fk_jac(q,robot.contacts[i].frame);
    Jd.block(3*i,0,3,(int)NUM_Q) << fk_jac_dot(q,qd,robot.contacts[i].frame);
  }

  InputVec q_des = VectorXd::Zero((int)NUM_U);
  InputVec qd_des = VectorXd::Zero((int)NUM_U);

  data.t += CTRL_LOOP_DURATION;
  for (int i = 0; i < 6; ++i) {
    traj_gen::cubic_poly_eval(data.traj[i], data.t, q_des[i+Q_X], qd_des[i+Q_X]);
  }

  Vector3d e_diff = orientation_diff_from_euler(q.segment<3>(Q_RX), q_des.segment<3>(Q_RX));

  StateVec q_err;
  StateVec qd_err;

  q_err << q.head(3) - q_des.head(3),
           e_diff,
           VectorXd::Zero(NUM_U);
  
  qd_err << qd.head(6) - qd_des.head(6), 
           VectorXd::Zero(NUM_U);
  
  qdd_des = data.Kp*(q_err) + data.Kd*(qd_err);

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

};

PrimitiveBehavior* create() {
  return new StandIdqp;
}
void destroy(PrimitiveBehavior* prim) {
  delete prim;
}

}
