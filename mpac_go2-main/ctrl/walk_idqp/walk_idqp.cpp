#include "robot.h"
#include <eigen3/Eigen/Dense>
#include <iostream>

#include "osqp.h"
#include "OsqpEigen/OsqpEigen.h"
#include <eigen3/Eigen/Dense>

#include "math_utils.h"
#include "ctrl_core.h"
#include "full_model_fd.h"
#include "ik.h"
#include "fk.h"
#include "walk_idqp.h"
#include "walk_idqp_args.h"
#include "walk_idqp_tlm.h"
#include "ctrl/ctrl_utils/traj_gen.h"
#include "ctrl_mode_class.h"

// #include <fstream>

extern Robot robot;

using namespace Eigen;

namespace walk_idqp {

static const int num_vars =   NUM_U /*inputs*/
                            + NUM_Q /*qdd-qdd_des*/
                            + 3*4   /*ground reaction forces*/;

static const int num_constraints =   NUM_U /*input constraints*/
                                   + NUM_Q /*dynamic constraints*/
                                   + 3*4   /*contact constraints*/
                                   + 4*4+4 /*admissable force constraints*/;
                                   //+ NUM_U /*qdd-qdd_des for swing legs */;
static const int qd_buff_size = 100;
const double step_height = 0.075;//0.3
const double step_time = 0.25;//0.3
const double dwell_time = 0.05;
const double qdd_des_limit[3] = {1.2, 0.8, 2};
// const double qd_des_limit[3] = {0.3, 0.2, 0.5};
const double qd_des_limit[3] = {0.4, 0.3, 0.5};

//double params[] = {50,50,500,200,200,400};
//double params[] = {0,0,300,300,300,400,
//                   25,25,35,15,15,35};
double params[] = {0,0,300,300,300,400,
                   25,25,35,15,15,35};

typedef enum {
  W_DIAG1 = 0,
  W_DIAG2,
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
  double i_err[2];
  
  bool swing_legs[4];
  
  double swing_leg_q_des[4][3];
  double swing_leg_qd_des[4][3];
  double swing_leg_qdd_des[4][3];
  
  std::deque<StateVec> qd_buffer;
  
  StateVec q_des_prev;
  
  double qd_arg_smoothed[3];
  
  OsqpEigen::Solver solver;

  std::ofstream outFile;

} Data;

class WalkIdqp: public PrimitiveBehavior {

WalkIdqpTelemetry walk_tlm;
Data data;

public:
WalkIdqp() {
  tlm_ptr = (char*) &walk_tlm;
  tlm_size = sizeof walk_tlm;
}
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
  //data.t = 0;
  
  //this is a hack, but might be a necessary one
  //ideally initialization doesn't depend on the previous 
  //controller, but that may not be possible
  //TODO probably add an "update" method to supercede init in the 
  //same primitive is called with different arguments
  if (robot.ctrl_curr.mode != C_WALK_IDQP) {

    double relative_z = 0;
    bool relative_z_valid = contact_feet_relative_z(q_in, c_s, relative_z);
    StateVec q = q_in;
    q[Q_Z] = relative_z;

    for (int i = 0; i < 4; ++i) {
      fk(q, data.foot_pos_init[i], (Frame)(i+F_FL_EE));
      data.swing_legs[i]=false;
    }

    nearest_phasing(data, q, qd, c_s, data.foot_pos_init);
    data.i_err[0] = 0;
    data.i_err[1] = 0;

    data.q_des_prev = q;
    data.qd_cmd_prev = qd.tail(NUM_U);

    // TODO should this really be FIR filter or IIR?
    for (int i=0; i < qd_buff_size; ++i) {
      data.qd_buffer.push_front(qd);
    }

    memcpy(&data.c_s_prev, c_s, sizeof(ContactState));

    //TODO transform this from current velocity?
    for (int i = 0; i < 3; ++i) {
      data.qd_arg_smoothed[i] = 0;
    }
    
  }


  //TODO probably need a finish method to deallocate memory from controllers
  //as is now this remains allocated for all time if you run this once
  if (!data.solver.isInitialized()) {

    data.B = SparseMatrix<double>((int)NUM_Q, (int)NUM_U);
    data.Kp = SparseMatrix<double>((int)NUM_Q, (int)NUM_Q);
    data.Kd = SparseMatrix<double>((int)NUM_Q, (int)NUM_Q);
    data.P = SparseMatrix<double>(num_vars,num_vars);
 
    data.A = SparseMatrix<double>(num_constraints,num_vars);
    data.input_lb = -23.7*VectorXd::Ones((int)NUM_U);
    data.input_ub =  23.7*VectorXd::Ones((int)NUM_U);
    data.force_lb = VectorXd::Zero(16+4);
    data.force_ub = VectorXd::Zero(16+4);

    data.Kp.coeffRef(Q_X,Q_X) = -params[0]; //-50;
    data.Kp.coeffRef(Q_Y,Q_Y) = -params[1]; //-50;
    data.Kp.coeffRef(Q_Z,Q_Z) = -params[2]; //-500;
    data.Kp.coeffRef(Q_RX,Q_RX) = -params[3]; //-200;
    data.Kp.coeffRef(Q_RY,Q_RY) = -params[4]; //-200;
    data.Kp.coeffRef(Q_RZ,Q_RZ) = -params[5]; //-400;

    data.Kd.coeffRef(Q_X,Q_X) = -params[6];
    data.Kd.coeffRef(Q_Y,Q_Y) = -params[7];
    data.Kd.coeffRef(Q_Z,Q_Z) = -params[8];
    data.Kd.coeffRef(Q_RX,Q_RX) = -params[9];
    data.Kd.coeffRef(Q_RY,Q_RY) = -params[10];
    data.Kd.coeffRef(Q_RZ,Q_RZ) = -params[11];

    for (int i = Q_FL1; i < NUM_Q; ++i) {
      data.Kp.coeffRef(i,i) = -5e4;
      data.Kd.coeffRef(i,i) = -5e2;
    }

    /* objective function x'Px + qo'x */
    for (int i = 0; i < num_vars; ++i) {
      data.P.coeffRef(i,i) = 1e-5;
    }
    /* 2-norm on body qdd-qdd_des */
    data.P.coeffRef(NUM_U+Q_X,NUM_U+Q_X) = 1;
    data.P.coeffRef(NUM_U+Q_Y,NUM_U+Q_Y) = 1;
    data.P.coeffRef(NUM_U+Q_Z,NUM_U+Q_Z) = 1;
    for (int i = NUM_U+3; i < NUM_U + 6; ++i) {
      data.P.coeffRef(i,i) = 1;
    }

    for (int i = NUM_U+6; i < NUM_U + NUM_Q; ++i) {
      data.P.coeffRef(i,i) = 1e-5;
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
      double mu = robot.contacts[i].mu;
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

  // save data ## todo: we need to modify the data saver later.....
  //  data.outFile.open("/home/jlee267/mpac/mpac_a1/data/walk_traj.txt",std::ios::app);

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

  /* reset tlm */
  memset(&walk_tlm, 0, sizeof(WalkIdqpTelemetry));
  if (c_s != data.c_s_prev) {
    memcpy(&data.c_s_prev, c_s, sizeof(ContactState));
  }
  data.t += CTRL_LOOP_DURATION;
  StateVec qd_filtered = update_qd_filter(qd);
  gait_state_machine(data.t, q, c_s, data.swing_legs, data.foot_pos_init);
  StateVec qd_des = compute_body_qd_des(q, args);
  StateVec q_des = compute_body_q_des(q, qd_des, args);
  StateVec qdd_des = compute_body_qdd_des(q, qd, q_des, qd_des);
  for (int i = 0; i < NUM_C; ++i) {
    walk_tlm.swing_leg[i] = data.swing_legs[i];
    if (data.swing_legs[i]) {
      //TODO inconsistent w.r.t. variables in data vs local to execute
      compute_swing_leg_joint_des(data, (Contact)i, q, qd, qd_filtered, qd_des);
      qdd_des[6+3*i] = data.swing_leg_qdd_des[i][0];
      qdd_des[6+3*i+1] = data.swing_leg_qdd_des[i][1];
      qdd_des[6+3*i+2] = data.swing_leg_qdd_des[i][2];
    }
  }
  InputVec u_sol;
  StateVec qdd_sol;
  Eigen::VectorXd fc_sol;
  bool feasible = solve_idqp(data, q, qd, c_s, qdd_des, u_sol, qdd_sol, fc_sol);

  // if (data.outFile.is_open()) {
  //     // Iterate through the vector and write each value to the file
  //     for (size_t i = 0; i < 3*NUM_C; ++i) {
  //         data.outFile << fc_sol[i];
  //         if (i != 3*NUM_C-1) {
  //             data.outFile << ", ";  // Add a comma after each value except the last one
  //         }
  //       }
  //      data.outFile << "\n";  
  // }

  // std::cout<<"Feasibility: "<< feasible <<std::endl;
  fill_act_cmds(data, act_cmds, feasible, q, qd, u_sol, qdd_sol);

  // fill the contact force from WBC
  // for (int i=0; i<NUM_C; ++i){ 
  //   act_cmds.fc(i,0) = fc_sol[3*i];
  //   act_cmds.fc(i,1) = fc_sol[3*i+1];
  //   act_cmds.fc(i,2) = fc_sol[3*i+2];
  // }

}

bool has_tlm() {return true;};

void create_attributes_type(hid_t &attr_type) {
  hid_t str32 = H5Tcopy (H5T_C_S1);
  size_t size = 32 * sizeof(char);
  H5Tset_size(str32, size);

  hid_t single_attr_type = H5Tcreate (H5T_COMPOUND, sizeof (TelemetryAttribute));
  H5Tinsert(single_attr_type, "name", HOFFSET (TelemetryAttribute, name), str32);
  H5Tinsert(single_attr_type, "units", HOFFSET (TelemetryAttribute, units), str32);
  H5Tinsert(single_attr_type, "crit_lo", HOFFSET (TelemetryAttribute, crit_lo), H5T_NATIVE_DOUBLE);
  H5Tinsert(single_attr_type, "warn_lo", HOFFSET (TelemetryAttribute, warn_lo), H5T_NATIVE_DOUBLE);
  H5Tinsert(single_attr_type, "warn_hi", HOFFSET (TelemetryAttribute, warn_hi), H5T_NATIVE_DOUBLE);
  H5Tinsert(single_attr_type, "crit_hi", HOFFSET (TelemetryAttribute, crit_hi), H5T_NATIVE_DOUBLE);

  hsize_t dims_feet[1] = {4};
  hid_t foot_num_vec = H5Tarray_create(single_attr_type, 1, dims_feet);
  hsize_t dims_feet_pos[1] = {12};
  hid_t foot_pos_vec = H5Tarray_create(single_attr_type, 1, dims_feet_pos);

  attr_type = H5Tcreate (H5T_COMPOUND, sizeof (TelemetryAttributes));
  H5Tinsert(attr_type, "swing_leg", HOFFSET (TelemetryAttributes, swing_leg), foot_num_vec);
  H5Tinsert(attr_type, "swing_feet_err", HOFFSET (TelemetryAttributes, swing_feet_err), foot_pos_vec);
};


void fill_attributes(std::vector<TelemetryAttribute> &attributes) {
  TelemetryAttributes attr;

  Robot robot;
  robot_params_init(robot);

  memset(&attr, 0, sizeof(TelemetryAttributes));

  for (int i = 0; i < 4; ++i) {
    sprintf(attr.swing_leg[i].name, "swing_%s", &robot.contacts[i].name[2]);
  }
  for (int i = 0; i < 4; ++i) {
    sprintf(attr.swing_feet_err[3*i].name, "swing_%s_err_x", &robot.contacts[i].name[2]);
    sprintf(attr.swing_feet_err[3*i+1].name, "swing_%s_err_y", &robot.contacts[i].name[2]);
    sprintf(attr.swing_feet_err[3*i+2].name, "swing_%s_err_z", &robot.contacts[i].name[2]);
  }
  for (int i = 0; i < 12; ++i) {
    strcpy(attr.swing_feet_err[i].units, "m");
    attr.swing_feet_err[i].crit_lo = -0.1;
    attr.swing_feet_err[i].warn_lo = -0.05;
    attr.swing_feet_err[i].crit_hi =  0.05;
    attr.swing_feet_err[i].warn_hi =  0.1;
  }

  //TODO probably a cleaner way...
  TelemetryAttribute *attr_array = (TelemetryAttribute*) &attr;
  for (int i=0;i < sizeof(TelemetryAttributes)/sizeof(TelemetryAttribute); ++i) {
    attributes.push_back(attr_array[i]);
  }
}

void create_timeseries_type(hid_t &timeseries_type) {
  hsize_t dims_feet[1] = {4};
  hid_t foot_num_vec = H5Tarray_create(H5T_NATIVE_INT, 1, dims_feet);
  hsize_t dims_feet_pos[1] = {12};
  hid_t foot_pos_vec = H5Tarray_create(H5T_NATIVE_DOUBLE, 1, dims_feet_pos);

  timeseries_type = H5Tcreate (H5T_COMPOUND, sizeof (WalkIdqpTelemetry));
  H5Tinsert(timeseries_type, "swing_leg", HOFFSET (WalkIdqpTelemetry, swing_leg), foot_num_vec);
  H5Tinsert(timeseries_type, "swing_feet_err", HOFFSET (WalkIdqpTelemetry, swing_feet_err), foot_pos_vec);
};
/* Note for softstop, we always want to be able to transition in case of emergency */
void setpoint(const StateVec &q_in,
              const StateVec &qd,
              const ContactState c_s,
              const Args &args,
              const double delta_t,
              StateVec &q_des,
              StateVec &qd_des,
              ContactState &c_s_des) const {
  //TODO fill in

  //TODO should just make this a function, it will be used commonly
  double relative_z = 0;
  bool relative_z_valid = contact_feet_relative_z(q_in, c_s, relative_z);
  StateVec q = q_in;
  q[Q_Z] = relative_z;

  q_des = q;
  qd_des = VectorXd::Zero((int)NUM_Q);
  q_des[Q_Z] = args.cont[ARG_H];
  q_des[Q_RX] = 0;
  q_des[Q_RY] = 0;
  qd_des[Q_X] = args.cont[ARG_VX]*cos(q[Q_RZ]) - args.cont[ARG_VY]*sin(q[Q_RZ]);
  qd_des[Q_Y] = args.cont[ARG_VX]*sin(q[Q_RZ]) + args.cont[ARG_VY]*cos(q[Q_RZ]);
  qd_des[Q_RZ] = args.cont[ARG_VRZ];
  memcpy(&c_s_des, c_s, sizeof(ContactState));
}

MatrixXd setpoint_jac(const StateVec &q,
                           const StateVec &qd,
                           const ContactState c_s,
                           const Args &args,
                           const double delta_t) const {
  //TODO this is wrong
  MatrixXd jac = MatrixXd::Zero((int)NUM_Q+NUM_Q, NUM_CONT_ARGS+1);
  jac(Q_Z, 0) = 1;
  jac(NUM_Q+Q_X, 1) = 0;
  jac(NUM_Q+Q_Y, 2) = 1;
  jac(NUM_Q+Q_RZ, 3) = 1;

  return jac;
}

bool in_sroa_underestimate(const StateVec &q,
                           const StateVec &qd,
                           const ContactState c_s,
                           const Args &args,
                           const double delta_t) const {
  // disable the safe certificate
  // if (!in_safe_set(q,qd,c_s,args,delta_t)) {
  //   std::cout<<"The state is not safe." <<std::endl;    
  //   return false;
  // }
  double qd_x = qd[Q_X]*cos(q[Q_RZ]) - qd[Q_Y]*sin(q[Q_RZ]);
  double qd_y = qd[Q_X]*sin(q[Q_RZ]) + qd[Q_Y]*cos(q[Q_RZ]);


  Matrix3d R_curr;
  R_curr = AngleAxisd(q(Q_RX), Vector3d::UnitX()) *
           AngleAxisd(q(Q_RY), Vector3d::UnitY()) *
           AngleAxisd(q(Q_RZ), Vector3d::UnitZ()); 

  double z_ang = acos(R_curr.col(2).transpose()*Vector3d::UnitZ());


  double relative_z = 0;
  bool relative_z_valid = contact_feet_relative_z(q, c_s, relative_z);

  if (fabs(relative_z - args.cont[ARG_H]) < 0.05 &&
      fabs(qd[Q_Z]) < 0.5) { //&&
      //fabs(z_ang) < 0.05) {
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
    std::cout<<"The state is not safe." <<std::endl;
    return false;
  }
  double qd_x = qd[Q_X]*cos(q[Q_RZ]) - qd[Q_Y]*sin(q[Q_RZ]);
  double qd_y = qd[Q_X]*sin(q[Q_RZ]) + qd[Q_Y]*cos(q[Q_RZ]);


  Matrix3d R_curr;
  R_curr = AngleAxisd(q(Q_RX), Vector3d::UnitX()) *
           AngleAxisd(q(Q_RY), Vector3d::UnitY()) *
           AngleAxisd(q(Q_RZ), Vector3d::UnitZ()); 

  double z_ang = acos(R_curr.col(2).transpose()*Vector3d::UnitZ());

  double relative_z = 0;
  bool relative_z_valid = contact_feet_relative_z(q, c_s, relative_z);
  // std::cout<< "relative z valid: " <<relative_z_valid <<std::endl;

  if (fabs(relative_z - args.cont[ARG_H]) < 0.05 &&
      fabs(qd[Q_Z]) < 0.5) { //&&
      //fabs(z_ang) < 0.05) {
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
    std::cout<<"vel safe:"<<in_vel_limits(qd)<<", pos safe:" << in_pos_limits(q) <<std::endl;
    return false;
  }

  // diagonal feet should be in contact
  // if ((c_s[0] && c_s[3]) || (c_s[1] && c_s[2])) {
  // //if (c_s[0] || c_s[3] || c_s[1] || c_s[2]) {
  //   return true;
  // } else {
  //   std::cout<<"contact is unstable: "<< c_s[0]<< ","<< c_s[1]<< ","<<c_s[2]<< "," << c_s[3]<<std::endl;
  //   return false;
  // }
  return true;
}

private:

StateVec compute_body_q_des(const StateVec &q,
                            const StateVec &qd_des,
                            const Args &args) {
  StateVec q_des = VectorXd::Zero(NUM_Q);
  q_des[Q_Z] = args.cont[ARG_H];
  q_des[Q_X] = CTRL_LOOP_DURATION*qd_des[Q_X] + data.q_des_prev[Q_X];
  q_des[Q_Y] = CTRL_LOOP_DURATION*qd_des[Q_Y] + data.q_des_prev[Q_Y];
  q_des[Q_RZ] = CTRL_LOOP_DURATION*qd_des[Q_RZ] + data.q_des_prev[Q_RZ];

  data.q_des_prev = q_des;
  return q_des;
}

StateVec compute_body_qd_des(const StateVec &q,
                             const Args &args) {
  /* limit desired acceleration for smoothness */
  const int map_to_arg[3] = {ARG_VX, ARG_VY, ARG_VRZ};
  for (int i = 0; i < 3; ++i) {
    data.qd_arg_smoothed[i] = fmax(fmin(args.cont[map_to_arg[i]], data.qd_arg_smoothed[i]
                                                                  +qdd_des_limit[i]*CTRL_LOOP_DURATION),
                                                                  data.qd_arg_smoothed[i]
                                                                  -qdd_des_limit[i]*CTRL_LOOP_DURATION);
    data.qd_arg_smoothed[i] = fmax(fmin(data.qd_arg_smoothed[i], qd_des_limit[i]),
                                                                -qd_des_limit[i]);
  }
  StateVec qd_des = VectorXd::Zero(NUM_Q);
  qd_des[Q_X] = data.qd_arg_smoothed[0]*cos(q[Q_RZ]) - data.qd_arg_smoothed[1]*sin(q[Q_RZ]);
  qd_des[Q_Y] = data.qd_arg_smoothed[0]*sin(q[Q_RZ]) + data.qd_arg_smoothed[1]*cos(q[Q_RZ]);
  qd_des[Q_RZ] = data.qd_arg_smoothed[2];
  //qd_des[Q_X] = args.cont[ARG_VX]*cos(q[Q_RZ]) - args.cont[ARG_VY]*sin(q[Q_RZ]);
  //qd_des[Q_Y] = args.cont[ARG_VX]*sin(q[Q_RZ]) + args.cont[ARG_VY]*cos(q[Q_RZ]);
  //qd_des[Q_RZ] = args.cont[ARG_VRZ];

  return qd_des;
}

StateVec compute_body_qdd_des(const StateVec &q,
                              const StateVec &qd,
                              const StateVec &q_des,
                              const StateVec &qd_des) {
  double Ki = 0.00015;
  double antiwindup_x = 0.05;
  double antiwindup_y = 0.05;

  data.i_err[0] += Ki*(qd_des[Q_X] - qd[Q_X]);
  data.i_err[1] += Ki*(qd_des[Q_Y] - qd[Q_Y]);

  data.i_err[0] = fmin(antiwindup_x, fmax(-antiwindup_x, data.i_err[0]));
  data.i_err[1] = fmin(antiwindup_y, fmax(-antiwindup_y, data.i_err[1]));

  Vector3d e_diff = orientation_diff_from_euler(q.segment<3>(Q_RX), q_des.segment<3>(Q_RX));

  VectorXd joint_pos_err = VectorXd::Zero(NUM_U);
  VectorXd joint_vel_err = VectorXd::Zero(NUM_U);

  StateVec q_err = VectorXd::Zero(NUM_Q);
  StateVec qd_err = VectorXd::Zero(NUM_Q);

  q_err << q.head(3) - q_des.head(3),
           e_diff,
           VectorXd::Zero(NUM_U);

  //TODO i dont think this is correct
  qd_err << qd.head(6) - qd_des.head(6), 
            VectorXd::Zero(NUM_U);
  
  return data.Kp*(q_err) + data.Kd*(qd_err);
}

StateVec update_qd_filter(const StateVec &qd) {
  data.qd_buffer.pop_back();
  data.qd_buffer.push_front(qd);

  StateVec qd_filtered = VectorXd::Zero((int)NUM_Q, 1);
  for (int i = 0; i < qd_buff_size; ++i) {
    qd_filtered += data.qd_buffer.at(i)/qd_buff_size;
  }
  return qd_filtered;
}

void fill_act_cmds(Data &data, ActuatorCmds &act_cmds, const bool feasible, const StateVec &q, const StateVec &qd, const InputVec &u_sol, const StateVec &qdd_sol) {
  if (feasible) {
    double gamma = 0.5;
    InputVec qd_fuse = (1-gamma)*qd.tail(NUM_U) + gamma*data.qd_cmd_prev;
    for (int i = 0; i < NUM_U; ++i) {
      act_cmds.mode[i] = CMD_MODE_TAU_VEL_POS;
      act_cmds.u[i] = u_sol[i];
      act_cmds.qd[i] = qd_fuse[i] + CTRL_LOOP_DURATION*(qdd_sol[i+Q_FL1]);
      act_cmds.q[i] = q[i+Q_FL1] + CTRL_LOOP_DURATION*qd_fuse[i]+0.5*CTRL_LOOP_DURATION*CTRL_LOOP_DURATION*(qdd_sol[i+Q_FL1]);
      act_cmds.kp[i] = 200;
      act_cmds.kd[i] = 5;
    }
    data.qd_cmd_prev = act_cmds.qd;

    for (int i = 0; i < NUM_C; ++i) {
      if (data.swing_legs[i]) {
        for (int j = 0; j < 3; ++j) {
          act_cmds.mode[3*i+j] = CMD_MODE_TAU_VEL_POS;
          act_cmds.u[3*i+j] = u_sol[3*i+j];
          act_cmds.qd[3*i+j] = data.swing_leg_qd_des[i][j];
          act_cmds.q[3*i+j] =  data.swing_leg_q_des[i][j];
          act_cmds.kp[3*i+j] = 200;
          act_cmds.kd[3*i+j] = 5;
        }
      }
    }
  } else {
    for (int i = 0; i < NUM_U; ++i) {
      act_cmds.mode[i] = CMD_MODE_TAU_VEL;
      act_cmds.u[i] = 0;
      act_cmds.qd[i] = 0;
      act_cmds.q[i] = 0;
      act_cmds.kp[i] = 200;
      act_cmds.kd[i] = 5;
    }
    data.qd_cmd_prev = act_cmds.qd;
  }

}

bool solve_idqp(Data &data, const StateVec &q, const StateVec &qd, const ContactState c_s, const StateVec &qdd_des, InputVec &u_sol, StateVec &qdd_sol, Eigen::VectorXd &fc_sol) {

  MatrixXd D = robot_D(q);
  MatrixXd H = robot_H(q,qd);

  MatrixXd J = MatrixXd::Zero(3*NUM_C, (int)NUM_Q);
  MatrixXd Jd = MatrixXd::Zero(3*NUM_C, (int)NUM_Q);
  //TODO not like this, constrain F to be zero instead
  //but it should get the same solution
  for (int i = 0; i < NUM_C; ++i) {
    if (!data.swing_legs[i]) {// && c_s[i]) {
      J.block(3*i,0,3,(int)NUM_Q) << fk_jac(q,robot.contacts[i].frame);
      Jd.block(3*i,0,3,(int)NUM_Q) << fk_jac_dot(q,qd,robot.contacts[i].frame);
    }
  }

  for (int i = 0; i < NUM_C; ++i) {
    if (data.swing_legs[i]) {// && c_s[i]) {
      for (int j = NUM_U+6+3*i; j < NUM_U+6+3*i+3; ++j) {
        data.P.coeffRef(j,j) = 1;
        data.P.coeffRef(j,j) = 1;
        data.P.coeffRef(j,j) = 1;
      }
    } else {
      for (int j = NUM_U+6+3*i; j < NUM_U+6+3*i+3; ++j) {
        data.P.coeffRef(j,j) = 1e-5;
        data.P.coeffRef(j,j) = 1e-5;
        data.P.coeffRef(j,j) = 1e-5;
      }
    }
  }
  data.solver.updateHessianMatrix(data.P);

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

  data.lb << data.input_lb, dynamics_lb, contact_lb, data.force_lb;
  data.ub << data.input_ub, dynamics_ub, contact_ub, data.force_ub;

  data.solver.updateLinearConstraintsMatrix(data.A);
  data.solver.updateBounds(data.lb, data.ub);
  bool success = data.solver.solve();
  VectorXd sol = data.solver.getSolution();

  u_sol = sol.head(NUM_U);
  qdd_sol = sol.segment<NUM_Q>(NUM_U) + qdd_des;

  fc_sol = sol.tail(3*NUM_C);


  // std::cout << "qdd_res: " << std::endl << qdd_sol.transpose() << std::endl << qdd_des.transpose() << std::endl;
  // std::cout << "u_res: " << std::endl << u_sol.transpose() << std::endl;
  // std::cout <<"f_c: "<< fc_sol <<std::endl;

  return success;
}

void gait_state_machine(double t, const StateVec &q, const ContactState c_s, bool swing_legs[4], double foot_pos_init[4][3]) {
  double duration = 2*(dwell_time+step_time);
  double t_phase = fmod(t,duration);
  double dwell_1 = t_phase < dwell_time;
  double swing_1 = dwell_time <= t_phase && t_phase < step_time + dwell_time;
  double early_contact_1 = dwell_time + step_time/2 <= t_phase && t_phase < step_time + dwell_time;
  double dwell_2 = step_time + dwell_time <= t_phase && t_phase < step_time + 2*dwell_time;
  double swing_2 = step_time + 2*dwell_time <= t_phase && t_phase < 2*step_time + 2*dwell_time;
  double early_contact_2 = step_time + 2*dwell_time + step_time/2 <= t_phase && t_phase < 2*step_time + 2*dwell_time;

  if (dwell_1 || dwell_2) {
    for (int i = 0; i < 4; ++i) {
      fk(q, foot_pos_init[i], (Frame)(i+F_FL_EE));
      swing_legs[i]=false;
    }
  } else if (swing_1) {
    if (early_contact_1) {
      swing_legs[0] &= !c_s[0];
      swing_legs[3] &= !c_s[3];
    } else {
      swing_legs[0] = true;
      swing_legs[3] = true;
    }
  } else if (swing_2) {
    if (early_contact_2) {
      swing_legs[1] &= !c_s[1];
      swing_legs[2] &= !c_s[2];
    } else {
      swing_legs[1] = true;
      swing_legs[2] = true;
    }
  }
}

void compute_swing_leg_joint_des(Data &data, Contact foot, const StateVec &q, const StateVec &qd, const StateVec &qd_filtered, const StateVec &qd_des) {
  double foot_pos_des[3];
  double foot_vel_des[3];
  double foot_acc_des[3];
  double neutral_pos[3];
  double goal_pos[3];

  StateVec q_neutral = VectorXd::Zero(NUM_Q);
  //TODO a little weird defining neutral like this (legs straight except for hip)
  q_neutral << q.head(3),0,0,q[Q_RZ],
               0.1,0,0,
              -0.1,0,0,
               0.1,0,0,
              -0.1,0,0;
  q_neutral[Q_X] -= data.i_err[0];
  q_neutral[Q_Y] -= data.i_err[1];
  fk(q_neutral, neutral_pos, robot.contacts[foot].frame);
  
  //TODO this is technically wrong, since t_phase can go negative,
  //but because of the way gait_state_machine works this function 
  //never gets called with such a t. Probably should clean up a bit.
  double t_phase = 1/step_time*(fmod(data.t, step_time+dwell_time)-dwell_time);

  neutral_pos[2] = 0.075*(1-2*fabs(t_phase-0.5));

  double neutral_ang = atan2(neutral_pos[1]-q[Q_Y],neutral_pos[0]-q[Q_X]);
  double r = sqrt((neutral_pos[1]-q[Q_Y])*(neutral_pos[1]-q[Q_Y])+
                  (neutral_pos[0]-q[Q_X])*(neutral_pos[0]-q[Q_X]));

  double rz_delta = 0.5*0.5*(step_time+dwell_time)*(qd_filtered[Q_RZ]+qd_des[Q_RZ]);
  double th = rz_delta + neutral_ang;
  double x_delta_from_ang = r*cos(th)+q[Q_X]-neutral_pos[0];
  double y_delta_from_ang = r*sin(th)+q[Q_Y]-neutral_pos[1];

  //TODO unclear which heuristic is actually better, they are very similiar numerically
  double x_delta = 0.5*0.5*(step_time+dwell_time)*(qd_filtered[Q_X]+qd_des[Q_X]);
  double y_delta = 0.5*0.5*(step_time+dwell_time)*(qd_filtered[Q_Y]+qd_des[Q_Y]);
  //TODO be careful with q[Q_Z], really need height above ground plane
  //double x_delta = 0.5*(step_time+dwell_time)*qd_des[Q_X] +
  //                 sqrt(q[Q_Z]/9.81)*(qd_filtered[Q_X]-qd_des[Q_X]);
  //double y_delta = 0.5*(step_time+dwell_time)*qd_des[Q_Y] +
  //                 sqrt(q[Q_Z]/9.81)*(qd_filtered[Q_Y]-qd_des[Q_Y]);

  //x_delta = fmax(fmin(x_delta,0.1),-0.1);
  //y_delta = fmax(fmin(y_delta,0.1),-0.1);

  x_delta += x_delta_from_ang;
  y_delta += y_delta_from_ang;
  
  //CoM offset. TODO: pull from model
  //neutral_pos[0] += 0.012731/2;
  //neutral_pos[1] += 0.002186/2;
  
  goal_pos[0] = neutral_pos[0]+x_delta;
  goal_pos[1] = neutral_pos[1]+y_delta;
  goal_pos[2] = neutral_pos[2];
  
  traj_gen::CubicPoly swing_traj[3];
  
  for (int i = 0; i < 2; ++i) {
    swing_traj[i] = traj_gen::cubic_poly_gen(data.foot_pos_init[foot][i],0,goal_pos[i],0,step_time*0.8);
    traj_gen::cubic_poly_eval_full(swing_traj[i],
                                   step_time*t_phase,
                                   foot_pos_des[i],
                                   foot_vel_des[i],
                                   foot_acc_des[i]);
  }

  if (t_phase < 0.5) {
    swing_traj[2] = traj_gen::cubic_poly_gen(0,0,step_height,0,step_time*0.5);
    traj_gen::cubic_poly_eval_full(swing_traj[2],
                                   step_time*t_phase,
                                   foot_pos_des[2],
                                   foot_vel_des[2],
                                   foot_acc_des[2]);
  } else {
    swing_traj[2] = traj_gen::cubic_poly_gen(step_height,0,0,0,step_time*0.5);
    traj_gen::cubic_poly_eval_full(swing_traj[2],
                                   step_time*(t_phase-0.5),
                                   foot_pos_des[2],
                                   foot_vel_des[2],
                                   foot_acc_des[2]);
  }

  double foot_pos[3];
  fk(q, foot_pos, robot.contacts[foot].frame);
  for (int i = 0; i < 3; ++i) {
    walk_tlm.swing_feet_err[3*foot+i] = foot_pos[i] - foot_pos_des[i];
  }
  
  ik(robot.contacts[foot].frame, q, foot_pos_des, data.swing_leg_q_des[foot]);

  MatrixXd J_swing = fk_jac(q, robot.contacts[foot].frame);

  VectorXd foot_vel_des_vec = VectorXd::Zero(3);
  VectorXd foot_acc_des_vec = VectorXd::Zero(3);
  for (int i = 0; i < 3; ++i) {
    foot_vel_des_vec[i] = foot_vel_des[i];
    foot_acc_des_vec[i] = foot_acc_des[i];
  }

  //TODO this give velocity w.r.t body frame, not world frame
  //either this or position needs to change
  VectorXd swing_leg_qd_des = J_swing.transpose()*foot_vel_des_vec;
  VectorXd swing_leg_qdd_des = J_swing.transpose()*foot_acc_des_vec;

  for (int i = 0; i < 3; ++i) {
    data.swing_leg_qd_des[foot][i] = swing_leg_qd_des[i];
    data.swing_leg_qdd_des[foot][i] = swing_leg_qdd_des[i];
  }
}

void nearest_phasing(Data &data, const StateVec &q, const StateVec &qd, const ContactState c_s, double foot_pos_init[4][3]) {
  double FL_BR_z = 0.5*(foot_pos_init[0][2] + foot_pos_init[3][2]);
  double FR_BL_z = 0.5*(foot_pos_init[1][2] + foot_pos_init[2][2]);
  
  VectorXd foot_vel = VectorXd::Zero(3);
  double z_vel = 0;
  double z_pos = 0;

  double t = dwell_time;
  if (fabs(FL_BR_z) > fabs(FR_BL_z)) {
    foot_vel = (fk_jac(q, F_FL_EE) + fk_jac(q, F_BR_EE))*qd;
    z_pos = FL_BR_z;
  } else {
    foot_vel = (fk_jac(q, F_FR_EE) + fk_jac(q, F_BL_EE))*qd;
    z_pos = FR_BL_z;
    t += step_time+dwell_time;
  }
  z_vel = 0.5*foot_vel[2];

  traj_gen::CubicPoly swing_traj;

  if (z_vel > 0) {
    swing_traj = traj_gen::cubic_poly_gen(0,0,step_height,0,step_time*0.5);
  } else {
    swing_traj = traj_gen::cubic_poly_gen(step_height,0,0,0,step_time*0.5);
    t += step_time*0.5;
  }

  double min_t=0;
  double min_dist=1e9;
  double z_pos_traj;
  double z_vel_traj;
  for (double i=0; i<0.5*step_time; i+=0.01) {
    traj_gen::cubic_poly_eval(swing_traj, i, z_pos_traj, z_vel_traj);
    if (min_dist > fabs(z_pos-z_pos_traj)) {
      min_dist = fabs(z_pos-z_pos_traj);
      min_t = i;
    }
  }

  data.t = min_t+t;

}

};

PrimitiveBehavior* create() {
  return new WalkIdqp;
}
void destroy(PrimitiveBehavior* prim) {
  delete prim;
}

}
