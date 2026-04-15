#include "robot.h"
#include <eigen3/Eigen/Dense>
#include <iostream>

#include "ctrl/ctrl_utils/joint_pos_pd.h"
#include "walk_traj.h"
#include "walk_traj_args.h"
#include "ctrl_core.h"
#include "math_utils.h"
#include "ctrl_mode_class.h"

namespace walk_traj {

typedef struct {
  double t;
  bool mirror;
  int rdy_for_transition;
  joint_pos_pd::JointPosPD jppd;
  BezierCurve walk_b[6];

} Data;

// in place
static const BezierCurve walk_0[] =  {
  {4, 0.2,       0,        0,        0,        0,        0},
  {4, 0.2,  0.8958,   1.1457,   1.3373,   1.1457,   0.8958},
  {4, 0.2, -1.7916,  -2.2915,  -2.6746,  -2.2915,  -1.7916},
  {4, 0.2,       0,        0,        0,        0,        0},
  {4, 0.2,  0.8957,   0.8957,   0.8957,   0.8957,   0.8957},
  {4, 0.2, -1.7913,  -1.7913,  -1.7913,  -1.7913,  -1.7913},
};

// slow
static const BezierCurve walk_1[] =  {
  {4, 0.2,        0,        0,        0,        0,        0},
  {4, 0.2,   0.9825,   1.3242,   1.3473,   0.9536,   0.8017},
  {4, 0.2,  -1.7842,  -2.2777,  -2.6946,  -2.2777,  -1.7842},
  {4, 0.2,        0,        0,        0,        0,        0},
  {4, 0.2,   0.7920,   0.8455,   0.8970,   0.9458,   0.9913},
  {4, 0.2,  -1.7833,  -1.7913,  -1.7940,  -1.7913,  -1.7833},
};

// medium
static const BezierCurve walk_2[] =  {
  {4, 0.2,       0,        0,        0,        0,        0},
  {4, 0.2,  1.0594,   1.4864,   1.3770,   0.7507,   0.7024},
  {4, 0.2, -1.7618,  -2.2370,  -2.7539,  -2.2370,  -1.7618},
  {4, 0.2,       0,        0,        0,        0,        0},
  {4, 0.2,  0.6823,   0.7943,   0.9011,   0.9969,   1.0771},
  {4, 0.2, -1.7594,  -1.7912,  -1.8021,  -1.7912,  -1.7594},
};

//fast
static const BezierCurve walk_4[] =  {
  {4, 0.2,       0,        0,        0,        0,        0 },
  {4, 0.2,  1.1246,   1.6306,   1.4253,   0.5401,   0.6002 },
  {4, 0.2, -1.7248,  -2.1707,  -2.8506,  -2.1707,  -1.7248 },
  {4, 0.2,       0,        0,        0,        0,        0 },
  {4, 0.2,  0.5685,   0.7412,   0.9079,   1.0496,   1.1513 },
  {4, 0.2, -1.7198,  -1.7908,  -1.8158,  -1.7908,  -1.7198 },
};

//faster
static const BezierCurve walk_3[] =  {
  {4, 0.2,       0,        0,        0,        0,        0},
  {4, 0.2,  1.1768,   1.7560,   1.4909,   0.3249,   0.4964},
  {4, 0.2, -1.6732,  -2.0809,  -2.9818,  -2.0809,  -1.6732},
  {4, 0.2,       0,        0,        0,        0,        0},
  {4, 0.2,  0.4520,   0.6856,   0.9176,   1.1044,   1.2127},
  {4, 0.2, -1.6647,  -1.7900,  -1.8353,  -1.7900,  -1.6647},
};

class WalkTraj: public PrimitiveBehavior {

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
  switch(args.disc[ARG_GAIT]) {
  case 0:
    memcpy(data.walk_b, walk_0, sizeof(walk_0));
    break;
  case 1:
    memcpy(data.walk_b, walk_1, sizeof(walk_1));
    break;
  case 2:
    memcpy(data.walk_b, walk_2, sizeof(walk_2));
    break;
  case 3:
    memcpy(data.walk_b, walk_3, sizeof(walk_3));
    break;
  default:
    memcpy(data.walk_b, walk_0, sizeof(walk_0));
    break;
  }

  data.mirror = true;
  InputVec q_des;
  InputVec qd_des;
  data.t = 0.1;
  q_des << bezier_interp(&data.walk_b[0], data.t),bezier_interp(&data.walk_b[1], data.t),bezier_interp(&data.walk_b[2], data.t),
           bezier_interp(&data.walk_b[3], data.t),bezier_interp(&data.walk_b[4], data.t),bezier_interp(&data.walk_b[5], data.t),
           bezier_interp(&data.walk_b[3], data.t),bezier_interp(&data.walk_b[4], data.t),bezier_interp(&data.walk_b[5], data.t),
           bezier_interp(&data.walk_b[0], data.t),bezier_interp(&data.walk_b[1], data.t),bezier_interp(&data.walk_b[2], data.t);
  qd_des << bezier_interp_dt(&data.walk_b[0], data.t),bezier_interp_dt(&data.walk_b[1], data.t),bezier_interp_dt(&data.walk_b[2], data.t),
            bezier_interp_dt(&data.walk_b[3], data.t),bezier_interp_dt(&data.walk_b[4], data.t),bezier_interp_dt(&data.walk_b[5], data.t),
            bezier_interp_dt(&data.walk_b[3], data.t),bezier_interp_dt(&data.walk_b[4], data.t),bezier_interp_dt(&data.walk_b[5], data.t),
            bezier_interp_dt(&data.walk_b[0], data.t),bezier_interp_dt(&data.walk_b[1], data.t),bezier_interp_dt(&data.walk_b[2], data.t);
  double norm_dist = (q.tail(NUM_U) - q_des).norm();
  double duration;
  if (norm_dist > 0.6) {
    duration = 0.12*norm_dist;
  } else {
    duration = 0.05;
  }
  joint_pos_pd::init(data.jppd, q.tail(NUM_U), qd.tail(NUM_U), q_des, qd_des, duration);
  data.rdy_for_transition = 0;
}

void execute(const StateVec &q,
             const StateVec &qd,
             const ContactState c_s,
             const Args &args, 
             ActuatorCmds &act_cmds) {
  if (!joint_pos_pd::execute(data.jppd, act_cmds, q, qd)) {
    return;
  }

  if (data.t > data.walk_b[0].duration) {
    data.mirror = !data.mirror;
    data.t = 0;
    data.rdy_for_transition += 1;
  }
  data.t += CTRL_LOOP_DURATION;
  double left[]  = {bezier_interp(&data.walk_b[0], data.t),
                    bezier_interp(&data.walk_b[1], data.t),
                    bezier_interp(&data.walk_b[2], data.t)};
  double right[] = {bezier_interp(&data.walk_b[3], data.t),
                    bezier_interp(&data.walk_b[4], data.t),
                    bezier_interp(&data.walk_b[5], data.t)};
  InputVec q_des;

  if (data.mirror) {
    memcpy(&q_des[0],  left, 3*sizeof(double));
    memcpy(&q_des[3], right, 3*sizeof(double));
    left[0] *= -1;
    right[0] *= -1;
    memcpy(&q_des[6], right, 3*sizeof(double));
    memcpy(&q_des[9],  left, 3*sizeof(double));
  } else {
    memcpy(&q_des[6],  left, 3*sizeof(double));
    memcpy(&q_des[9], right, 3*sizeof(double));
    left[0] *= -1;
    right[0] *= -1;
    memcpy(&q_des[0], right, 3*sizeof(double));
    memcpy(&q_des[3],  left, 3*sizeof(double));
  }

  for (int i = 0; i < NUM_U; ++i) {
    act_cmds.mode[i] = CMD_MODE_TAU_VEL_POS;
    act_cmds.u[i] = 0;
    act_cmds.q[i] = q_des[i];
    act_cmds.qd[i] = 0;//qd_cmd;
    act_cmds.kp[i] = 200;
    act_cmds.kd[i] = 15;
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
  return MatrixXd::Zero((int)NUM_Q+NUM_Q, 1);
}

bool in_sroa_underestimate(const StateVec &q,
                      const StateVec &qd,
                      const ContactState c_s,
                      const Args &args,
                      const double delta_t) const {
  return in_sroa_overestimate(q, qd, c_s, args, delta_t);
}
bool in_sroa_overestimate(const StateVec &q,
                          const StateVec &qd,
                          const ContactState c_s,
                          const Args &args,
                          const double delta_t) const {
  if (!in_safe_set(q,qd,c_s,args,delta_t)) {
    return false;
  }

  BezierCurve walk_c[6];
  double qd_x_des = 0;
  switch(args.disc[ARG_GAIT]) {
  case 0:
    memcpy(walk_c, walk_0, sizeof(walk_0));
    qd_x_des = 0;
    break;
  case 1:
    memcpy(walk_c, walk_1, sizeof(walk_1));
    qd_x_des = 0.2;
    break;
  case 2:
    memcpy(walk_c, walk_2, sizeof(walk_2));
    qd_x_des = 0.3;
    break;
  case 3:
    memcpy(walk_c, walk_3, sizeof(walk_3));
    qd_x_des = 0.4;
    break;
  default:
    memcpy(walk_c, walk_0, sizeof(walk_0));
    break;
  }

  double a = 0.1;
  InputVec q_des;
  q_des << bezier_interp(&walk_c[0], a),bezier_interp(&walk_c[1], a),bezier_interp(&walk_c[2], a),
           bezier_interp(&walk_c[3], a),bezier_interp(&walk_c[4], a),bezier_interp(&walk_c[5], a),
           bezier_interp(&walk_c[3], a),bezier_interp(&walk_c[4], a),bezier_interp(&walk_c[5], a),
           bezier_interp(&walk_c[0], a),bezier_interp(&walk_c[1], a),bezier_interp(&walk_c[2], a);

  double norm = (q.tail(NUM_U)-q_des).squaredNorm();
  //return (norm < 0.3);

  double qd_x = qd[Q_X]*cos(q[Q_RZ]) - qd[Q_Y]*sin(q[Q_RZ]);
  double qd_y = qd[Q_X]*sin(q[Q_RZ]) + qd[Q_Y]*cos(q[Q_RZ]);
  if (fabs(q[Q_Z] - 0.25) < 0.05 &&
      fabs(qd_x - qd_x_des) < 0.5 &&
      fabs(qd_y) < 0.5)
      {
    return true;
  }
  return false;
}
bool in_safe_set(const StateVec &q,
                 const StateVec &qd,
                 const ContactState c_s,
                 const Args &args,
                 const double delta_t) const {
  return in_vel_limits(qd) && in_pos_limits(q);
}

};

PrimitiveBehavior* create() {
  return new WalkTraj;
}
void destroy(PrimitiveBehavior* prim) {
  delete prim;
}

}
