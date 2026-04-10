#ifndef TRAJ_GEN_H
#define TRAJ_GEN_H

namespace traj_gen {

typedef struct {
  double c[4];
  double max_duration;

} CubicPoly;

CubicPoly cubic_poly_gen(double q, double qd, double q_des, double qd_des, double duration);
bool cubic_poly_eval(CubicPoly cubic, double t, double &q, double &qd);
bool cubic_poly_eval_full(CubicPoly cubic, double t, double &q, double &qd, double &qdd);

}

#endif //TRAJ_GEN_H
