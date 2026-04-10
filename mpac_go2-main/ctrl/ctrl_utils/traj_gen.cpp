#include <math.h>
#include "traj_gen.h"

namespace traj_gen {

/* q  = c0 + c1*t + c2*t^2 + c3*t^3
 * qd = c1 + 2*c2*t + 3*c3*t^2 */
CubicPoly cubic_poly_gen(double q, double qd, double q_des, double qd_des, double duration_des) {
  CubicPoly cubic = {};
  double duration = fmax(1e-3,duration_des);
  cubic.max_duration = duration;
 
  /* initial condition @ t=0 */
  cubic.c[0] = q;
  cubic.c[1] = qd;

  /* final condition @ t=duration
   *   c2*duration^2 +   c3*duration^3  = q_des - c0 - c1*duration
   * 2*c2*duration   + 3*c3*duration^2  = qd_des - c1
   * =>  2*c2*duration^2   + 2*c3*duration^3  = 2*(q_des - c0 - c1*duration)
   *     2*c2*duration^2   + 3*c3*duration^3  = duration*(qd_des - c1)
   * =>  c3*duration^3 = duration*(qd_des - c1) - 2*(q_des - c0 - c1*duration)
   *                   = duration*qd_des - duration*c1 - 2*(q_des - c0) + 2*c1*duration
   *                   = duration*qd_des - 2*(q_des - c0) + duration*c1
   *                   = duration*(qd_des + c1) - 2*(q_des - c0)
   * =>  c3 = (1/duration^3)*(duration*(qd_des + c1) - 2*(q_des - c0))
   *        = 1/duration^2*(qd_des + c1) - 2/duration^3*(q_des - c0)
   * =>  c2 = -(3/2)c3*duration + 1/(2*duration)*(qd_des - c1) */
  cubic.c[3] =   1/(duration*duration)*(qd_des + cubic.c[1])
               - 2/(duration*duration*duration)*(q_des - cubic.c[0]);
  cubic.c[2] = -1.5*cubic.c[3]*duration + 0.5/duration*(qd_des - cubic.c[1]);
  return cubic;
}

bool cubic_poly_eval_full(CubicPoly cubic, double t_in, double &q, double &qd, double &qdd) {
  double t = fmin(t_in, cubic.max_duration); 
  q  = cubic.c[0] + cubic.c[1]*t + cubic.c[2]*t*t + cubic.c[3]*t*t*t;
  qd = cubic.c[1] + 2*cubic.c[2]*t + 3*cubic.c[3]*t*t;
  qdd = 2*cubic.c[2] + 6*cubic.c[3]*t;

  if (t_in >= cubic.max_duration) {
    return true;
  }
  return false;
}

bool cubic_poly_eval(CubicPoly cubic, double t_in, double &q, double &qd) {
  double qdd;
  return cubic_poly_eval_full(cubic, t_in, q, qd, qdd);
}


}
