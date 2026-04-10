#include <stdio.h>
#include <unistd.h>

#include "ik.h"
#include "fk.h"
#include "pinocchio/multibody/data.hpp"

extern Robot robot;

#include "ctrl/walk_idqp/walk_idqp_args.h"
#include "ctrl_mode_graph_core.h"

bool fk_ik_comparison_test();

void graph_search_test();

int main() {

  printf("Running Unit Tests\n");
  
  //printf("Forward/Inverse Kinematics Comparision: %s\n", fk_ik_comparison_test() ? "PASS" : "FAIL");
  graph_search_test();

}

void graph_search_test() {
  CtrlState ctrl_next;
  CtrlState ctrl_des;
  ctrl_des.mode = C_WALK_IDQP;
  ctrl_des.args.cont[walk_idqp::ARG_H] = 0.25;
  StateVec q = VectorXd::Zero((int)NUM_Q);
  StateVec qd = VectorXd::Zero((int)NUM_Q);
  ContactState c_s = {1,1,1,1};
  //q << 0,    0, 0.12,
  //     0,    0,    0,
  //     0, 1.57, -2.7,
  //     0, 1.57, -2.7,
  //     0, 1.57, -2.7,
  //     0, 1.57, -2.7;
  q << 0,    0, -0.4,
       0,    0,    0,
       0, 3.14, 0,
       0, 3.14, 0,
       0, 3.14, 0,
       0, 3.14, 0;
  print_nodes(ctrl_mode_graph_search(q, qd, c_s, ctrl_des));
}

bool fk_ik_comparison_test() { 
  robot_init();
  bool pass = true;
  
  double q[3];
  double p[3];
  double p2[3];
  StateVec x = {};
  StateVec x2 = {};

  x[Q_X] = 0.3;
  x2[Q_X] = 0.3;
  x[Q_Y] = 0.3;
  x2[Q_Y] = 0.3;
  x[Q_Z] = 0.3;
  x2[Q_Z] = 0.3;

  double step = 0.3;

  for (double rx = -M_PI; rx < M_PI; rx+=step) {
    x[Q_RX] = rx;
    x2[Q_RX] = rx;
    for (double ry = -M_PI; ry < M_PI; ry+=step) {
      x[Q_RY] = ry;
      x2[Q_RY] = ry;
      for (double rz = -M_PI; rz < M_PI; rz+=step) {
        x[Q_RZ] = rz;
        x2[Q_RZ] = rz;
        for (double q1 = -0.802; q1 < 0.802; q1+=step) {
          x[Q_FL1] = q1;
          x[Q_FR1] = q1;
          x[Q_BL1] = q1;
          x[Q_BR1] = q1;
          for (double q2 = -1.05; q2 < 4.19; q2+=step) {
            x[Q_FL2] = q2;
            x[Q_FR2] = q2;
            x[Q_BL2] = q2;
            x[Q_BR2] = q2;
            for (double q3 = -2.7; q3 < -0.916; q3+=step) {
              x[Q_FL3] = q3;
              x[Q_FR3] = q3;
              x[Q_BL3] = q3;
              x[Q_BR3] = q3;
              for (int frame = F_FL_EE; frame  < F_BR_EE; ++frame) {
                fk(x, p, (Frame)frame);
                ik((Frame)frame, x, p, q);
                switch (frame) {
                  case F_FL_EE:
                    x2[Q_FL1] = q[0];
                    x2[Q_FL2] = q[1];
                    x2[Q_FL3] = q[2];
                    break;
                  case F_FR_EE:
                    x2[Q_FR1] = q[0];
                    x2[Q_FR2] = q[1];
                    x2[Q_FR3] = q[2];
                    break;
                  case F_BL_EE:
                    x2[Q_BL1] = q[0];
                    x2[Q_BL2] = q[1];
                    x2[Q_BL3] = q[2];
                    break;
                  case F_BR_EE:
                    x2[Q_BR1] = q[0];
                    x2[Q_BR2] = q[1];
                    x2[Q_BR3] = q[2];
                    break;
                  default:
                    break;
                }
                fk(x2, p2, (Frame)frame);
                if (fabs(p[0]-p2[0]) + fabs(p[1]-p2[1]) + fabs(p[2]-p2[2]) > 1e-4) {
                  printf("Frame: %d ", frame);
                  printf("q: %f %f %f ", q1, q2, q3);
                  printf("q_ik: %f %f %f ", q[0], q[1], q[2]);
                  printf("q_err: %f %f %f ", q[0]-q1, q[1]-q2, q[2]-q3);
                  printf("p: %f %f %f ", p[0], p[1], p[2]);
                  printf("p2: %f %f %f ", p2[0], p2[1], p2[2]);
                  printf("pos_err: %f %f %f \n", p[0]-p2[0], p[1]-p2[1], p[2]-p2[2]);
                  pass = false;
                }
              }
            }
          } 
        }
      }
    }
  }
  return pass;
}



