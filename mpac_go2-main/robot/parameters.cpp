#include "robot.h"
#include <math.h>

void robot_params_init(Robot &r) {
  //                       parent, child,     type,   act,a_x, a_y, a_z
  set_morph_node(r, Q_X,   L_ORIGIN, L_X, PRISMATIC, U_NONE, 1,   0,   0);
  set_morph_node(r, Q_Y,   L_X,      L_Y, PRISMATIC, U_NONE, 0,   1,   0);
  set_morph_node(r, Q_Z,   L_Y,      L_Z, PRISMATIC, U_NONE, 0,   0,   1);
  set_morph_node(r, Q_RX,  L_Z,     L_RX,  REVOLUTE, U_NONE, 1,   0,   0);
  set_morph_node(r, Q_RY,  L_RX,    L_RY,  REVOLUTE, U_NONE, 0,   1,   0);
  set_morph_node(r, Q_RZ,  L_RY,  L_BODY,  REVOLUTE, U_NONE, 0,   0,   1);
  set_morph_node(r, Q_FL1, L_BODY, L_FL1,  REVOLUTE,  U_FL1, 1,   0,   0);
  set_morph_node(r, Q_FL2, L_FL1,  L_FL2,  REVOLUTE,  U_FL2, 0,   1,   0);
  set_morph_node(r, Q_FL3, L_FL2,  L_FL3,  REVOLUTE,  U_FL3, 0,   1,   0);
  set_morph_node(r, Q_FR1, L_BODY, L_FR1,  REVOLUTE,  U_FR1, 1,   0,   0);
  set_morph_node(r, Q_FR2, L_FR1,  L_FR2,  REVOLUTE,  U_FR2, 0,   1,   0);
  set_morph_node(r, Q_FR3, L_FR2,  L_FR3,  REVOLUTE,  U_FR3, 0,   1,   0);
  set_morph_node(r, Q_BL1, L_BODY, L_BL1,  REVOLUTE,  U_BL1, 1,   0,   0);
  set_morph_node(r, Q_BL2, L_BL1,  L_BL2,  REVOLUTE,  U_BL2, 0,   1,   0);
  set_morph_node(r, Q_BL3, L_BL2,  L_BL3,  REVOLUTE,  U_BL3, 0,   1,   0);
  set_morph_node(r, Q_BR1, L_BODY, L_BR1,  REVOLUTE,  U_BR1, 1,   0,   0);
  set_morph_node(r, Q_BR2, L_BR1,  L_BR2,  REVOLUTE,  U_BR2, 0,   1,   0);
  set_morph_node(r, Q_BR3, L_BR2,  L_BR3,  REVOLUTE,  U_BR3, 0,   1,   0);

  
  //                                   l_x,     l_y,  l_z,   b, j, cpt, upper_pos, lower_pos, vel, effort
  set_joint_params(r, Q_X,  "Q_X",        0,        0,      0,   0, 0, 0,    INFINITY, -INFINITY,   10,   0);
  set_joint_params(r, Q_Y,  "Q_Y",        0,        0,      0,   0, 0, 0,    INFINITY, -INFINITY,   10,   0);
  set_joint_params(r, Q_Z,  "Q_Z",        0,        0,      0,   0, 0, 0,    INFINITY, -INFINITY,   10,   0);
  set_joint_params(r, Q_RX, "Q_RX",       0,        0,      0,   0, 0, 0,    INFINITY, -INFINITY,   10,   0);
  set_joint_params(r, Q_RY, "Q_RY",       0,        0,      0,   0, 0, 0,    INFINITY, -INFINITY,   10,   0);
  set_joint_params(r, Q_RZ, "Q_RZ",       0,        0,      0,   0, 0, 0,    INFINITY, -INFINITY,   10,   0);
  set_joint_params(r, Q_FL1,"Q_FL1", 0.1934,   0.0465,      0,   1, 0, 0,      1.0472,    -1.0472,   10,  23.7);
  set_joint_params(r, Q_FL2,"Q_FL2",      0,   0.0955,      0,   1, 0, 0,      3.4907,    -1.5708,   10,  23.7);
  set_joint_params(r, Q_FL3,"Q_FL3",      0,        0, -0.213,   1, 0, 0,    -0.83776,    -2.7227,   10,  45.43);
  set_joint_params(r, Q_FR1,"Q_FR1", 0.1934,  -0.0465,      0,   1, 0, 0,      1.0472,    -1.0472,   10,  23.7);
  set_joint_params(r, Q_FR2,"Q_FR2",      0,  -0.0955,      0,   1, 0, 0,      3.4907,    -1.5708,   10,  23.7);
  set_joint_params(r, Q_FR3,"Q_FR3",      0,        0, -0.213,   1, 0, 0,    -0.83776,    -2.7227,   10,  45.43);
  set_joint_params(r, Q_BL1,"Q_BL1",-0.1934,   0.0465,      0,   1, 0, 0,      1.0472,    -1.0472,   10,  23.7);
  set_joint_params(r, Q_BL2,"Q_BL2",      0,   0.0955,      0,   1, 0, 0,      4.5379,    -0.5236,   10,  23.7);
  set_joint_params(r, Q_BL3,"Q_BL3",      0,        0, -0.213,   1, 0, 0,    -0.83776,    -2.7227,   10,  45.43);
  set_joint_params(r, Q_BR1,"Q_BR1",-0.1934,  -0.0465,      0,   1, 0, 0,      1.0472,    -1.0472,   10,  23.7);
  set_joint_params(r, Q_BR2,"Q_BR2",      0,  -0.0955,      0,   1, 0, 0,      4.5379,    -0.5236,   10,  23.7);
  set_joint_params(r, Q_BR3,"Q_BR3",      0,        0, -0.213,   1, 0, 0,    -0.83776,    -2.7227,   10,  45.43);


  //                                       m,        c_x,         c_y,        c_z,         j_xx,       j_yy,        j_zz,        j_xy,         j_xz,        j_yz
  set_link_params(r, L_X,   "L_X",         0,          0,           0,          0,            0,          0,           0,           0,            0,           0);
  set_link_params(r, L_Y,   "L_Y",         0,          0,           0,          0,            0,          0,           0,           0,            0,           0);
  set_link_params(r, L_Z,   "L_Z",         0,          0,           0,          0,            0,          0,           0,           0,            0,           0);
  set_link_params(r, L_RX,  "L_RX",        0,          0,           0,          0,            0,          0,           0,           0,            0,           0);
  set_link_params(r, L_RY,  "L_RY",        0,          0,           0,          0,            0,          0,           0,           0,            0,           0);
  set_link_params(r, L_BODY,"L_BODY",  6.921,   0.021112,           0,  -0.005366,     0.107027,  0.0980771,   0.0244531, 8.38742e-05,  0.000597672, 2.51329e-05);
  set_link_params(r, L_FL1, "L_FL1",   0.678,    -0.0054,     0.00194,  -0.000105,   0.00088403, 0.00059600, 0.000479967,  -9.409e-06,    -3.42e-07,   -4.66e-07);
  set_link_params(r, L_FL2, "L_FL2",   1.152,   -0.00374,     -0.0223,    -0.0327,   0.00584149, 0.00513934, 0.000878787,   4.825e-06,  0.000343869,  2.2448e-05);
  set_link_params(r, L_FL3, "L_FL3",   0.2413,    0.0044,     -0.0008,    -0.1352,    0.0014901, 0.00146356, 5.31397e-05,           0, -0.000167427,           0);
  set_link_params(r, L_FR1, "L_FR1",   0.678,    -0.0054,    -0.00194,  -0.000105,   0.00088403, 0.00059600, 0.000479967,   9.409e-06,    -3.42e-07,    4.66e-07);
  set_link_params(r, L_FR2, "L_FR2",   1.152,   -0.00374,      0.0223,    -0.0327,   0.00594973, 0.00584149, 0.000878787, 0.000878787,  0.000343869, -2.2448e-05); 
  set_link_params(r, L_FR3, "L_FR3",   0.2413,    0.0044,      0.0008,    -0.1352,    0.0014901, 0.00146356, 5.31397e-05,           0, -0.000167427,           0);
  set_link_params(r, L_BL1, "L_BL1",   0.678,     0.0054,     0.00194,  -0.000105,   0.00088403, 0.00059600, 0.000479967,   9.409e-06,     3.42e-07,   -4.66e-07);
  set_link_params(r, L_BL2, "L_BL2",   1.152,   -0.00374,     -0.0223,    -0.0327,   0.00594973, 0.00584149, 0.000878787,   4.825e-06,  0.000343869,  2.2448e-05);
  set_link_params(r, L_BL3, "L_BL3",   0.2413,    0.0044,     -0.0008,    -0.1352,    0.0014901, 0.00146356, 5.31397e-05,           0, -0.000167427,           0);
  set_link_params(r, L_BR1, "L_BR1",   0.678,     0.0054,    -0.00194,  -0.000105,   0.00088403, 0.00059600, 0.000479967,  -9.409e-06,     3.42e-07,    4.66e-07);
  set_link_params(r, L_BR2, "L_BR2",   1.152,   -0.00374,      0.0223,    -0.0327,   0.00594973, 0.00584149, 0.000878787,  -4.825e-06,  0.000343869, -2.2448e-05);
  set_link_params(r, L_BR3, "L_BR3",   0.2413,    0.0044,      0.0008,    -0.1352,    0.0014901, 0.00146356, 5.31397e-05,           0, -0.000167427,           0);

  //                                name,     parent,   x,   y, z
  r.frames[F_Q_FL1] = (RobotFrame) {"F_Q_FL1", L_FL1,   0,   0, 0};
  r.frames[F_Q_FL2] = (RobotFrame) {"F_Q_FL2", L_FL2,   0,   0, 0};
  r.frames[F_Q_FL3] = (RobotFrame) {"F_Q_FL3", L_FL3,   0,   0, 0};
  r.frames[F_Q_FR1] = (RobotFrame) {"F_Q_FR1", L_FR1,   0,   0, 0};
  r.frames[F_Q_FR2] = (RobotFrame) {"F_Q_FR2", L_FR2,   0,   0, 0};
  r.frames[F_Q_FR3] = (RobotFrame) {"F_Q_FR3", L_FR3,   0,   0, 0};
  r.frames[F_Q_BL1] = (RobotFrame) {"F_Q_BL1", L_BL1,   0,   0, 0};
  r.frames[F_Q_BL2] = (RobotFrame) {"F_Q_BL2", L_BL2,   0,   0, 0};
  r.frames[F_Q_BL3] = (RobotFrame) {"F_Q_BL3", L_BL3,   0,   0, 0};
  r.frames[F_Q_BR1] = (RobotFrame) {"F_Q_BR1", L_BR1,   0,   0, 0};
  r.frames[F_Q_BR2] = (RobotFrame) {"F_Q_BR2", L_BR2,   0,   0, 0};
  r.frames[F_Q_BR3] = (RobotFrame) {"F_Q_BR3", L_BR3,   0,   0, 0};
  r.frames[F_FL_EE] = (RobotFrame) {"F_FL_EE", L_FL3,   0,   0, -0.213};
  r.frames[F_FR_EE] = (RobotFrame) {"F_FR_EE", L_FR3,   0,   0, -0.213};
  r.frames[F_BL_EE] = (RobotFrame) {"F_BL_EE", L_BL3,   0,   0, -0.213};
  r.frames[F_BR_EE] = (RobotFrame) {"F_BR_EE", L_BR3,   0,   0, -0.213};

  //                                      frame,      name,    k,    b, mu, c_x, x_y, c_z
  r.contacts[C_FL_EE] = (RobotContact) {F_FL_EE, "C_FL_EE", 20000, 1000,  0.7, 0,0,0};
  r.contacts[C_FR_EE] = (RobotContact) {F_FR_EE, "C_FR_EE", 20000, 1000,  0.7, 0,0,0};
  r.contacts[C_BL_EE] = (RobotContact) {F_BL_EE, "C_BL_EE", 20000, 1000,  0.7, 0,0,0};
  r.contacts[C_BR_EE] = (RobotContact) {F_BR_EE, "C_BR_EE", 20000, 1000,  0.7, 0,0,0};
  
}

void robot_state_init(Robot &robot) {

  robot.q = VectorXd::Zero((int)NUM_Q);
  robot.qd = VectorXd::Zero((int)NUM_Q);

  robot.q << 0,    0, 0.12,
             0,    0,    0,
             0, 1.57, -2.7,
             0, 1.57, -2.7,
             0, 1.57, -2.7,
             0, 1.57, -2.7;

  robot.q_sim = robot.q;
  robot.qd_sim = robot.qd;

  memset(&robot.act_cmds, 0, sizeof(ActuatorCmds));

  memset(&robot.c_s, false, sizeof(ContactState));
  
  robot.ctrl_curr.mode = C_SOFT_STOP;
  robot.ctrl_next.mode = C_SOFT_STOP;
  robot.ctrl_des.mode = C_SOFT_STOP;

}

//TODO parameterize limit and handle this in core
//TODO include velocity and position limits
void enforce_limits(ActuatorCmds &act_cmds) {
  // limit torque, TODO limit this with pos and vel in a dedicated function
  for (int i =0; i< NUM_U; ++i) {
    if ( i == U_FL3 || i ==  U_FR3 || i == U_BL3 || U_BR3)
    {
      act_cmds.u[i]=fmax(fmin(act_cmds.u[i],45.43),-45.43);
    }
    else{
      act_cmds.u[i]=fmax(fmin(act_cmds.u[i],23.7),-23.7);
    }
  }
  
}

