#include "atnmy/atnmy.h"
#include "ctrl_modes_core.h"
#include "ctrl/ctrl_args.h"
#include "tlm/tlm.h"
#include "robot.h"
#include "atnmy_core.h"

#include <iostream>
#include <string>
#include <unistd.h>

static bool seqq = false;
static int seq_num = 0;

void atnmy_cmdline(Telemetry tlm, CtrlState *ctrl_des);
void atnmy_demo_seq(CtrlState *ctrl_des);

void atnmy(Telemetry tlm, CtrlState *ctrl_des) {
  if (seqq) {
    atnmy_demo_seq(ctrl_des);
  } else {
    atnmy_cmdline(tlm, ctrl_des);
  }
}

void atnmy_demo_seq(CtrlState *ctrl_des) {
  double x_speed = 0.25;
  double y_speed = 0.2;

  switch (seq_num) {
  case 0:
    ctrl_des->mode = C_LIE;
    printf("lie\n");
    break;

  case 1:
    sleep(3);
    ctrl_des->mode = C_WALK_IDQP;
    ctrl_des->args.cont[walk_idqp::ARG_H] = 0.25;
    printf("walk\n");
    break;

  case 2:
    sleep(3);
    ctrl_des->mode = C_WALK_IDQP;
    ctrl_des->args.cont[walk_idqp::ARG_H] = 0.25;
    ctrl_des->args.cont[walk_idqp::ARG_VX] = x_speed;
    printf("walk\n");
    break;

  case 3:
    sleep(3);
    ctrl_des->mode = C_WALK_IDQP;
    ctrl_des->args.cont[walk_idqp::ARG_H]  = 0.25;
    ctrl_des->args.cont[walk_idqp::ARG_VX] = -x_speed/2;
    printf("walk\n");
    break;

  case 4:
    sleep(3);
    ctrl_des->mode = C_WALK_IDQP;
    ctrl_des->args.cont[walk_idqp::ARG_H] = 0.25;
    ctrl_des->args.cont[walk_idqp::ARG_VY] = y_speed;
    printf("walk\n");
    break;

  case 5:
    sleep(3);
    ctrl_des->mode = C_WALK_IDQP;
    ctrl_des->args.cont[walk_idqp::ARG_H] = 0.25;
    ctrl_des->args.cont[walk_idqp::ARG_VY] = -y_speed;
    printf("walk\n");
    break;

  case 6:
    sleep(3);
    ctrl_des->mode = C_WALK_IDQP;
    ctrl_des->args.cont[walk_idqp::ARG_H] = 0.25;
    printf("walk\n");
    break;

  case 7:
    sleep(3);
    ctrl_des->mode = C_WALK_IDQP;
    ctrl_des->args.cont[walk_idqp::ARG_H] = 0.25;
    ctrl_des->args.cont[walk_idqp::ARG_VRZ] = 0.5;
    printf("walk\n");
    break;

  case 8:
    sleep(6);
    ctrl_des->mode = C_WALK_IDQP;
    ctrl_des->args.cont[walk_idqp::ARG_H] = 0.25;
    ctrl_des->args.cont[walk_idqp::ARG_VX]  = x_speed/2;
    ctrl_des->args.cont[walk_idqp::ARG_VRZ] = -0.5;
    printf("walk\n");
    break;

  default:
    sleep(10);
    ctrl_des->mode = C_LIE;
    printf("lie\n");
    seq_num = 0;
    seqq = false;
    break;
  }
  seq_num++;
}

void atnmy_cmdline(Telemetry tlm, CtrlState *ctrl_des) {
  char state_des_str[128] = "";
  std::cout << "Enter desired ctrl mode: ";
  std::cin.getline(state_des_str,32);
  char delim[] = ", ";

  ctrl_des->mode = tlm.ctrl_des.mode;
  if (strcmp(state_des_str, "seq") == 0) {
    ctrl_des->mode = C_LIE;
    seqq = true;
    seq_num = 0;
    return;
  } else
  if (strcmp(state_des_str, "") == 0) {
    if (tlm.ctrl_curr.mode == C_SOFT_STOP ||
        tlm.ctrl_curr.mode == C_HARD_STOP) {
      ctrl_des->mode = C_HARD_STOP;
    } else {
      ctrl_des->mode = C_SOFT_STOP;
    }
  } else {
    char arg_des_str[128] = "";
    strcpy(arg_des_str, state_des_str + strcspn(state_des_str, delim)+1);
	  char *mode_des_str = strtok(state_des_str, delim);
    ctrl_mode_from_string(ctrl_des->mode, mode_des_str);
    ctrl_args_from_string(*ctrl_des, arg_des_str);
  }
}
