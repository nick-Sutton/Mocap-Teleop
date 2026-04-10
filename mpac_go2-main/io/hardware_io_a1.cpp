#include "stdio.h"

#include "robot.h"
#include "io/hardware_io.h"
#include "math_utils.h"

#include "unitree_legged_sdk/unitree_legged_sdk.h"
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <stdint.h>
#include <limits>

#include "ros/ros.h"
#include <geometry_msgs/PoseStamped.h>

using namespace UNITREE_LEGGED_SDK;

namespace hardware_io {

LowCmd cmd = {0};
LowState state = {0};
double ground_truth[6];
Safety safe(LeggedType::A1);
UDP* udp;

bool using_ros = false;

bool ros_init();

void init(char* args) {
  udp = new UDP(LOWLEVEL);
  InitEnvironment();
  udp->InitCmdData(cmd);
  if (args) {
    if (strcmp(args, "ground_truth") == 0) {
      using_ros = ros_init();
    }
  }
}

void read(OutputVec &y) {
  udp->Recv();

  udp->GetRecv(state);

  ground_truth[0] = std::numeric_limits<double>::quiet_NaN();
  ground_truth[1] = std::numeric_limits<double>::quiet_NaN();
  ground_truth[2] = std::numeric_limits<double>::quiet_NaN();
  ground_truth[3] = std::numeric_limits<double>::quiet_NaN();
  ground_truth[4] = std::numeric_limits<double>::quiet_NaN();
  ground_truth[5] = std::numeric_limits<double>::quiet_NaN();
  if (using_ros) {
    ros::spinOnce();
  }

  // IMU
  Eigen::Quaterniond quat;
  quat.w() = state.imu.quaternion[0];
  quat.x() = state.imu.quaternion[1];
  quat.y() = state.imu.quaternion[2];
  quat.z() = state.imu.quaternion[3];

  Matrix3d m = quat.toRotationMatrix();
  Vector3d euler = euler_wrap(m.eulerAngles(0, 1, 2));
  
  // Convert accel to global frame
  Vector3d accel_body;
  accel_body << state.imu.accelerometer[0],
                state.imu.accelerometer[1],
                state.imu.accelerometer[2];

  Vector3d accel_global = m*accel_body;
  y(S_IMU_ACC_X) = accel_global(0);
  y(S_IMU_ACC_Y) = accel_global(1);
  y(S_IMU_ACC_Z) = accel_global(2);

  y(S_IMU_FUSED_RX) = euler[0];
  y(S_IMU_FUSED_RY) = euler[1];
  y(S_IMU_FUSED_RZ) = euler[2];

  //TODO seems like gyro is reported in global frame,
  //I would have expect body frame, but they must transform 
  //before reporting. We use body frame for these channels, 
  //so rotate by the measured orientation.
  //Should investigate a little more.
  Vector3d gyro_global;
  gyro_global << state.imu.gyroscope[0],
                 state.imu.gyroscope[1],
                 state.imu.gyroscope[2];
  Vector3d gyro_body = m*gyro_global;

  y(S_IMU_GYRO_RX) = gyro_body(0);
  y(S_IMU_GYRO_RY) = gyro_body(1);
  y(S_IMU_GYRO_RZ) = gyro_body(2);

  // foot force
  y(F_FRC_FL_EE) = state.footForce[FL_];
  y(F_FRC_FR_EE) = state.footForce[FR_];
  y(F_FRC_BL_EE) = state.footForce[RL_];
  y(F_FRC_BR_EE) = state.footForce[RR_];

  // Joints, converting from unitree convention to ours
  y(S_Q_FL1) = state.motorState[FL_0].q;
  y(S_Q_FL2) = state.motorState[FL_1].q;
  y(S_Q_FL3) = state.motorState[FL_2].q;
                                       
  y(S_Q_FR1) = state.motorState[FR_0].q;
  y(S_Q_FR2) = state.motorState[FR_1].q;
  y(S_Q_FR3) = state.motorState[FR_2].q;
                                      
  y(S_Q_BL1) = state.motorState[RL_0].q;
  y(S_Q_BL2) = state.motorState[RL_1].q;
  y(S_Q_BL3) = state.motorState[RL_2].q;
                                     
  y(S_Q_BR1) = state.motorState[RR_0].q;
  y(S_Q_BR2) = state.motorState[RR_1].q;
  y(S_Q_BR3) = state.motorState[RR_2].q;

  y(S_QD_FL1) = state.motorState[FL_0].dq;
  y(S_QD_FL2) = state.motorState[FL_1].dq;
  y(S_QD_FL3) = state.motorState[FL_2].dq;
                                    
  y(S_QD_FR1) = state.motorState[FR_0].dq;
  y(S_QD_FR2) = state.motorState[FR_1].dq;
  y(S_QD_FR3) = state.motorState[FR_2].dq;
                                   
  y(S_QD_BL1) = state.motorState[RL_0].dq;
  y(S_QD_BL2) = state.motorState[RL_1].dq;
  y(S_QD_BL3) = state.motorState[RL_2].dq;
                                  
  y(S_QD_BR1) = state.motorState[RR_0].dq;
  y(S_QD_BR2) = state.motorState[RR_1].dq;
  y(S_QD_BR3) = state.motorState[RR_2].dq;

  y(S_U_FL1) = state.motorState[FL_0].tauEst;
  y(S_U_FL2) = state.motorState[FL_1].tauEst;
  y(S_U_FL3) = state.motorState[FL_2].tauEst;
      
  y(S_U_FR1) = state.motorState[FR_0].tauEst;
  y(S_U_FR2) = state.motorState[FR_1].tauEst;
  y(S_U_FR3) = state.motorState[FR_2].tauEst;
     
  y(S_U_BL1) = state.motorState[RL_0].tauEst;
  y(S_U_BL2) = state.motorState[RL_1].tauEst;
  y(S_U_BL3) = state.motorState[RL_2].tauEst;
    
  y(S_U_BR1) = state.motorState[RR_0].tauEst;
  y(S_U_BR2) = state.motorState[RR_1].tauEst;
  y(S_U_BR3) = state.motorState[RR_2].tauEst;

  y(S_TEMP_FL1) = state.motorState[FL_0].temperature;
  y(S_TEMP_FL2) = state.motorState[FL_1].temperature;
  y(S_TEMP_FL3) = state.motorState[FL_2].temperature;

  y(S_TEMP_FR1) = state.motorState[FR_0].temperature;
  y(S_TEMP_FR2) = state.motorState[FR_1].temperature;
  y(S_TEMP_FR3) = state.motorState[FR_2].temperature;
     
  y(S_TEMP_BL1) = state.motorState[RL_0].temperature;
  y(S_TEMP_BL2) = state.motorState[RL_1].temperature;
  y(S_TEMP_BL3) = state.motorState[RL_2].temperature;
    
  y(S_TEMP_BR1) = state.motorState[RR_0].temperature;
  y(S_TEMP_BR2) = state.motorState[RR_1].temperature;
  y(S_TEMP_BR3) = state.motorState[RR_2].temperature;

  y(S_GROUND_TRUTH_X) = ground_truth[0];
  y(S_GROUND_TRUTH_Y) = ground_truth[1];
  y(S_GROUND_TRUTH_Z) = ground_truth[2];
  y(S_GROUND_TRUTH_RX) = ground_truth[3];
  y(S_GROUND_TRUTH_RY) = ground_truth[4];
  y(S_GROUND_TRUTH_RZ) = ground_truth[5];

}

void write(const ActuatorCmds &act_cmds) {
  
  int map_to_unitree[NUM_U] = {};
  map_to_unitree[U_FL1] = FL_0;
  map_to_unitree[U_FL2] = FL_1;
  map_to_unitree[U_FL3] = FL_2;
  map_to_unitree[U_FR1] = FR_0;
  map_to_unitree[U_FR2] = FR_1;
  map_to_unitree[U_FR3] = FR_2;
  map_to_unitree[U_BL1] = RL_0;
  map_to_unitree[U_BL2] = RL_1;
  map_to_unitree[U_BL3] = RL_2;
  map_to_unitree[U_BR1] = RR_0;
  map_to_unitree[U_BR2] = RR_1;
  map_to_unitree[U_BR3] = RR_2;
 
  for (int i = 0; i < NUM_U; ++i) {
    if (act_cmds.mode[i] == CMD_MODE_TAU) {
      cmd.motorCmd[map_to_unitree[i]].q = PosStopF;
      cmd.motorCmd[map_to_unitree[i]].dq = VelStopF;
      cmd.motorCmd[map_to_unitree[i]].Kp = 0;
      cmd.motorCmd[map_to_unitree[i]].Kd = 0;
    }
    if (act_cmds.mode[i] == CMD_MODE_TAU_VEL) {
      cmd.motorCmd[map_to_unitree[i]].q = PosStopF;
      cmd.motorCmd[map_to_unitree[i]].dq = act_cmds.qd[i];
      cmd.motorCmd[map_to_unitree[i]].Kp = 0;
      cmd.motorCmd[map_to_unitree[i]].Kd = act_cmds.kd[i];
    }
    if (act_cmds.mode[i] == CMD_MODE_TAU_POS) {
      cmd.motorCmd[map_to_unitree[i]].q = act_cmds.q[i];
      cmd.motorCmd[map_to_unitree[i]].dq = VelStopF;
      cmd.motorCmd[map_to_unitree[i]].Kp = act_cmds.kp[i];
      cmd.motorCmd[map_to_unitree[i]].Kd = 0;
    }
    if (act_cmds.mode[i] == CMD_MODE_TAU_VEL_POS) {
      cmd.motorCmd[map_to_unitree[i]].q = act_cmds.q[i];
      cmd.motorCmd[map_to_unitree[i]].dq = act_cmds.qd[i];
      cmd.motorCmd[map_to_unitree[i]].Kp = act_cmds.kp[i];
      cmd.motorCmd[map_to_unitree[i]].Kd = act_cmds.kd[i];
    }
    cmd.motorCmd[map_to_unitree[i]].tau = act_cmds.u[i];

  }

  safe.PowerProtect(cmd, state, 10);

  udp->SetSend(cmd);

  udp->Send();
}

void finish() {
  delete udp;
  if (using_ros) {
    ros::shutdown();
  }
}

void groundTruthCallback(const geometry_msgs::PoseStamped &msg)
{ 
  ground_truth[0] = msg.pose.position.x;
  ground_truth[1] = msg.pose.position.y;
  ground_truth[2] = msg.pose.position.z;

  Eigen::Quaterniond quat;
  quat.w() = msg.pose.orientation.w;
  quat.x() = msg.pose.orientation.x;
  quat.y() = msg.pose.orientation.y;
  quat.z() = msg.pose.orientation.z;

  Matrix3d m = quat.toRotationMatrix();
  Vector3d euler = euler_wrap(m.eulerAngles(0, 1, 2));

  ground_truth[3] = euler[0];
  ground_truth[4] = euler[1];
  ground_truth[5] = euler[2];
  
}

ros::Subscriber ground_truth_sub;

bool ros_init() {
  int argc = 0;
  ros::init(argc, NULL, "a1_listener", ros::init_options::NoSigintHandler);

  if (!ros::master::check()) {
    std::cout << "ROS master not found, running without ROS io." << std::endl;
    return false;
  }

  ros::NodeHandle nm;
  ground_truth_sub = nm.subscribe("/vrpn_client_node/test/pose", 1, groundTruthCallback);
  return true;
}

}
