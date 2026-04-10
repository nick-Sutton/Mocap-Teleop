#include "robot.h"

#include "ctrl_core.h"
#include "io/mujoco_io.h"
#include "sim_core.h"

#define TOPIC_LOWCMD "rt/lowcmd"
#define TOPIC_LOWSTATE "rt/lowstate"

constexpr double PosStopF = (2.146E+9f);
constexpr double VelStopF = (16000.0f);

extern Robot robot;

Mujoco_io::Mujoco_io() {}

Mujoco_io::~Mujoco_io() {}

void Mujoco_io::LowStateMessageHandler(const void *message) {
  low_state = *(unitree_go::msg::dds_::LowState_ *)message;
}

uint32_t crc32_core(uint32_t *ptr, uint32_t len) {
  unsigned int xbit = 0;
  unsigned int data = 0;
  unsigned int CRC32 = 0xFFFFFFFF;
  const unsigned int dwPolynomial = 0x04c11db7;

  for (unsigned int i = 0; i < len; i++) {
    xbit = 1 << 31;
    data = ptr[i];
    for (unsigned int bits = 0; bits < 32; bits++) {
      if (CRC32 & 0x80000000) {
        CRC32 <<= 1;
        CRC32 ^= dwPolynomial;
      } else {
        CRC32 <<= 1;
      }

      if (data & xbit)
        CRC32 ^= dwPolynomial;
      xbit >>= 1;
    }
  }

  return CRC32;
}

void Mujoco_io::init(bool hardware) {

  is_hardware = hardware;
  InitLowCmd();


  // TODO: Put these constants somewhere else
  if (is_hardware) {
    FR_ = 0; // leg index
    FL_ = 1;
    RR_ = 2;
    RL_ = 3;

    FL_0 = 3; // joint index
    FL_1 = 4;
    FL_2 = 5;

    FR_0 = 0;
    FR_1 = 1;
    FR_2 = 2;

    RL_0 = 9;
    RL_1 = 10;
    RL_2 = 11;

    RR_0 = 6;
    RR_1 = 7;
    RR_2 = 8;
  } else {
    FR_ = 1; // leg index
    FL_ = 0;
    RR_ = 3;
    RL_ = 2;

    FL_0 = 0;
    FL_1 = 1;
    FL_2 = 2;

    FR_0 = 3; // joint index
    FR_1 = 4;
    FR_2 = 5;

    RL_0 = 6;
    RL_1 = 7;
    RL_2 = 8;

    RR_0 = 9;
    RR_1 = 10;
    RR_2 = 11;
  }

  if (hardware == true) {
    std::cout << "Hardware mode" << std::endl;
    std::cout << "Connecting to: enxc8a362a87d02" << std::endl;
    ChannelFactory::Instance()->Init(0, "enxc8a362a87d02");
  } else {
    std::cout << "Simulation mode" << std::endl;
    std::cout << "Connecting to: sim" << std::endl;
    ChannelFactory::Instance()->Init(1, "lo");
  }

  /*create publisher*/
  lowcmd_publisher.reset(
      new ChannelPublisher<unitree_go::msg::dds_::LowCmd_>(TOPIC_LOWCMD));
  std::string channelName = lowcmd_publisher->GetChannelName();
  lowcmd_publisher->InitChannel();

  /*create subscriber*/
  lowstate_subscriber.reset(
      new ChannelSubscriber<unitree_go::msg::dds_::LowState_>(TOPIC_LOWSTATE));
  lowstate_subscriber->InitChannel(std::bind(&Mujoco_io::LowStateMessageHandler,
                                             this, std::placeholders::_1),
                                   1);

  /*loop publishing thread*/
  lowCmdWriteThreadPtr = CreateRecurrentThreadEx(
      "writebasiccmd", UT_CPU_ID_NONE, 500, &Mujoco_io::write, this);
}

void Mujoco_io::InitLowCmd() {
  low_cmd.head()[0] = 0xFE;
  low_cmd.head()[1] = 0xEF;
  low_cmd.level_flag() = 0xFF;
  low_cmd.gpio() = 0;

  for (int i = 0; i < 20; i++) {
    low_cmd.motor_cmd()[i].mode() = (0x01); // motor switch to servo (PMSM) mode
    low_cmd.motor_cmd()[i].q() = (PosStopF);
    low_cmd.motor_cmd()[i].kp() = (0);
    low_cmd.motor_cmd()[i].dq() = (VelStopF);
    low_cmd.motor_cmd()[i].kd() = (0);
    low_cmd.motor_cmd()[i].tau() = (0);
  }
}

void Mujoco_io::read(OutputVec &y) {

  if (is_hardware) {
    read_hardware(y);
  } else {
    read_sim(y);
  }
}

void Mujoco_io::update(const ActuatorCmds &act_update) {
  act_cmds = act_update;
}

void Mujoco_io::write() {

  if (is_hardware) {
    write_hardware();
  } else {
    write_sim();
  }
}

void Mujoco_io::read_sim(OutputVec &y) {
  Eigen::Quaterniond quat;
  quat.w() = low_state.imu_state().quaternion()[0];
  quat.x() = low_state.imu_state().quaternion()[1];
  quat.y() = low_state.imu_state().quaternion()[2];
  quat.z() = low_state.imu_state().quaternion()[3];

  Matrix3d m = quat.toRotationMatrix();
  Vector3d euler = euler_wrap(m.eulerAngles(0, 1, 2));

  y(S_IMU_FUSED_RX) = euler[0];
  y(S_IMU_FUSED_RY) = euler[1];
  y(S_IMU_FUSED_RZ) = euler[2];

  Vector3d gyro_global;
  gyro_global << low_state.imu_state().gyroscope()[0],
      low_state.imu_state().gyroscope()[1],
      low_state.imu_state().gyroscope()[2];
  Vector3d gyro_body = m * gyro_global;

  y(S_IMU_ACC_X) = low_state.imu_state().accelerometer()[0];
  y(S_IMU_ACC_Y) = low_state.imu_state().accelerometer()[1];
  y(S_IMU_ACC_Z) = low_state.imu_state().accelerometer()[2];

  y(S_IMU_GYRO_RX) = gyro_body(0);
  y(S_IMU_GYRO_RY) = gyro_body(1);
  y(S_IMU_GYRO_RZ) = gyro_body(2);

  // foot force
  y(F_FRC_FL_EE) = low_state.foot_force()[FL_];
  y(F_FRC_FR_EE) = low_state.foot_force()[FR_];
  y(F_FRC_BL_EE) = low_state.foot_force()[RL_];
  y(F_FRC_BR_EE) = low_state.foot_force()[RR_];

  // Joints, converting from unitree convention to ours
  y(S_Q_FL1) = low_state.motor_state()[FL_0].q();
  y(S_Q_FL2) = low_state.motor_state()[FL_1].q();
  y(S_Q_FL3) = low_state.motor_state()[FL_2].q();

  y(S_Q_FR1) = low_state.motor_state()[FR_0].q();
  y(S_Q_FR2) = low_state.motor_state()[FR_1].q();
  y(S_Q_FR3) = low_state.motor_state()[FR_2].q();

  y(S_Q_BL1) = low_state.motor_state()[RL_0].q();
  y(S_Q_BL2) = low_state.motor_state()[RL_1].q();
  y(S_Q_BL3) = low_state.motor_state()[RL_2].q();

  y(S_Q_BR1) = low_state.motor_state()[RR_0].q();
  y(S_Q_BR2) = low_state.motor_state()[RR_1].q();
  y(S_Q_BR3) = low_state.motor_state()[RR_2].q();

  y(S_QD_FL1) = low_state.motor_state()[FL_0].dq();
  y(S_QD_FL2) = low_state.motor_state()[FL_1].dq();
  y(S_QD_FL3) = low_state.motor_state()[FL_2].dq();

  y(S_QD_FR1) = low_state.motor_state()[FR_0].dq();
  y(S_QD_FR2) = low_state.motor_state()[FR_1].dq();
  y(S_QD_FR3) = low_state.motor_state()[FR_2].dq();

  y(S_QD_BL1) = low_state.motor_state()[RL_0].dq();
  y(S_QD_BL2) = low_state.motor_state()[RL_1].dq();
  y(S_QD_BL3) = low_state.motor_state()[RL_2].dq();

  y(S_QD_BR1) = low_state.motor_state()[RR_0].dq();
  y(S_QD_BR2) = low_state.motor_state()[RR_1].dq();
  y(S_QD_BR3) = low_state.motor_state()[RR_2].dq();

  y(S_U_FL1) = low_state.motor_state()[FL_0].tau_est();
  y(S_U_FL2) = low_state.motor_state()[FL_1].tau_est();
  y(S_U_FL3) = low_state.motor_state()[FL_2].tau_est();

  y(S_U_FR1) = low_state.motor_state()[FR_0].tau_est();
  y(S_U_FR2) = low_state.motor_state()[FR_1].tau_est();
  y(S_U_FR3) = low_state.motor_state()[FR_2].tau_est();

  y(S_U_BL1) = low_state.motor_state()[RL_0].tau_est();
  y(S_U_BL2) = low_state.motor_state()[RL_1].tau_est();
  y(S_U_BL3) = low_state.motor_state()[RL_2].tau_est();

  y(S_U_BR1) = low_state.motor_state()[RR_0].tau_est();
  y(S_U_BR2) = low_state.motor_state()[RR_1].tau_est();
  y(S_U_BR3) = low_state.motor_state()[RR_2].tau_est();

  y(S_GROUND_TRUTH_X) = std::numeric_limits<double>::quiet_NaN();
  y(S_GROUND_TRUTH_Y) = std::numeric_limits<double>::quiet_NaN();
  y(S_GROUND_TRUTH_Z) = std::numeric_limits<double>::quiet_NaN();
  y(S_GROUND_TRUTH_RX) = std::numeric_limits<double>::quiet_NaN();
  y(S_GROUND_TRUTH_RY) = std::numeric_limits<double>::quiet_NaN();
  y(S_GROUND_TRUTH_RZ) = std::numeric_limits<double>::quiet_NaN();
}

void Mujoco_io::write_sim() {
  for (int i = 0; i < NUM_U; ++i) {
    if (act_cmds.mode[i] == CMD_MODE_TAU) {
      low_cmd.motor_cmd()[i].q() = PosStopF;
      low_cmd.motor_cmd()[i].dq() = VelStopF;
      low_cmd.motor_cmd()[i].kp() = 0;
      low_cmd.motor_cmd()[i].kd() = 0;
    }
    if (act_cmds.mode[i] == CMD_MODE_TAU_VEL) {
      low_cmd.motor_cmd()[i].q() = PosStopF;
      low_cmd.motor_cmd()[i].dq() = act_cmds.qd[i];
      low_cmd.motor_cmd()[i].kp() = 0;
      low_cmd.motor_cmd()[i].kd() = act_cmds.kd[i];
    }
    if (act_cmds.mode[i] == CMD_MODE_TAU_POS) {
      low_cmd.motor_cmd()[i].q() = act_cmds.q[i];
      low_cmd.motor_cmd()[i].dq() = VelStopF;
      low_cmd.motor_cmd()[i].kp() = act_cmds.kp[i];
      low_cmd.motor_cmd()[i].kd() = 0;
    }
    if (act_cmds.mode[i] == CMD_MODE_TAU_VEL_POS) {
      low_cmd.motor_cmd()[i].q() = act_cmds.q[i];
      low_cmd.motor_cmd()[i].dq() = act_cmds.qd[i];
      low_cmd.motor_cmd()[i].kp() = act_cmds.kp[i];
      low_cmd.motor_cmd()[i].kd() = act_cmds.kd[i];
    }
    low_cmd.motor_cmd()[i].tau() = act_cmds.u[i];
  }

  low_cmd.crc() = crc32_core((uint32_t *)&low_cmd,
                             (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2) - 1);
  lowcmd_publisher->Write(low_cmd);
}

void Mujoco_io::read_hardware(OutputVec &y) {
  read_sim(y); // same as sim
}

void Mujoco_io::write_hardware() {
  auto setcommand = [&](int i, int j) {
    if (act_cmds.mode[j] == CMD_MODE_TAU) {
      low_cmd.motor_cmd()[i].q() = PosStopF;
      low_cmd.motor_cmd()[i].dq() = VelStopF;
      low_cmd.motor_cmd()[i].kp() = 0;
      low_cmd.motor_cmd()[i].kd() = 0;
    }
    if (act_cmds.mode[j] == CMD_MODE_TAU_VEL) {
      low_cmd.motor_cmd()[i].q() = PosStopF;
      low_cmd.motor_cmd()[i].dq() = act_cmds.qd[j];
      low_cmd.motor_cmd()[i].kp() = 0;
      low_cmd.motor_cmd()[i].kd() = act_cmds.kd[j];
    }
    if (act_cmds.mode[j] == CMD_MODE_TAU_POS) {
      low_cmd.motor_cmd()[i].q() = act_cmds.q[j];
      low_cmd.motor_cmd()[i].dq() = VelStopF;
      low_cmd.motor_cmd()[i].kp() = act_cmds.kp[j];
      low_cmd.motor_cmd()[i].kd() = 0;
    }
    if (act_cmds.mode[j] == CMD_MODE_TAU_VEL_POS) {
      low_cmd.motor_cmd()[i].q() = act_cmds.q[j];
      low_cmd.motor_cmd()[i].dq() = act_cmds.qd[j];
      low_cmd.motor_cmd()[i].kp() = act_cmds.kp[j];
      low_cmd.motor_cmd()[i].kd() = act_cmds.kd[j];
    }
    low_cmd.motor_cmd()[i].tau() = act_cmds.u[j];
  };

  // FR
  setcommand(0, 3);
  setcommand(1, 4);
  setcommand(2, 5);

  // FL
  setcommand(3, 0);
  setcommand(4, 1);
  setcommand(5, 2);

  // RR
  setcommand(6, 9);
  setcommand(7, 10);
  setcommand(8, 11);

  // RL
  setcommand(9, 6);
  setcommand(10, 7);
  setcommand(11, 8);

  low_cmd.crc() = crc32_core((uint32_t *)&low_cmd,
                             (sizeof(unitree_go::msg::dds_::LowCmd_) >> 2) - 1);
  lowcmd_publisher->Write(low_cmd);
}

void Mujoco_io::finish() {}
