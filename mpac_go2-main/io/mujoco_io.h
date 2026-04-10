#ifndef MUJOCO_IO_H
#define MUJOCO_IO_H

#include "robot.h"

#include "fk.h"
#include "math_utils.h"

#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/go2/LowState_.hpp>
#include <unitree/idl/go2/LowCmd_.hpp>
#include <unitree/common/time/time_tool.hpp>
#include <unitree/common/thread/thread.hpp>

#include <vector>
#include <string>
#include <math.h>
#include <stdint.h>

using namespace unitree::common;
using namespace unitree::robot;

class Mujoco_io
{
  public:
    Mujoco_io();
    ~Mujoco_io();

    void init(bool hardware);
    void read(OutputVec &y);
    void update(const ActuatorCmds &act_update);
    void write();
    void finish();
    void InitLowCmd();

  private:
    unitree_go::msg::dds_::LowCmd_ low_cmd{};     // default init
    unitree_go::msg::dds_::LowState_ low_state{}; // default init

    /*publisher*/
    ChannelPublisherPtr<unitree_go::msg::dds_::LowCmd_> lowcmd_publisher;
    /*subscriber*/
    ChannelSubscriberPtr<unitree_go::msg::dds_::LowState_> lowstate_subscriber;

    /*LowCmd write thread*/
    ThreadPtr lowCmdWriteThreadPtr;

    void LowStateMessageHandler(const void *message);

    double ground_truth[3];
    double qd_offsets[NUM_U] = {};
    bool tick_ready = false;
    double dt = 0.001;

    ActuatorCmds act_cmds;

    static const int buff_size = 3; 

    // definition of each leg and joint
    int FR_ = 1;       // leg index
    int FL_ = 0;
    int RR_ = 3;
    int RL_ = 2;

    int FL_0 = 0;
    int FL_1 = 1;
    int FL_2 = 2;

    int FR_0 = 3;      // joint index
    int FR_1 = 4;      
    int FR_2 = 5;

    int RL_0 = 6;
    int RL_1 = 7;
    int RL_2 = 8; 

    int RR_0 = 9;
    int RR_1 = 10;
    int RR_2 = 11;


    bool is_hardware = false;
    void read_sim(OutputVec &y);
    void write_sim();

    void read_hardware(OutputVec &y);
    void write_hardware();


};

#endif //MUJOCO_IO_H


