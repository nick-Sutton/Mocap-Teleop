# Mango Pineapple Avocado Cherry


This repository contains the Go2 quadruped specific code for the MPAC framework,
including robot description, Go2 I/O, estimation, telemetry, visualization,
autonomy, and motion primitive implementations.

## Install
```bash
git clone --recurse-submodules git@github.com:Hier-Lab/mpac_go2.git && cd mpac_go2
sudo ./dependencies.bash
source $( [[ $(getent passwd $USER | cut -d: -f7) == *"zsh"* ]] && echo "$HOME/.zshrc" || echo "$HOME/.bashrc" )
mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release
bear -- make -j10 # or bear make -j10 on failure
```

## Usage

```
build/ctrl --io_mode=mujoco
build/ctrl --io_mode=hardware
```

If two controllers are to be run at the same time (e.g. hardware + sim), unique autonomy, telemetry and control ports have to be picked (to prevent cross-talk)

```
CTRL_PORT=8081 ATNMY_PORT=8082 TLM_PORT=8083 build/ctrl --io_mode=mujoco
CTRL_PORT=8091 ATNMY_PORT=8092 TLM_PORT=8093 build/ctrl --io_mode=mujoco
```

## Run manual autonomy
```
python3 atnmy/mpac_cmd.py
```

If multiple controllers are used, `CTRL_PORTS` and `ATNMY_PORTS` need to be set to comma separated ports
```
CTRL_PORTS="8081,8091" ATNMY_PORTS="8082,8092" python3 atnmy/mpac_cmd.py
```
