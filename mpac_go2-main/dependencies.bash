#!/usr/bin/env bash

# add function to echo in green
function echo_green() {
    echo -e "\033[32m$1\033[0m"
}

function echo_red() {
    echo -e "\033[31m$1\033[0m"
}

function echo_line() {
    echo "----------------------------------------"
}

if [ "$EUID" -ne 0 ]; then
    USER_HOME=$(eval echo ~$USER)
else
    USER_HOME=$(eval echo ~$SUDO_USER)
fi

echo_green "User home directory: $USER_HOME"

# Determine user's shell and set RC file
USER_SHELL=$(getent passwd $SUDO_USER | cut -d: -f7)
if [[ "$USER_SHELL" == *"zsh"* ]]; then
    RC_FILE="$USER_HOME/.zshrc"
    echo_green "Detected zsh shell, using $RC_FILE"
else
    RC_FILE="$USER_HOME/.bashrc"
    echo_green "Detected bash or other shell, using $RC_FILE"
fi

if [ -f /etc/lsb-release ]; then
    UBUNTU_VERSION=$(lsb_release -r | awk '{print $2}')
    # Only support Ubuntu 20.04, 22.04 and 24.04
    if [[ "$UBUNTU_VERSION" != "20.04" && "$UBUNTU_VERSION" != "22.04" && "$UBUNTU_VERSION" != "24.04" ]]; then
        echo_red "Unsupported Ubuntu version: $UBUNTU_VERSION"
        echo_red "This script only supports Ubuntu 20.04, 22.04 and 24.04"
        exit 1
    fi
fi

echo_green "Detected Ubuntu version: $UBUNTU_VERSION"

# Check that the script is run as root
if [ "$EUID" -ne 0 ]; then
    echo_red "Please run this script as root: sudo $0"
    exit 1
fi

# Update system
echo_line
echo_green "Updating system"
apt-get update
echo_green "Updating system done"

# Install dependencies
echo_line
echo_green "Installing build-essential"
apt-get install build-essential
echo_green "Installing build-essential done"

# Install bear
echo_line
echo_green "Installing bear"
apt-get install bear
echo_green "Installing bear done"

echo_line
echo_green "Installing cmake"
apt-get install cmake
echo_green "Installing cmake done"

echo_line
echo_green "Installing libeigen3-dev"
apt-get install libeigen3-dev
echo_green "Installing libeigen3-dev done"

echo_line
echo_green "Installing libhdf5-dev"
apt-get install libhdf5-dev
echo_green "Installing libhdf5-dev done"

echo_line
echo_green "Installing lcm"
apt-get install liblcm-dev
echo_green "Installing lcm done"

echo_line
echo_green "Installing pinocchio" # https://stack-of-tasks.github.io/pinocchio/download.html
apt install -qqy lsb-release curl
mkdir -p /etc/apt/keyrings
curl http://robotpkg.openrobots.org/packages/debian/robotpkg.asc | tee /etc/apt/keyrings/robotpkg.asc
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/robotpkg.asc] http://robotpkg.openrobots.org/packages/debian/pub $(lsb_release -cs) robotpkg" \
    | tee /etc/apt/sources.list.d/robotpkg.list
apt-get update

if [[ "$UBUNTU_VERSION" == "20.04" ]]; then
    apt install -qqy robotpkg-py38-pinocchio
elif [[ "$UBUNTU_VERSION" == "22.04" ]]; then
    apt install -qqy robotpkg-py310-pinocchio
elif [[ "$UBUNTU_VERSION" == "24.04" ]]; then
    apt install -qqy robotpkg-py312-pinocchio
fi

if ! grep -q "export PATH=/opt/openrobots/bin:" $RC_FILE; then
    echo_green "Adding openrobots to PATH in $RC_FILE"
    echo "export PATH=/opt/openrobots/bin:$PATH" >> $RC_FILE
fi
if ! grep -q "export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:" $RC_FILE; then
    echo_green "Adding openrobots to PKG_CONFIG_PATH in $RC_FILE"
    echo "export PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig:$PKG_CONFIG_PATH" >> $RC_FILE
fi
if ! grep -q "export LD_LIBRARY_PATH=/opt/openrobots/lib:" $RC_FILE; then
    echo_green "Adding openrobots to LD_LIBRARY_PATH in $RC_FILE"
    echo "export LD_LIBRARY_PATH=/opt/openrobots/lib:$LD_LIBRARY_PATH" >> $RC_FILE
fi
if ! grep -q "export CMAKE_PREFIX_PATH=/opt/openrobots:" $RC_FILE; then
    echo_green "Adding openrobots to CMAKE_PREFIX_PATH in $RC_FILE"
    echo "export CMAKE_PREFIX_PATH=/opt/openrobots:$CMAKE_PREFIX_PATH" >> $RC_FILE
fi

if [[ "$UBUNTU_VERSION" == "20.04" ]]; then
    if ! grep -q "export PYTHONPATH=/opt/openrobots/lib/python3.8/site-packages:" $RC_FILE; then
        echo_green "Adding openrobots to PYTHONPATH in $RC_FILE"
        echo "export PYTHONPATH=/opt/openrobots/lib/python3.8/site-packages:$PYTHONPATH" >> $RC_FILE
    fi
elif [[ "$UBUNTU_VERSION" == "22.04" ]]; then
    if ! grep -q "export PYTHONPATH=/opt/openrobots/lib/python3.10/site-packages:" $RC_FILE; then
        echo_green "Adding openrobots to PYTHONPATH in $RC_FILE"
        echo "export PYTHONPATH=/opt/openrobots/lib/python3.10/site-packages:$PYTHONPATH" >> $RC_FILE
    fi
elif [[ "$UBUNTU_VERSION" == "24.04" ]]; then
    if ! grep -q "export PYTHONPATH=/opt/openrobots/lib/python3.12/site-packages:" $RC_FILE; then
        echo_green "Adding openrobots to PYTHONPATH in $RC_FILE"
        echo "export PYTHONPATH=/opt/openrobots/lib/python3.12/site-packages:$PYTHONPATH" >> $RC_FILE
    fi
fi

echo_line
echo_green "Adding Eigen3 to /usr/include"
if [ ! -e /usr/include/Eigen ]; then
    ln -s /usr/include/eigen3/Eigen /usr/include/Eigen
fi
if [ ! -e /usr/include/unsupported ]; then
    ln -s /usr/include/eigen3/unsupported /usr/include/unsupported
fi
echo_green "Adding Eigen3 to /usr/include done"

echo_green "Make sure to run source $RC_FILE"
echo_green "Installation complete. Now build something great!"
