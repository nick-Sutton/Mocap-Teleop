FROM ubuntu:22.04

# Avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y build-essential cmake git libeigen3-dev libboost-all-dev liblcm-dev libhdf5-dev tar vim lsb-release

# Copy compressed mpac_go2
COPY mpac_go2.tar.gz /root/
RUN tar -xzvf /root/mpac_go2.tar.gz -C /root/

# Set working directory
WORKDIR /root/mpac_go2
