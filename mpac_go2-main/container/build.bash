#!/usr/bin/env bash

# Pack the backpack
mkdir mpac_go2
cp -r ../* ./mpac_go2
tar -czvf mpac_go2.tar.gz --exclude=build --exclude=container mpac_go2
rm -rf mpac_go2

# Build container
docker build -t mpac_go2 .

# Remove junk
rm mpac_go2.tar.gz
