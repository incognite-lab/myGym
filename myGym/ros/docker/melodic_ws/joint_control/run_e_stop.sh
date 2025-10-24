#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PYTHONPATH:-/opt/ros/melodic/lib/python2.7/dist-packages}"
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
export ROS_IP="${ROS_IP:-127.0.0.1}"

exec sudo --preserve-env=PYTHONPATH,ROS_MASTER_URI,ROS_IP /usr/bin/python e_stop.py "$@"