#!/bin/bash
set -e

# Source the ROS environment
source /opt/ros/melodic/setup.bash

# Go to your workspace directory
cd /tiago_public_ws

# Build user_packages if it exists
USER_PKG_DIR="/tiago_public_ws/src/user_packages/"
ALL_PACKAGES=$(catkin list)
USER_PACKAGE_PATHS=$(find "$USER_PKG_DIR" -mindepth 1 -maxdepth 3 -type f -name 'package.xml' -exec dirname {} \;)
PACKAGES_TO_BUILD=()

for pkg_path in $USER_PACKAGE_PATHS; do
  pkg_name=$(basename "$pkg_path")
  if echo "$ALL_PACKAGES" | grep -q "$pkg_name"; then
    PACKAGES_TO_BUILD+=("$pkg_name")
  fi
done

echo "Packages to build: ${PACKAGES_TO_BUILD[@]}"

if [ ${#PACKAGES_TO_BUILD[@]} -gt 0 ]; then
  catkin build "${PACKAGES_TO_BUILD[@]}"
else
  echo "No user packages found to build."
fi

# Source the workspace setup file if it exists
if [ -f "/tiago_public_ws/devel/setup.bash" ]; then
  source /tiago_public_ws/devel/setup.bash
fi

# Set default ROS environment variables if not already set
export ROS_MASTER_URI=${ROS_MASTER_URI:-http://tiago-114c:11311}
export ROS_IP=${ROS_IP:-10.68.0.128}

exec "$@"

