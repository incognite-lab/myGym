FROM osrf/ros:melodic-desktop-full-bionic

LABEL maintainer="Yue Erro <yue.erro@pal-robotics.com>"

# Set up Python 2.7 environment
RUN update-alternatives --install /usr/bin/python python /usr/bin/python2.7 1

ARG REPO_WS=/tiago_public_ws
RUN mkdir -p $REPO_WS/src
WORKDIR $REPO_WS


# Try to remove old keys
RUN apt-key del AD19BAB3CBF125EA || true
RUN apt-key del C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 || true
RUN sed -i.bak '/snapshots.ros.org\/melodic\/final/ s/^/#/' /etc/apt/sources.list.d/ros1-snapshots.list

RUN apt-get -y update && apt-get -y install wget curl ca-certificates
# Install new GPG ROS key
RUN wget -qO /usr/share/keyrings/ros-archive-keyring.gpg \
      https://github.com/ros-infrastructure/ros-apt-source/raw/refs/heads/main/ros-apt-source/keys/ros-archive-keyring.gpg \
    && chmod a+r /usr/share/keyrings/ros-archive-keyring.gpg

RUN echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros/ubuntu bionic main" \
    > /etc/apt/sources.list.d/ros-latest.list

# RUN apt install ros-apt-source -y && apt update
# Install dependencies with Python 2 compatibility
RUN apt-get -y update && apt-get install -y --no-install-recommends \
    apt-utils \
    libv4l-dev \
    libv4l2rds0 \
    git \
    wget \
    vim \
    locales \
    dpkg \
    ssh \
    curl \
    aptitude \
    g++ \
    gcc \
    openvpn \
    gnupg \
    bash-completion \
    vim-gtk3 \
    nano \
    psmisc \
    ccache \
    gdb \
    qtcreator \
    htop \
    man \
    meld \
    silversearcher-ag \
    terminator \
    tig \
    valgrind \
    iputils-ping \
    glmark2 \
    python-ipython \
    python-scipy \
    python-wstool \
    python-networkx \
    python-zmq \
    python-cloudpickle \
    python-kdl-parser \
    python-pip \
    python-vcstool \
    python-rosinstall \
    python-catkin-tools \
    ros-melodic-joint-state-controller \
    ros-melodic-twist-mux \
    ros-melodic-ompl \
    ros-melodic-controller-manager \
    ros-melodic-moveit-core \
    ros-melodic-moveit-ros-perception \
    ros-melodic-moveit-ros-move-group \
    ros-melodic-moveit-kinematics \
    ros-melodic-moveit-ros-planning-interface \
    ros-melodic-moveit-simple-controller-manager \
    ros-melodic-moveit-planners-ompl \
    ros-melodic-moveit-setup-assistant \
    ros-melodic-moveit-visual-tools \
    ros-melodic-moveit-ros-visualization \
    ros-melodic-python-orocos-kdl \
    ros-melodic-joy \
    ros-melodic-joy-teleop \
    ros-melodic-teleop-tools \
    ros-melodic-control-toolbox \
    ros-melodic-sound-play \
    ros-melodic-navigation \
    ros-melodic-depthimage-to-laserscan \
    ros-melodic-moveit-commander \
    ros-melodic-people \
    ros-melodic-octomap-server \
    ros-melodic-urdf-test \
    ros-melodic-urdf-geometry-parser \
    ros-melodic-teb-local-planner \
    ros-melodic-libuvc-camera \
    ros-melodic-pointcloud-to-laserscan \
    ros-melodic-ddynamic-reconfigure \
    ros-melodic-ddynamic-reconfigure-python \
    ros-melodic-ira-laser-tools \
    ros-melodic-sick-scan \
    ros-melodic-hector-gazebo \
    ros-melodic-librealsense2 \
    ros-melodic-realsense2-camera \
    ros-melodic-realsense2-description \
    ros-melodic-spatio-temporal-voxel-layer \
    ros-melodic-slam-toolbox \
    ros-melodic-openni2-camera \
    ros-melodic-openni2-launch \
    ros-melodic-roboticsgroup-upatras-gazebo-plugins \
    ros-melodic-usb-cam \
    ros-melodic-sick-tim \
    ros-melodic-four-wheel-steering-msgs \
    ros-melodic-moveit-fake-controller-manager

RUN rm -rf /var/lib/apt/lists/*

# Install ROS packages
# RUN wget https://raw.githubusercontent.com/pal-robotics/tiago_tutorials/melodic-devel/tiago_public-melodic.rosinstall \
RUN wget https://raw.githubusercontent.com/pal-robotics/tiago_dual_tutorials/refs/heads/master/tiago_dual_public-melodic.rosinstall \
  && vcs import src < tiago_dual_public-melodic.rosinstall

# Update rosdep ignore list for Melodic
ARG ROSDEP_IGNORE="joint_impedance_trajectory_controller pal_motion_loader urdf_test omni_drive_controller orocos_kdl pal_filters libgazebo9-dev pal_usb_utils speed_limit_node camera_calibration_files pal_moveit_plugins pal_startup_msgs pal_local_joint_control pal_pcl_points_throttle_and_filter current_limit_controller hokuyo_node dynamixel_cpp pal_moveit_capabilities pal_pcl dynamic_footprint gravity_compensation_controller pal-orbbec-openni2 pal_loc_measure pal_map_manager ydlidar_ros_driver"

# RUN apt-get update && rosdep update --include-eol-distros && \
#     rosdep install --from-paths src --ignore-src -y --rosdistro melodic --skip-keys="${ROSDEP_IGNORE}"
RUN apt-get update
RUN rosdep update --include-eol-distros
RUN rosdep install --from-paths src --ignore-src -y --rosdistro melodic --skip-keys="${ROSDEP_IGNORE}"

# Build with Melodic dependencies
RUN bash -c "source /opt/ros/melodic/setup.bash \
    && catkin build -DCATKIN_ENABLE_TESTING=0 -j $(expr $(nproc) / 2) \
    && echo 'source $REPO_WS/devel/setup.bash' >> ~/.bashrc"

COPY entrypoint.sh /
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
