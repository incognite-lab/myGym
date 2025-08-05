# mygym-ros

## Installation

Execute in the myGym root directory the following command:

```bash
python -m pip install -e .
```

## Dockerfile

For use with Tiago running on ROS 1 Melodic, build and start the accompanying
Docker container with the following command:

```bash
docker compose build
```

Make sure to correctly configure the `ROS_MASTER_URI` or any other environment variable in the `docker-compose.yml` file.

Start the container with the following commands:

```bash
docker compose up -d
```


Drop into the container with the following command:

```bash
docker compose exec melodic bash
```

At the end, stop the container with the following command:

```bash
docker compose down
```

## Running ROS joint trajectory sender

```bash
python src/user_packages/joint_control/joint_zmq_server.py
```

Or with arguments:

```bash
python src/user_packages/joint_control/joint_zmq_server.py --side right --ml 2 -m 0.1 -s -o 7 -d 5
```

- `--side`: left or right
- `--ml`: maximum length of trajectory (max num of joint points; 0 = no limit)
- `--m`: merge trajectory points closer than this value (in meters)
- `--s`: smooth the trajectory
- `--o`: start offset of the trajectory (omits first N points)
- `--d`: duration of each trajectory point (in seconds)

## Recording ROS data

The following command will start a recording script with web GUI:

```bash
python -m mygym_ros
```

In the web GUI, click `Create new Instant Playback` - recording will start automatically.
To record data, simply run myGym simulation, making sure `ROSRobot` class is used to create the robot in myGym environment.

All that is needed to do that is to include the following code before the environment is created:

```python
    from myGym.envs.robot import ROSRobot, Robot
    Robot.shim_to(ROSRobot)  # swap Robot for ROSRobot class
```

## Playback ROS data

Select the following transmit plugin: `<class 'mygym_ros.transmit_plugins.tp_melodic_direct.RosMelodicTransmitPlugin'>`

After enough data has been recorded, click `Stop recording`. A `Play` button will appear. Click it to start playback.

Make sure the `joint_zmq_server.py` ROS node is running, before clicking `Play`.

If everything went well, the `joint_zmq_server` node should display the received trajectory and prompt you to press `y` to start executing the trajectory (or `n` to cancel).

## Displaying the trajectory before executing it

The `joint_zmq_server` node will also display the trajectory before executing it. To visualize it, use RViz.
Make sure RViz runs in the same ROS distro as the `joint_zmq_server` node.
That is, when using the `joint_zmq_server` node on ROS 1 Melodic, start RViz in the Melodic container:

```bash
docker compose exec melodic rosrun rviz rviz
```

or

```bash
docker compose exec melodic rviz
```

Then, add the topics in RViz:

- `/move_group/display_planned_path`
- `/visualization_marker_array`
