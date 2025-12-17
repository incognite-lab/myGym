# Sim2Real

Simulation enables fast prototyping and testing. This work focuses on trajectory-based Sim2Real transfer. The joint trajectories are reused on the real robot.


## /myGym/test.py

Launch the simulation and save the trajectory of the 7-joint arm.

`python test.py --config /home/student/Documents/myGym/myGym/trained_models/tiago_dual_fix/A/joints_ppo_1/train.json --pretrained_model /home/student/Documents/myGym/myGym/trained_models/tiago_dual_fix/A/joints_ppo_1 --eval_episodes 1 --g 1`


## /myGym/test_AG.py

Launch the simulation and save the trajectory of the 7-joint arm + 2-joint gripper, for approach and grasps task.

`python test_AG.py --config /home/student/Documents/myGym/myGym/trained_models/tiago_dual_fix/AG/joints_gripper_ppo/train.json --pretrained_model /home/student/Documents/myGym/myGym/trained_models/tiago_dual_fix/AG/joints_gripper_ppo --eval_episodes 1 --g 1`

## /zmq-comm/marker_to_targets_file.py

Receive the object coordinates estimated from gdr-net and overwrite them in `/zmq-comm/target.txt`.

## /zmq-comm/send_joint_test.py

Run `/myGym/test.py` and send the trajectory to the specified coordinates to the ZMQ server.(decied position manualy)

## /zmq-comm/send_joint_multitest.py

Run `/myGym/test.py` and send the trajectories to the coordinates written in `/zmq-comm/target.txt` to the ZMQ server.(send multi positon)

## /zmq-comm/send_joint_gdrnet.py

Run `/myGym/test.py` and send the trajectories to the coordinates written in `/zmq-comm/target.txt` to the ZMQ server.(based on gdr-net)

## /zmq-comm/send_joint_gdrnet_AG_pre.py

Run `/myGym/test_AG.py` and send the trajectories to the coordinates written in `/zmq-comm/target.txt` to the ZMQ server.(based on gdr-net and execute AGtask)

`python send_joint_gdrnet_AG_pre.py --test-args "--config /home/student/Documents/myGym/myGym/trained_models/tiago_dual_fix/AG/joints_gripper_ppo/train.json --pretrained_model /home/student/Documents/myGym/myGym/trained_models/tiago_dual_fix/AG/joints_gripper_ppo --eval_episodes 1 --g 1"`

# How to execute trajectory based on GDR-net

## Launch gdrnet

`cd && sh .reasoner_master_script.sh`

## Launch marker_to_targets_file.py

Open a new terminal

`source /opt/ros/noetic/setup.bash`

`source /home/student/code/simon_repos/reasoner_docker_pipeline/noetic_ws/gdrnet_ws/devel/setup.bash`

`export ROS_MASTER_URI=http://tiago-114c:11311 ROS_IP=10.68.0.128`

`cd Documents/myGym/zmq-comm`

`python marker_to_targets_file.py`

The data obtained from GDRnet will flow in terminal.

## Launch code to execute robot
Open a new terminal

`cd Documents/myGym/myGym/ros/docker/`

`docker compose up -d`

`docker compose exec melodic bash`

After enter the docker

`python src/user_packages/joint_control/joint_zmq_server.py --exec-delay 13`

If the adress already in use. exit it and enter again.

It will enter trajectory standby mode.

## Launch code to get trajectory and send to server
Open a new terminal

`cd Documents/myGym/zmq-comm`

`python send_joint_gdrnet_AG_pre.py --test-args "--config /home/student/Documents/myGym/myGym/trained_models/tiago_dual_fix/AG/joints_gripper_ppo/train.json --pretrained_model /home/student/Documents/myGym/myGym/trained_models/tiago_dual_fix/AG/joints_gripper_ppo --eval_episodes 1 --g 1"`(For AGtask)

Use send_joint appropriately as needed.

The TIAGo robot reaches toward the object and grasps it.

