from importlib.resources import files
import time
import yaml
from zmq_comm.pub_sub import ParamPublisher


if __name__ == "__main__":
    try:
        yaml_path = files("myGym").joinpath("ros/zmq_config.yaml")
    except ModuleNotFoundError:
        yaml_path = files("mygym_ros").joinpath("../zmq_config.yaml")
    if not yaml_path.exists():
        raise FileNotFoundError(f"ZMQ config file not found at {yaml_path}")
    with yaml_path.open("r") as f:
        _zmq_config = yaml.safe_load(f)

    pb = ParamPublisher(
        **_zmq_config
    )

    for i in range(100):
        pb.publish("robot_action", i)
        print(f"Published {i}")
        time.sleep(0.1)
