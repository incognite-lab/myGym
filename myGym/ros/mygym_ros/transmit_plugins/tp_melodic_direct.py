import traceback as tb
from zmq_comm.srv import ServiceClient


class RosMelodicTransmitPlugin:
    def __init__(self):
        self.client = ServiceClient()

    def send_data(self, data_dict):
        # data = np.asarray(data, dtype=np.float16).tolist()
        print(f"Sending: {data_dict}")
        ret = self.client.call(data_dict)
        print(f"Response: {ret}")
        return ret
