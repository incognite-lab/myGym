#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 14:56:16 2021

@author: syxtreme
"""

import zmq
from threading import Thread, RLock
from warnings import warn
import cloudpickle as cpl
from zmq.backend import zmq_errno
from zmq.utils.strtypes import asbytes
import time
from typing import Callable, Optional, Any, Tuple
from uuid import uuid4


class ParamBaseClient():

    def __init__(self, start_port: int = 25652, addr: str = "0.0.0.0", protocol: str = "tcp", timeout: int = 500):
        """ Creates a client for parameters. The client can subscribe to various parameters
        and then it will receive updates about changes to these parameters.
        The parameters can be accessed as if they were properties of this client.

        Args:
            start_port (int, optional): A port where the parameters server operates. This is a port on which
            only the publisher will send parameter updates. Parameter changes and definitions/declarations
            will be done over ports with numbers +3 and +7 higher. Defaults to 25652.
            addr (str, optional): IP address on which the server operates. Defaults to "0.0.0.0".
            protocol (str, optional): Protocol to use for communication.
        """
        self._timeout = timeout
        self.__context = zmq.Context()

        # subscirbe to parameter changes
        self.__addr_pub = f"{protocol}://{addr}:{str(start_port)}"
        self._subscriber = self.__context.socket(zmq.SUB)
        self._subscriber.connect(self.__addr_pub)
        self._subscriber.setsockopt(zmq.RCVTIMEO, self._timeout)

        self._params = {}
        self.active = True

        # crazy hack
        new_class = type(self.__class__.__name__ + f"_{str(uuid4())}", (self.__class__,), {})
        self.__class__ = new_class

        self.poller_thread = Thread(target=self._poll, daemon=True)
        self.poller_thread.start()

    def wait_for_param(self, param: str, timeout: float = 0) -> Tuple[bool, dict]:
        msg = self.wait_receive_param(param, timeout)
        if msg is None:
            return False, None
        return True, msg

    def wait_receive_param(self, param: str, timeout: float = 0) -> Optional[dict]:
        self._subscriber.setsockopt(zmq.SUBSCRIBE, param.encode('utf-8'))
        if timeout > 0:
            self._subscriber.setsockopt(zmq.RCVTIMEO, timeout)
        msg = None
        while True:
            try:
                rcv_param, data = self._subscriber.recv_multipart()
            except zmq.error.Again:
                print(f"Parameter {param} request timed out!")
                return None
            if rcv_param.decode() == param:
                msg = cpl.loads(data)
                print(param, msg)
                break
        self._subscriber.setsockopt(zmq.UNSUBSCRIBE, param.encode('utf-8'))
        if self._timeout > 0:
            self._subscriber.setsockopt(zmq.RCVTIMEO, self._timeout)
        if msg is not None:
            return msg

    def subscribe(self, param: str) -> None:
        """Subscribe to parameter updates. This does not guarantee that the parameter
        exists and has any value. This only tells the server that this client
        wants to be notified if a parameter with the specified name changes.
        At the begining, the value of the parameter will be "None" and will remain so
        until this client receives value update for the parameter. This will occur
        rather quickly if the parameter exists on the server but this is not known
        at the time this method is called. This should be fine, just be aware of it.

        Args:
            param (str): The name of the parameter
        """
        if hasattr(self, param):
            return

        self._params[param] = None
        setattr(self.__class__, param, property(
                lambda this, param=param: this._params[param],
                lambda this, value, param=param: this.__set_param_extern(param, value),
                lambda this, param=param: this._del_param(param)
            ))
        self._subscriber.setsockopt(zmq.SUBSCRIBE, param.encode('utf-8'))

    def destroy(self):
        """Stops the updates and closes all connection.
        """
        self.active = False

    def __set_param_extern(self, param: str, value: Any) -> None:
        raise ValueError("Parameters are read-only! Use different client if you need to change the parameter.")

    def _set_param(self, param: str, value: Any) -> None:
        self._params[param] = value

    def _del_param(self, param: str):
        if param not in self._params:
            raise ValueError(f"Not subscribed to parameter {param}!")
        self._subscriber.setsockopt(zmq.UNSUBSCRIBE, param.encode('utf-8'))
        delattr(self.__class__, param)
        del self._params[param]

    def _poll(self) -> None:
        while self.active:
            try:
                param, data = self._subscriber.recv_multipart()
            except zmq.Again:
                continue
            msg = cpl.loads(data)
            param = param.decode()
            self._set_param(param, msg)
        self._subscriber.close()

    def __contains__(self, param: str):
        return param in self._params

    def __len__(self):
        return len(self._params)


class ParamSubscriber(ParamBaseClient):
    """The same as ParamBaseClient, but calls a function.
    when receiving a parameter.
    """

    def __init__(self, start_port=25652, addr="0.0.0.0", protocol="tcp", callback: Callable = None):
        super().__init__(start_port, addr, protocol)
        self._cb = callback if callback else lambda para, msg: None  # default "empty" callback function

    def set_callback(self, cb: Callable) -> None:
        self._cb = cb

    def _set_param(self, param: str, value: Any) -> None:
        self._cb(param, value)
        return super()._set_param(param, value)
