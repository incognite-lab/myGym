#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 14:19:55 2021

@author: syxtreme
"""

import zmq
from threading import Thread
from warnings import warn
import cloudpickle as cpl



class ParamPublisher():

    def __init__(self, start_port=25652, addr="0.0.0.0", protocol="tcp"):
        """ Creates a server for parameters. ParamClient nodes can subscribe to parameter updates
        handled via this server. If one client changes a parameter, other clients receive and update.

        Args:
            start_port (int, optional): A port on which this server operates. This is a port on which
            only the publisher will send parameter updates. Client parameter updates and definitions/declarations
            will be done over ports with numbers +3 and +7 higher. Defaults to 25652.
            addr (str, optional): IP address on which this server operates. See PyZMQ documentation
            for various options (e.g. "localhost", "*", etc.). Defaults to "0.0.0.0".
            protocol (str, optional): Protocol to use for communication. Probably don't change this,
            unless you really know what you are doing. Defaults to "tcp".
        """
        self.__context = zmq.Context()

        # socket to publish parameters
        self.__addr_pub = f"{protocol}://{addr}:{str(start_port)}"
        self._publisher = self.__context.socket(zmq.PUB)
        self._publisher.bind(self.__addr_pub)

    def publish(self, param, value):
        self._publisher.send_multipart([param.encode('utf-8'), cpl.dumps(value)])
