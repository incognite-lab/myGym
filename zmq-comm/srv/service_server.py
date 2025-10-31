#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 14:56:16 2021

@author: syxtreme
"""

import zmq
from threading import Thread
from warnings import warn
import cloudpickle as cpl
from zmq.utils.strtypes import asbytes


class ServiceServer():
    SEND_TIMEOUT = 3000  # in ms (how long to wait to send the request - doesn't do much...)
    RECEIVE_TIMEOUT = 3000  # in ms (how long to wait to receive a response)

    def __init__(self, callback, port=242424, addr="0.0.0.0", protocol="tcp"):
        """Create a service server. Callback should handle the incoming requests.
        A request will be a dictionary with some data (has to be agreed upon externally).
        Port should serve as a service identifier (if more services are used).

        Args:
            callback (function): Request handler.
            port (int, optional): This is the main service identifier. Defaults to 242424.
            addr (str, optional): Address of the service server. Defaults to "127.0.0.1".
            protocol (str, optional): Protocol to use, keep on default. Defaults to "tcp".
        """
        self.__context = zmq.Context()
        self.__addr = f"{protocol}://{addr}:{str(port)}"   # full address ~ sort of like a service name/identifier
        print(f"Creating service on {self.__addr} (if localhost/loopback address is used, service will be visible to localhost only)")  # use actual ip addr for network visibility
        # bind the ZMQ socket
        self._connect()

        self._callback = callback
        self.__active = True
        # thread to wait for requests
        self.poller_thread = Thread(target=self.__poll, daemon=True)
        self.poller_thread.start()

    def _reconnect(self):
        """Reconnect after error (e.g., service timeout) otherwise socket in weird state = will not work
        """
        print("Someone messed up and I had to reconnect the ZMQ socket!")
        self._zmq_socket.close(self.__addr)
        self._connect()

    def _connect(self):
        self._zmq_socket = self.__context.socket(zmq.REP)
        self._zmq_socket.setsockopt(zmq.SNDTIMEO, self.SEND_TIMEOUT)
        self._zmq_socket.setsockopt(zmq.RCVTIMEO, self.RECEIVE_TIMEOUT)
        self._zmq_socket.bind(self.__addr)

    def destroy(self):
        self.__active = False
        self.poller_thread.join()
        self._zmq_socket.close()

    @staticmethod
    def _convert_dict_to_unicode(indict):
        return {asbytes(k): v for k, v in indict.items()}

    def __poll(self):
        while self.__active:
            try:
                request = self._zmq_socket.recv()  # wait for a request
            except zmq.Again:
                continue
            # unpickle and send to callback
            request_dict = cpl.loads(request, encoding="latin1")
            try:
                response_dict = self._callback(request_dict)
            except Exception as e:
                # if the callback rises unhandled error, send empty dict
                print(f"Error in the service callback:\n{e}")
                response_dict = {}
            # pickle the response and send back to the caller
            # response_dict = self._convert_dict_to_unicode(response_dict)
            response = cpl.dumps(response_dict, protocol=2)
            try:
                self._zmq_socket.send(response)
            except zmq.Again:
                self._reconnect()
