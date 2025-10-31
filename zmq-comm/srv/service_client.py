#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 26 14:56:16 2021

@author: syxtreme
"""
from __future__ import unicode_literals
import zmq
from threading import Thread, Lock
from warnings import warn
import cloudpickle as cpl
from zmq.utils.strtypes import asbytes

class SyFuture(object):

    def __init__(self):
        self.__done = False
        self.__success = False
        self.__result = None

    @property
    def done(self):
        return self.__done

    @property
    def success(self):
        return self.__success

    def get_result(self):
        if not self.done:
            return None
        return self.__result

    def set_result(self, result=None):
        self.__success = result is not None
        self.__result = result
        self.__done = True


class ServiceClient():
    REQUEST_TIMEOUT = 3000  # in ms (how long to wait to send the request - doesn't do much...)
    RESPONSE_TIMEOUT = 3000  # in ms (how long to wait to receive a response)

    def __init__(self, port=242424, addr="0.0.0.0", protocol="tcp"):
        """Create service client.

        Args:
            port (int, optional): Port for the service server. Defaults to 242424.
            addr (str, optional): IP address of the service server. Defaults to "0.0.0.0".
            protocol (str, optional): Transport protocol to be used. Defaults to "tcp".
        """
        self.__context = zmq.Context()  # ZMQ context
        self.__addr = "{protocol}://{addr}:{port}".format(protocol=protocol, addr=addr, port=str(port))  # full address ~ sort of like a service name/identifier
        print("Creating service client on {}".format(self.__addr))
        self._connect()
        self.__lock = Lock()

    def _connect(self):
        """Create a new ZMQ client, connect to the service server and set socket options
        """
        self._zmq_client = self.__context.socket(zmq.REQ)
        self._zmq_client.connect(self.__addr)
        self._zmq_client.setsockopt(zmq.SNDTIMEO, self.REQUEST_TIMEOUT)
        self._zmq_client.setsockopt(zmq.RCVTIMEO, self.RESPONSE_TIMEOUT)

    def _reconnect(self):
        """Reconnect after error (e.g., service timeout) otherwise socket in weird state = will not work
        """
        self._zmq_client.disconnect(self.__addr)
        self._connect()

    def destroy(self):
        self._zmq_client.close()

    def call(self, request):
        result = None
        self.__lock.acquire()
        try:
            # 1) pickle and send the request
            request = self._convert_dict_to_unicode(request)
            self._zmq_client.send(cpl.dumps(request, protocol=2))
        except zmq.Again:  # timeout when sending (should not happen, unless ZMQ error)
            self._reconnect()
        else:
            try:
                # 2) wait and receive the response and unpickle it
                result = cpl.loads(self._zmq_client.recv())
            except zmq.Again:  # response did not arrive in time
                self._reconnect()
        # return the response
        self.__lock.release()
        return result

    @staticmethod
    def _convert_dict_to_unicode(indict):
        d = {k.encode("latin1"): v for k, v in indict.items()}
        # print(d)
        return d

    def call_async(self, request):
        self.__lock.acquire()
        try:
            # pickle and send the request
            request = self._convert_dict_to_unicode(request)
            self._zmq_client.send(cpl.dumps(request, protocol=2))
        except zmq.Again:  # timeout when sending (should not happen, unless ZMQ error)
            self._reconnect()
            self.__lock.release()
            return None
        handle = SyFuture()
        th = Thread(target=self.__wait_for_response, daemon=True, kwargs={"future": handle, "lock": self.__lock})
        th.start()
        return handle

    def __wait_for_response(self, future, lock):
        result = None
        try:
            # wait and receive the response
            result = cpl.loads(self._zmq_client.recv())
        except zmq.Again:  # response did not arrive in time
            self._reconnect()
        finally:
            lock.release()
        future.set_result(result)
