import os
import time
import datetime
import logging
import cfclient
import sys

import numpy as np

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper
from cflib.utils.multiranger import Multiranger
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie


# Logging 초기값 설정
URI = uri_helper.uri_from_env(default='radio://0/70/2M/E7E7E7E704')
if len(sys.argv) > 1:
    URI = sys.argv[1]

logging.basicConfig(level=logging.ERROR)

__author__ = 'Bitcraze AB'
__all__ = ['LogWriter']

logger = logging.getLogger(__name__)

class LogWriter():
    """Create a writer for a specific log block"""
    vbat = 'pm.vbat'

    def __init__(self, logblock, connected_ts=None, directory=None):
        """Initialize the writer"""
        self._block = logblock
        self._dir = directory
        self._connected_ts = connected_ts

        self._dir = os.path.join(cfclient.config_path, "logdata",
                                 connected_ts.strftime("%Y%m%dT%H-%M-%S"))
        self._file = None
        self._header_written = False
        self._header_values = []
        self._filename = None

    def _write_header(self):
        """Write the header to the file"""
        if not self._header_written:
            s = "Timestamp"
            for v in self._block.variables:
                s += "," + v.name
                self._header_values.append(v.name)
            s += '\n'
            self._file.write(s)
            self._header_written = True

    def _new_data(self, timestamp, data, logconf):
        """Callback when new data arrives from the Crazyflie"""
        if self._file:
            s = "%d" % timestamp
            for col in self._header_values:
                s += "," + str(data[col])
            s += '\n'
            self._file.write(s)

    def writing(self):
        """Return True if the file is open and we are using it,
        otherwise false"""
        return True if self._file else False

    def stop(self):
        """Stop the logging to file"""
        if self._file:
            self._file.close()
            self._file = None
            self._block.data_received_cb.remove_callback(self._new_data)
            logger.info("Stopped logging of block [%s] to file [%s]",
                        self._block.name, self._filename)
            self._header_values = []
            self._header_written = False

    def start(self):
        """Start the logging to file"""

        # Due to concurrency let's not check first, just create
        try:
            os.makedirs(self._dir)
        except OSError:
            logger.debug("logdata directory already exists")

        if not self._file:
            time_now = datetime.datetime.now()
            block_name_corr = self._block.name.replace('/', '-')
            name = "{0}-{1}.csv".format(block_name_corr,
                                        time_now.strftime(
                                            "%Y%m%dT%H-%M-%S"))
            self._filename = os.path.join(self._dir, name)
            self._file = open(self._filename, 'w')
            self._write_header()
            self._block.data_received_cb.add_callback(self._new_data)
            logger.info("Started logging of block [%s] to file [%s]",
                        self._block.name, self._filename)

if __name__ == '__main__':
    cflib.crtp.init_drivers()

    cf = Crazyflie(rw_cache='./cache')
    with SyncCrazyflie(URI, cf=cf) as scf:

        time.sleep(1)
        with LogWriter(scf) as Lwr:
        #    with MotionCommander(scf) as motion_commander: # 이 라인을 활성화할 경우 갑자기 이륙할 수 있으므로 주의
                with Multiranger(scf) as multiranger: 
                    keep_flying = True