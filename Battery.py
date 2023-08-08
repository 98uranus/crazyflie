
import logging
from math import degrees
from operator import pos, truediv
from os import times
from struct import unpack
import sys
import time

import numpy as np
from numpy.core.numeric import zeros_like

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper
from cflib.utils.multiranger import Multiranger
from numpy.core.shape_base import block

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.core.shape_base import block

from timeit import default_timer as timer
from datetime import timedelta


# Logging 초기값 설정
# URI = 'radio://0/80/2M/E7E7E7E703'
URI = uri_helper.uri_from_env(default='radio://0/90/2M/E7E7E7E702')
if len(sys.argv) > 1:
    URI = sys.argv[1]

logging.basicConfig(level=logging.ERROR)



class Battery:  # drone의 현재 위치 Logging
    A = 'pm.vbat'
    B = 'pm.batteryLevel'
    C = 'pm.state'

    def __init__(self, crazyflie, rate_ms=100):
        if isinstance(crazyflie, SyncCrazyflie):
            self._cf = crazyflie.cf
            
        else:
            self._cf = crazyflie
        self._log_config = self._create_log_config(rate_ms)

        self._bat = None
        self._bat_lev = None
        self._bat_state = None
        
        
    def _create_log_config(self, rate_ms):
        log_config = LogConfig('Stabilizer', rate_ms)
        log_config.add_variable(self.A)
        log_config.add_variable(self.B)
        log_config.add_variable(self.C)
        log_config.data_received_cb.add_callback(self._data_received)

        return log_config

    def _data_received(self, timestamp, data, logconf):
        self._bat = data[self.A]
        self._bat_lev = data[self.B]
        self._bat_state = data[self.C]

    @property
    def bat(self):
        return self._bat
    @property
    def bat_lev(self):
        return self._bat_lev
    @property
    def bat_state(self):
        return self._bat_state
    def __enter__(self):
        self.start()
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    def start(self):
        self._cf.log.add_config(self._log_config)
        self._log_config.start()
    def stop(self):
        self._log_config.delete()

def param_deck_flow(name, value_str):
    value = int(value_str)
    print(value)
    global is_deck_attached
    if value:
        is_deck_attached = True
        print('Deck is attached!')
    else:
        is_deck_attached = False
        print('Deck is NOT attached!')

if __name__ == '__main__':
    cflib.crtp.init_drivers()
    cf = Crazyflie(rw_cache='./cache')
    with SyncCrazyflie(URI, cf=cf) as scf:
        scf.cf.param.add_update_callback(group="deck", name="bcFlow",  # flowdeck v2를 사용할 경우 이 라인 제거
                                cb=param_deck_flow)
        time.sleep(1)
        with Battery(scf) as Bat:
            while True:
                # battery = BT.bat
                xxx = Bat.bat
                bbb = Bat.bat_lev
                ccc = Bat.bat_state

                # print(battery)
                print(xxx, bbb, ccc)