import logging
from math import degrees
from operator import pos, truediv
from os import times
from struct import unpack
import sys
import time
import csv
import os

import numpy as np

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper
from numpy.core.shape_base import block

from numpy.core.shape_base import block

# Logging 초기값 설정
# URI = 'radio://0/80/2M/E7E7E7E703'
URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E704')
if len(sys.argv) > 1:
    URI = sys.argv[1]

logging.basicConfig(level=logging.ERROR)


class Yaw:  # drone의 현재 위치 Logging
    A = 'stabilizer.yaw'

    def __init__(self, crazyflie, rate_ms=100):
        if isinstance(crazyflie, SyncCrazyflie):
            self._cf = crazyflie.cf
            
        else:
            self._cf = crazyflie
        self._log_config = self._create_log_config(rate_ms)

        self._yaw = None

        
        
    def _create_log_config(self, rate_ms):
        log_config = LogConfig('Stabilizer', rate_ms)
        log_config.add_variable(self.A)
        log_config.data_received_cb.add_callback(self._data_received)

        return log_config

    def _data_received(self, timestamp, data, logconf):
        self._yaw = data[self.A]
        
    @property
    def yaw(self):
        return self._yaw
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
        # print('Deck is NOT attached!')
        print('Deck is attached!')

def Yaw_coord(yaw) : 
    
    with open ('yaw_coord.csv', 'a', newline='') as f : 
        tm_min = tm1.tm_min
        tm_sec = tm1.tm_sec
        wr = csv.writer(f)
        wr.writerow([tm_min, tm_sec, yaw])
        
        f.close()

if __name__ == '__main__':
    cflib.crtp.init_drivers()
    cf = Crazyflie(rw_cache='./cache')
    with SyncCrazyflie(URI, cf=cf) as scf:
        scf.cf.param.add_update_callback(group="deck", name="bcFlow",  # flowdeck v2를 사용할 경우 이 라인 제거
                                cb=param_deck_flow)
        with MotionCommander(scf) as mc :
            with Yaw(scf) as yaw:
                keep_flying = True
                # tm = time.localtime(1575142526.500323)
                while keep_flying:
                    while True:
                        tm1 = time.localtime(time.time()) # 국제표준시 기준 실제 시간
                        tm_hour = tm1.tm_hour - 9
                        string = time.strftime('%Y-%m-%d %I:%M:%S %p', tm1) # 날짜와 시간 형태 변환
                        yaw1 = yaw.yaw
                        time.sleep(1)
                        Yaw_coord(yaw1)
                        print(string, yaw1)
                        mc.turn_left(90)
                        time.sleep(1)
                        yaw2 = yaw.yaw
                        Yaw_coord(yaw2)
                        print(string, yaw2)
                        mc.turn_right(90)
                        
                        

                        

                        
                    
                        
                        