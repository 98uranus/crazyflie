import logging
from math import degrees
from operator import pos, truediv
from os import times
from struct import unpack
import sys
import time
import csv
import os
import pandas as pd

import numpy as np

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils import uri_helper
from numpy.core.shape_base import block
from cflib.utils.multiranger import Multiranger
from numpy.core.shape_base import block

# Logging 초기값 설정
# URI = 'radio://0/80/2M/E7E7E7E703'
URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E704')
if len(sys.argv) > 1:
    URI = sys.argv[1]

logging.basicConfig(level=logging.ERROR)

sequence = [
    (0, 0, 0.6, 0),
    (0.5/np.sqrt(2), 0, 0.6, 0),
    (0, 0.5/np.sqrt(2), 0.6, 0),
    (-0.5/np.sqrt(2), 0, 0.6, 0),
    (0, -0.5/np.sqrt(2), 0.6, 0),
    (0.5/np.sqrt(2), 0, 0.6, 0),
    (0, 0, 0.6, 0),
    (0, 0, 0.2, 0)
]

class Acc:  # drone의 현재 위치 Logging
    A = 'acc.x'
    B = 'acc.y'
    C = 'acc.z'

    def __init__(self, crazyflie, rate_ms=100):
        if isinstance(crazyflie, SyncCrazyflie):
            self._cf = crazyflie.cf
            
        else:
            self._cf = crazyflie
        self._log_config = self._create_log_config(rate_ms)

        self._acc_x = None
        self._acc_y = None
        self._acc_z = None
        
    def _create_log_config(self, rate_ms):
        log_config = LogConfig('Stabilizer', rate_ms)
        log_config.add_variable(self.A)
        log_config.add_variable(self.B)
        log_config.add_variable(self.C)
        log_config.data_received_cb.add_callback(self._data_received)

        return log_config

    def _data_received(self, timestamp, data, logconf):
        self._acc_x = data[self.A]
        self._acc_y = data[self.B]
        self._acc_z = data[self.C]

    @property
    def acc_x(self):
        return self._acc_x
    @property
    def acc_y(self):
        return self._acc_y
    @property
    def acc_z(self):
        return self._acc_z
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

class Gyro:  # drone의 현재 위치 Logging
    A = 'gyro.x'
    B = 'gyro.y'
    C = 'gyro.z'

    def __init__(self, crazyflie, rate_ms=100):
        if isinstance(crazyflie, SyncCrazyflie):
            self._cf = crazyflie.cf
            
        else:
            self._cf = crazyflie
        self._log_config = self._create_log_config(rate_ms)

        self._gyro_x = None
        self._gyro_y = None
        self._gyro_z = None
        
    def _create_log_config(self, rate_ms):
        log_config = LogConfig('Stabilizer', rate_ms)
        log_config.add_variable(self.A)
        log_config.add_variable(self.B)
        log_config.add_variable(self.C)
        log_config.data_received_cb.add_callback(self._data_received)

        return log_config

    def _data_received(self, timestamp, data, logconf):
        self._gyro_x = data[self.A]
        self._gyro_y = data[self.B]
        self._gyro_z = data[self.C]

    @property
    def gyro_x(self):
        return self._gyro_x
    @property
    def gyro_y(self):
        return self._gyro_y
    @property
    def gyro_z(self):
        return self._gyro_z
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

class PosEstAlt:  # drone의 현재 위치 Logging
    A = 'posEstAlt.estimatedZ'
    B = 'posEstAlt.estVZ'
    C = 'posEstAlt.velocityZ'

    def __init__(self, crazyflie, rate_ms=100):
        if isinstance(crazyflie, SyncCrazyflie):
            self._cf = crazyflie.cf
            
        else:
            self._cf = crazyflie
        self._log_config = self._create_log_config(rate_ms)

        self._estimatedZ = None
        self._estVZ = None
        self._velocityZ = None
        
    def _create_log_config(self, rate_ms):
        log_config = LogConfig('Stabilizer', rate_ms)
        log_config.add_variable(self.A)
        log_config.add_variable(self.B)
        log_config.add_variable(self.C)
        log_config.data_received_cb.add_callback(self._data_received)

        return log_config

    def _data_received(self, timestamp, data, logconf):
        self._estimatedZ = data[self.A]
        self._estVZ = data[self.B]
        self._velocityZ = data[self.C]

    @property
    def estimatedZ(self):
        return self._estimatedZ
    @property
    def estVZ(self):
        return self._estVZ
    @property
    def velocityZ(self):
        return self._velocityZ
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



def acc_gyro_log(state, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z) : 
    
    with open('acc_gyro_log.csv', 'a', newline='') as f : 
        # Accelerometer data
        a = acc_x
        b = acc_y
        c = acc_z
        # Gyroscope data
        d = gyro_x
        e = gyro_y
        g = gyro_z
        wr = csv.writer(f)
        wr.writerow([state,a,b,c,d,e,g])

        f.close()

def is_close(range):
    MIN_DISTANCE = 0.4  # m

    if range is None:
        return False
    else:
        return range < MIN_DISTANCE

if __name__ == '__main__':
    cflib.crtp.init_drivers()
    cf = Crazyflie(rw_cache='./cache')
    with SyncCrazyflie(URI, cf=cf) as scf:
        scf.cf.param.add_update_callback(group="deck", name="bcFlow",  # flowdeck v2를 사용할 경우 이 라인 제거
                                cb=param_deck_flow)
        with MotionCommander(scf) as mc:
            with Multiranger(scf) as multi_ranger:
                with Acc(scf) as acc:
                    with Gyro(scf) as gyro:
                        keep_flying = True
                        while keep_flying:
                            VELOCITY = 0.4
                            velocity_x = 0.0
                            velocity_y = 0.0
                            state = 0

                            # Accelerometer data [g's] -> [m/s^2]
                            a = acc.acc_x
                            b = acc.acc_y
                            c = acc.acc_z
                            # Gyroscope data [deg/s]
                            d = gyro.gyro_x
                            e = gyro.gyro_y
                            f = gyro.gyro_z

                            acc_gyro_log(state,a,b,c,d,e,f)

                            if is_close(multi_ranger.front):
                                velocity_x -= VELOCITY
                                state = 1
                                acc_gyro_log(state,a,b,c,d,e,f)

                            if is_close(multi_ranger.back):
                                velocity_x += VELOCITY
                                state = 2
                                acc_gyro_log(state,a,b,c,d,e,f)

                            if is_close(multi_ranger.left):
                                velocity_y -= VELOCITY
                                state = 3
                                acc_gyro_log(state,a,b,c,d,e,f)

                            if is_close(multi_ranger.right):
                                velocity_y += VELOCITY
                                state = 4
                                acc_gyro_log(state,a,b,c,d,e,f)

                            if is_close(multi_ranger.up):
                                keep_flying = False
                                state = 5
                                acc_gyro_log(state,a,b,c,d,e,f)

                            mc.start_linear_motion(
                                velocity_x, velocity_y, 0)

                            time.sleep(0.1)
                                                        
                            
                            
                    

                    

                    
                
                    
                    