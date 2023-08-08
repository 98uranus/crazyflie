import logging
from math import degrees
from operator import pos, truediv
from os import times
from struct import unpack
import sys
import time

import cflib.crtp
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.swarm import Swarm
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.utils.multiranger import Multiranger
from cflib.positioning.motion_commander import MotionCommander 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.core.shape_base import block
from timeit import default_timer as timer
from datetime import timedelta


def log_position(scf):
    lg_stab = LogConfig(name='position', period_in_ms=100)
    lg_stab.add_variable('stateEstimate.x', 'float')
    lg_stab.add_variable('stateEstimate.y', 'float')
    lg_stab.add_variable('stateEstimate.z', 'float')

    def is_close(range):

        MIN_DISTANCE = 0.5  # m

        if range is None:
            return False
        else:
            return range < MIN_DISTANCE
     
    uri = scf.cf.link_uri
    with SyncLogger(scf, lg_stab) as logger:
        with MotionCommander(scf) as motion_commander:
            with Multiranger(scf) as multi_ranger:
             
                for log_entry in logger:
                    while keep_flying:

                        data = log_entry[1]
                        x_pose = data['stateEstimate.x']
                        y_pose = data['stateEstimate.y']
                        z_pose = data['stateEstimate.z']
                        print(uri, "is at", x_pose, y_pose, z_pose)

                        VELOCITY = 0.5
                        velocity_x = 0.0
                        velocity_y = 0.0

                        if is_close(multi_ranger.front):
                            velocity_x -= VELOCITY
                        if is_close(multi_ranger.back):
                            velocity_x += VELOCITY
                        if is_close(multi_ranger.left):
                            velocity_y -= VELOCITY
                        if is_close(multi_ranger.right):
                            velocity_y += VELOCITY
                        if is_close(multi_ranger.up):
                            keep_flying = False
                 
                        motion_commander.start_linear_motion(
                            velocity_x, velocity_y, 0)
                        time.sleep(0.1)  




if __name__ == '__main__':
    cflib.crtp.init_drivers(enable_debug_driver=False)

    URI1 = 'radio://0/70/2M/E7E7E7E704'   
    # URI2 = 'radio://0/90/2M/E7E7E7E702'

    uris = [

        URI1,
        # URI2       

    ]

    factory = CachedCfFactory(rw_cache='./cache')
    with Swarm(uris, factory=factory) as swarm:
        swarm.parallel_safe(log_position)
