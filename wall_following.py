import logging
import sys
import time
import csv
import os
import networkx as nx
import numpy as np
import math

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils.multiranger import Multiranger
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger

URI = 'radio://0/70/2M/E7E7E7E704'

if len(sys.argv) > 1:
    URI = sys.argv[1]

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

# Bitcraze
def position_callback ( timestamp , data , logconf ) :
    x = data ['kalman . stateX ']
    y = data ['kalman . stateY ']
    z = data ['kalman . stateZ ']
    print ( 'pos : ({} , {} , {})'.format (x, y, z))
    with open ('position.csv ', 'a') as csvfile :
        writer = csv.writer ( csvfile , delimiter =',')
        writer.writerow ([ x , y , z ])
    csvfile.close ()

# Bitcraze
def start_position_printing ( scf ) :
    log_conf = LogConfig ( name ='Position ', period_in_ms =500)
    log_conf.add_variable ('kalman . stateX ', 'float ')
    log_conf.add_variable ('kalman . stateY ', 'float ')
    log_conf.add_variable ('kalman . stateZ ', 'float ')
    scf.cf.log.add_config ( log_conf )
    log_conf.data_received_cb.add_callback ( position_callback )
    log_conf.start ()


def is_close(range):
    MIN_DISTANCE = 0.5  # m
    if range is None:
        return False
    else:
        return range < MIN_DISTANCE

def is_front(range):
    min_distance=0.15
    #print(range)
    if range > min_distance:
        vel=0.1
        kf=True
    else:
        vel = 0.0
        kf=True
    return vel,kf

def is_left(range, vel_x):
    min_distance_left = 0.2
    max_distance_left = 0.3
    #vel = 0.2
    print(range)
    if range < min_distance_left:
        vel_y = -0.1
        vel_x = 0

    
    elif range> max_distance_left:
        vel_y = 0.1
        vel_x = 0
    else:
        vel_y = 0
    
    return vel_x, vel_y

def front_left2(range_f, range_l):
    
    if(range_f > 0.3 and range_l > 0.3):
        motion_commander.turn_left(90)


def front_left(range_f, range_l):
    
    if(range_f < 0.3 and range_l < 0.3):
        motion_commander.turn_right(90)





if __name__ == '__main__':
    # Initialize the low-level drivers (don't list the debug drivers)
    cflib.crtp.init_drivers()
    #c = 0
    cf = Crazyflie(rw_cache='./cache')
    with SyncCrazyflie(URI, cf=cf) as scf:
        with MotionCommander(scf) as motion_commander:
            with Multiranger(scf) as multiranger:
                keep_flying = True
                time.sleep(3)
                motion_commander.up(0.5)
                motion_commander.forward(0.5)
                time.sleep(3)
                while keep_flying:
	            
                    # VELOCITY = 0.5
                    # velocity_x = 0.0
                    velocity_y = 0.0
                    velocity_x, keep_flying = is_front(multiranger.front)
                    velocity_x, velocity_y = is_left(multiranger.left, velocity_x)

                    front_left(multiranger.front, multiranger.left)

                    print(velocity_x, velocity_y)
                    #c = c+1

		            
                    if multiranger.up < 0.7:
                        keep_flying = True
                    #     velocity_x -= VELOCITY
                    # if is_close(multiranger.back):
                    #     velocity_x += VELOCITY

                    # if is_close(multiranger.left):
                    #     velocity_y -= VELOCITY
                    # if is_close(multiranger.right):
                    #     velocity_y += VELOCITY

                    # #if  c == 50:
                    #     #keep_flying = False

                    motion_commander.start_linear_motion(velocity_x, velocity_y, 0)
                    time.sleep(0.1)
                
                motion_commander.down(0.2)
                time.sleep(2)
            print('Demo terminated!')