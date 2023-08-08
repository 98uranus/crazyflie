# 2-D Grid Map
# 수동조종
import logging
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
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from numpy.core.shape_base import block

# Logging 초기값 설정
URI = uri_helper.uri_from_env(default='radio://0/70/2M/E7E7E7E704')
if len(sys.argv) > 1:
    URI = sys.argv[1]

logging.basicConfig(level=logging.ERROR)

# Grid map 전역변수
EXTEND_AREA = 1.0
Sensor_Maxrange = 2.0


def bresenham(start, end): # Bresenham의 직선 알고리즘
    # setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1
    is_steep = abs(dy) > abs(dx)  # determine how steep the line is
    if is_steep:  # rotate line
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    # swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
    dx = x2 - x1  # recalculate differentials
    dy = y2 - y1  # recalculate differentials
    error = int(dx / 2.0)  # calculate error
    y_step = 1 if y1 < y2 else -1
    # iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = [y, x] if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += y_step
            error += dx
    if swapped:  # reverse the list if the coordinates were swapped
        points.reverse()
    points = np.array(points)
    return points

def calc_grid_map_config(xy_resolution):
    min_x = round(-2 - Sensor_Maxrange )
    min_y = round(-2 - Sensor_Maxrange )
    max_x = round(2 + Sensor_Maxrange )
    max_y = round(2 + Sensor_Maxrange )
    xw = int(round((max_x - min_x) / xy_resolution))
    yw = int(round((max_y - min_y) / xy_resolution))
    print("The grid map is ", xw, "x", yw, ".")
    return min_x, min_y, max_x, max_y, xw, yw

def ox_and_oy(poses, measures):
    OX = np.zeros((1,4))   # OX = [ox1, ox2, ox3, ox4]
    OY = np.zeros((1,4))
    for i in [1, 2, 3, 4]:
        ox = np.sin(poses[2]+np.pi*(i-1)/2) * measures[i-1] + poses[0]
        oy = np.cos(poses[2]+np.pi*(i-1)/2) * measures[i-1] + poses[1]
        OX[0,i-1] = ox
        OY[0,i-1] = oy
    return OX, OY

def cell_number(poses, ox, oy, min_x, min_y, xy_resolution):
    icx = int(round((poses[0] - min_x)/xy_resolution))
    icy = int(round((poses[1] - min_y)/xy_resolution))

    IX = np.zeros((1,4))  
    IY = np.zeros((1,4))
    for i in [1, 2, 3, 4]:
        ix = int(round((ox[0,i-1] - min_x) / xy_resolution))
        iy = int(round((oy[0,i-1] - min_y) / xy_resolution))
        IX[0,i-1] = ix   # IX = [ix1, ix2, ix3, ix4]
        IY[0,i-1] = iy
        IX = IX.astype('int32') # 정수로 변환
        IY = IY.astype('int32')
    return icx, icy, IX, IY

def generate_ray_casting_grid_map(poses, measures, xy_resolution, occupancy_map):
    
    min_x, min_y, max_x, max_y, x_w, y_w = calc_grid_map_config(xy_resolution)
    ox, oy = ox_and_oy(poses, measures)
    icx, icy, ix, iy = \
        cell_number(poses, ox, oy, min_x, min_y, xy_resolution)

    laser_beams1 = bresenham((icx, icy), (
        ix[0, 0], iy[0, 0]))  # line form the lidar to the occupied point
    laser_beams2 = bresenham((icx, icy), (
        ix[0, 1], iy[0, 1]))
    laser_beams3 = bresenham((icx, icy), (
        ix[0, 2], iy[0, 2]))
    laser_beams4 = bresenham((icx, icy), (
        ix[0, 3], iy[0, 3]))
    laser_beams = np.vstack((laser_beams1, laser_beams2, laser_beams3, laser_beams4))

    for i in range(1,len(laser_beams)):
        occupancy_map[laser_beams[i,0], laser_beams[i,1]] = 0.0
    
    for i in [0, 1, 2, 3]:
        if ix[0,i] == icx and iy[0,i] == icy:
            occupancy_map[ix[0, i]][iy[0, i]] = 0.0  
            occupancy_map[ix[0, i] + 1][iy[0, i]] = 0.0  
            occupancy_map[ix[0, i]][iy[0, i] + 1] = 0.0  
            occupancy_map[ix[0, i] + 1][iy[0, i] + 1] = 0.0  
        else:
            occupancy_map[ix[0, i]][iy[0, i]] = 1.0  # occupied area 1.0
            occupancy_map[ix[0, i] + 1][iy[0, i]] = 1.0  # extend the occupied area
            occupancy_map[ix[0, i]][iy[0, i] + 1] = 1.0  # extend the occupied area
            occupancy_map[ix[0, i] + 1][iy[0, i] + 1] = 1.0  # extend the occupied area

    occupancy_map[icx][icy] = -0.4   # 로봇의 Location plotting

    return occupancy_map, xy_resolution
    

class State:  # drone의 현재 위치 Logging
    X = 'stateEstimate.x'
    Y = 'stateEstimate.y'
    Z = 'stateEstimate.z'

    def __init__(self, crazyflie, rate_ms=100):
        if isinstance(crazyflie, SyncCrazyflie):
            self._cf = crazyflie.cf
            
        else:
            self._cf = crazyflie
        self._log_config = self._create_log_config(rate_ms)

        self._x_pos = None
        self._y_pos = None
        self._z_pos = None
        
        
    def _create_log_config(self, rate_ms):
        log_config = LogConfig('Position', rate_ms)
        log_config.add_variable(self.X)
        log_config.add_variable(self.Y)
        log_config.add_variable(self.Z)
        log_config.data_received_cb.add_callback(self._data_received)

        return log_config

    def _data_received(self, timestamp, data, logconf):
        self._x_pos = data[self.X]
        self._y_pos = data[self.Y]
        self._z_pos = data[self.Z]

    @property
    def Xpos(self):
        return self._x_pos
    @property
    def Ypos(self):
        return self._y_pos
    @property
    def Zpos(self):
        return self._z_pos
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


def updatefig(*args):
    global xy_resolution, occupancy_map
    global scf, multiranger, Ste

    if Ste.Xpos != None and Ste.Ypos != None\
        and Ste.Zpos != None:
        poses = np.array(
                    [Ste.Xpos,
                    Ste.Ypos,
                    Ste.Zpos])
    
    if multiranger.front != None and multiranger.left != None\
        and multiranger.back != None and multiranger.right != None:
        measures = np.array([
        multiranger.front,
        multiranger.left,
        multiranger.back,
        multiranger.right])
        occupancy_map, xy_resolution = \
            generate_ray_casting_grid_map(poses, measures, xy_resolution, occupancy_map)
        im = plt.imshow(occupancy_map, animated=True, cmap="PuOr")  # Occupancy map을 이미지 파일로 변환
    else:
        occupancy_map = np.full((200,200),0.5)
        im = plt.imshow(occupancy_map, animated=True, cmap="PuOr")
    
    return im,

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



def is_close(range):
    MIN_DISTANCE = 1  # m

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


def front_left(range_f, range_l):
    
    if(range_f < 0.3 and range_l < 0.3):
        motion_commander.turn_right(90)


if __name__ == '__main__':
    cflib.crtp.init_drivers()
    # Figure 초기값 설정
    fig, ax = plt.subplots()

    #Occupancy Grid 초기값 설정
    xy_resolution = 0.05  # x-y grid resolution
    min_x, min_y, max_x, max_y, x_w, y_w = calc_grid_map_config(xy_resolution)
    occupancy_map = np.ones((x_w, y_w)) / 2  # Occupancy map Initializing
    
    cf = Crazyflie(rw_cache='./cache')
    with SyncCrazyflie(URI, cf=cf) as scf:
        scf.cf.param.add_update_callback(group="deck", name="bcFlow",  # flowdeck v2를 사용할 경우 이 라인 제거
                                cb=param_deck_flow)
        time.sleep(1)
        with State(scf) as Ste:
            with MotionCommander(scf) as motion_commander:
                with Multiranger(scf) as multiranger:
                    keep_flying = True
                    keep_flying = True
                    time.sleep(3)
                    ani = animation.FuncAnimation(fig, updatefig, interval=10, blit=True) # 이미지 파일 연속재생
                    plt.show()  
                    motion_commander.up(0.5)
                    motion_commander.forward(0.5)
                    time.sleep(3)
                    while keep_flying:
                    
                        # VELOCITY = 0.5
                        # velocity_x = 0.0
                        velocity_y = 0.0
                        velocity_x,keep_flying=is_front(multiranger.front)
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
            