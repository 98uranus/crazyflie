# 2-d map, Occupancy Grid Map
# Wall following
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
    min_x = round(-8 - Sensor_Maxrange )
    min_y = round(-8 - Sensor_Maxrange )
    max_x = round(8 + Sensor_Maxrange )
    max_y = round(8 + Sensor_Maxrange )
    xw = int(round((max_x - min_x) / xy_resolution))
    yw = int(round((max_y - min_y) / xy_resolution))
    print("The grid map is ", xw, "x", yw, ".")
    return min_x, min_y, max_x, max_y, xw, yw

def ox_and_oy(poses, measures):
    OX = np.zeros((1,4))   # OX = [ox1, ox2, ox3, ox4]
    OY = np.zeros((1,4))
    for i in [1, 2, 3, 4]:
        ox = np.cos(np.deg2rad(poses[2]+90*(i-1))) * measures[i-1] + poses[0]
        oy = np.sin(np.deg2rad(poses[2]+90*(i-1))) * measures[i-1] + poses[1]
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

    return occupancy_map
    
class State:  # drone의 현재 위치 Logging
    X = 'stateEstimate.x'
    Y = 'stateEstimate.y'
    Yaw = 'stateEstimate.yaw'

    def __init__(self, crazyflie, rate_ms=100):
        if isinstance(crazyflie, SyncCrazyflie):
            self._cf = crazyflie.cf
            
        else:
            self._cf = crazyflie
        self._log_config = self._create_log_config(rate_ms)

        self._x_pos = None
        self._y_pos = None
        self._yaw_pos = None
        
        
    def _create_log_config(self, rate_ms):
        log_config = LogConfig('Position', rate_ms)
        log_config.add_variable(self.X)
        log_config.add_variable(self.Y)
        log_config.add_variable(self.Yaw)
        log_config.data_received_cb.add_callback(self._data_received)

        return log_config

    def _data_received(self, timestamp, data, logconf):
        self._x_pos = data[self.X]
        self._y_pos = data[self.Y]
        self._yaw_pos = data[self.Yaw]

    @property
    def Xpos(self):
        return self._x_pos
    @property
    def Ypos(self):
        return self._y_pos
    @property
    def Yawpos(self):
        return self._yaw_pos
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

def is_close(range):
    MIN_DISTANCE = 1  # m
    if range is None:
        return False
    else:
        return range < MIN_DISTANCE

def is_front(range):
    min_distance = 0.5
    if range > min_distance:
        vel = 0.3
    else:
        vel = 0.0
    return vel

def is_left(range, vel_x):
    min_distance_left = 0.4
    max_distance_left = 0.6
    room_distance_left = 0.9
    if range < min_distance_left:
        vel_y = -0.1
    elif room_distance_left > range > max_distance_left:
        vel_y = 0.1        
    elif range > room_distance_left:
        vel_x = 0  
        vel_y = 0      
    else:
        vel_y = 0
    return vel_x, vel_y

def turn(degree):
    turning = True
    yaw_angle = Ste.Yawpos
    mc.stop()
    if degree > 0:
        mc.start_turn_left()
    elif degree <= 0:
        mc.start_turn_right()

    return turning, yaw_angle, degree


if __name__ == '__main__':
    cflib.crtp.init_drivers()
    # Figure 초기값 설정
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    ims = []
    hpose = np.zeros((1,3))
    hox = np.zeros((1,4))
    hoy = np.zeros((1,4))

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
            with MotionCommander(scf) as mc: # 이 라인을 활성화할 경우 갑자기 이륙할 수 있으므로 주의
                with Multiranger(scf) as multiranger:
                    keep_flying = True
                    turning = False
                    velocity = 0.2

                    if keep_flying:
                        mc.up(0.1)
                        time.sleep(2)

                    while keep_flying:
                        front = multiranger.front
                        left = multiranger.left
                        back = multiranger.back
                        right = multiranger.right
                        up = multiranger.up

                        Xpos = Ste.Xpos
                        Ypos = Ste.Ypos
                        Yaw = Ste.Yawpos

                        if up < 0.2:  # 드론 위를 가리면 Landing
                                keep_flying = False

                        if turning:
                            yaw_t2 = Yaw
                            if abs(degree) > abs(yaw_t2 - yaw_t1):
                                mc.stop()
                                turning = False

                        else:
                            mc.start_forward(velocity)
                            if front < 0.4 :  # 전방 장애물 발생시 우회전
                                # if left < 0.5:
                                    mc.stop()
                                    turning, yaw_t1, degree = turn(-90)
                                # else:
                                #     mc.stop()
                                #     turning, yaw_t1, degree = turn(90)
                            else:
                                if left < 0.3:
                                    mc.start_linear_motion(velocity, -0.1, 0)
                                elif 0.3 <= left < 0.5:
                                    mc.start_linear_motion(velocity, 0, 0)
                                elif 0.5 <= left < 0.7:
                                    mc.start_linear_motion(velocity, 0.1, 0)
                                else:
                                    turning, yaw_t1, degree = turn(90)

                        if Xpos != None and Ypos != None\
                            and Yaw != None:
                            poses = np.array([Xpos, Ypos, Yaw])
                            hpose = np.vstack((hpose, poses))
                        
                        if front != None and left != None\
                            and back != None and right != None:
                            measures = np.array([front, left, back, right])

                            # Grid map
                            occupancy_map = \
                                    generate_ray_casting_grid_map(poses, measures, xy_resolution, occupancy_map)
                            im = ax1.imshow(occupancy_map.T, origin='lower', animated=True, cmap="PuOr")  # Occupancy map을 이미지 파일로 변환
                        else:
                            occupancy_map = np.full((400,400),0.5)
                            im = ax1.imshow(occupancy_map.T, origin='lower', animated=True, cmap="PuOr")
                        
                        ims.append([im])
                        time.sleep(0.05)
                    
                        # Real-time plotting
                        ox, oy = ox_and_oy(poses, measures)
                        hox = np.vstack((hox, ox))
                        hoy = np.vstack((hoy, oy))
                        
                        plt.cla()
                        plt.gcf().canvas.mpl_connect(   # for stopping simulation with the esc key.
                            'key_release_event',
                            lambda event: [exit(0) if event.key == 'escape' else None])
                        
                        ax2.plot(hpose[:, 0],  # plot ground truth
                                hpose[:, 1], "-r", label='position')
                        for i in range(4):
                            ax2.plot(hox[:, i],    # plot pose(with noisy input)
                                    hoy[:, i], ".k", label='measurements')
                        ax2.legend()
                        ax2.axis("equal")
                        ax2.grid(True)
                        plt.pause(0.001)

                    mc.land()
                    plt.close(fig2)

                    ani = animation.ArtistAnimation(fig1, ims, interval=50, repeat_delay=2000)
                    ani.save('house.gif', fps=10000, dpi=80)   # .gif 로 저장
                    plt.show()