# 실시간 플로팅 + 그리드맵 작업중
# 해야할 것: 좌표 변환, trace 추가, for문

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

import numpy as np
from numpy.core.numeric import zeros_like

tm = time.localtime(time.time()) # 국제표준시 기준 실제 시간
tm_hour = tm.tm_hour - 9
string = time.strftime('%Y-%m-%d %I:%M:%S %p', tm) # 날짜와 시간 형태 변환

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
    min_y = round(-10 - Sensor_Maxrange )
    max_x = round(30 + Sensor_Maxrange )
    max_y = round(10 + Sensor_Maxrange )
    xw = int(round((max_x - min_x) / xy_resolution))
    yw = int(round((max_y - min_y) / xy_resolution))
    # print("The grid map is ", xw, "x", yw, ".")
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


def generate_ray_casting_grid_map(params, xy_resolution, occupancy_map):

    min_x, min_y, max_x, max_y, x_w, y_w = calc_grid_map_config(xy_resolution)

    poses2 = np.array([
                    params[URI2][0]['position'][0],
                    params[URI2][0]['position'][1],
                    params[URI2][0]['position'][2]
                     ])
    measures2 = np.array([
                    params[URI2][0]['measurement'][0],
                    params[URI2][0]['measurement'][1],
                    params[URI2][0]['measurement'][2],
                    params[URI2][0]['measurement'][3]
                     ])
    poses3 = np.array([
                    params[URI3][0]['position'][0],
                    params[URI3][0]['position'][1],
                    params[URI3][0]['position'][2]
                     ])
    measures3 = np.array([
                    params[URI3][0]['measurement'][0],
                    params[URI3][0]['measurement'][1],
                    params[URI3][0]['measurement'][2],
                    params[URI3][0]['measurement'][3]
                     ])
    poses4 = np.array([
                    params[URI4][0]['position'][0],
                    params[URI4][0]['position'][1],
                    params[URI4][0]['position'][2]
                     ])
    measures4 = np.array([
                    params[URI4][0]['measurement'][0],
                    params[URI4][0]['measurement'][1],
                    params[URI4][0]['measurement'][2],
                    params[URI4][0]['measurement'][3]
                     ])



    # ox1, oy1 = ox_and_oy(poses1, measures1)
    ox2, oy2 = ox_and_oy(poses2, measures2)
    ox3, oy3 = ox_and_oy(poses3, measures3)
    ox4, oy4 = ox_and_oy(poses4, measures4)

    
    # icx1, icy1, ix1, iy1 = \
    #     cell_number(poses1, ox1, oy1, min_x, min_y, xy_resolution)
    icx2, icy2, ix2, iy2 = \
        cell_number(poses2, ox2, oy2, min_x, min_y, xy_resolution)
    icx3, icy3, ix3, iy3 = \
        cell_number(poses3, ox3, oy3, min_x, min_y, xy_resolution)
    icx4, icy4, ix4, iy4 = \
        cell_number(poses4, ox4, oy4, min_x, min_y, xy_resolution)

    laser_beams2 = np.empty((1, 2))
    laser_beams2 = laser_beams2.astype('int32')
    for i in range(4):
        laser = bresenham((icx2, icy2), (ix2[0, i], iy2[0, i]))
        laser_beams2 = np.vstack((laser_beams2, laser))

    laser_beams3 = np.empty((1, 2))
    laser_beams3 = laser_beams3.astype('int32')
    for i in range(4):
        laser = bresenham((icx3, icy3), (ix3[0, i], iy3[0, i]))
        laser_beams3 = np.vstack((laser_beams3, laser))

    laser_beams4 = np.empty((1, 2))
    laser_beams4 = laser_beams4.astype('int32')
    for i in range(4):
        laser = bresenham((icx4, icy4), (ix4[0, i], iy4[0, i]))
        laser_beams4 = np.vstack((laser_beams4, laser))

    for i in range(1,len(laser_beams2)):
        occupancy_map[laser_beams2[i,0], laser_beams2[i,1]] = 0.0
    for i in range(1,len(laser_beams3)):
        occupancy_map[laser_beams3[i,0], laser_beams3[i,1]] = 0.0
    for i in range(1,len(laser_beams4)):
        occupancy_map[laser_beams4[i,0], laser_beams4[i,1]] = 0.0


    # 4번 반복
    for i in [0, 1, 2, 3]:
            if ix2[0,i] == icx2 and iy2[0,i] == icy2:
                occupancy_map[ix2[0, i]][iy2[0, i]] = 0.0  
                occupancy_map[ix2[0, i] + 1][iy2[0, i]] = 0.0  
                occupancy_map[ix2[0, i]][iy2[0, i] + 1] = 0.0  
                occupancy_map[ix2[0, i] + 1][iy2[0, i] + 1] = 0.0  
            else:
                occupancy_map[ix2[0, i]][iy2[0, i]] = 1.0  # occupied area 1.0
                occupancy_map[ix2[0, i] + 1][iy2[0, i]] = 1.0  # extend the occupied area
                occupancy_map[ix2[0, i]][iy2[0, i] + 1] = 1.0  # extend the occupied area
                occupancy_map[ix2[0, i] + 1][iy2[0, i] + 1] = 1.0  # extend the occupied area

    for i in [0, 1, 2, 3]:
            if ix3[0,i] == icx3 and iy3[0,i] == icy3:
                occupancy_map[ix3[0, i]][iy3[0, i]] = 0.0  
                occupancy_map[ix3[0, i] + 1][iy3[0, i]] = 0.0  
                occupancy_map[ix3[0, i]][iy3[0, i] + 1] = 0.0  
                occupancy_map[ix3[0, i] + 1][iy3[0, i] + 1] = 0.0  
            else:
                occupancy_map[ix3[0, i]][iy3[0, i]] = 1.0  # occupied area 1.0
                occupancy_map[ix3[0, i] + 1][iy3[0, i]] = 1.0  # extend the occupied area
                occupancy_map[ix3[0, i]][iy3[0, i] + 1] = 1.0  # extend the occupied area
                occupancy_map[ix3[0, i] + 1][iy3[0, i] + 1] = 1.0  # extend the occupied area

    for i in [0, 1, 2, 3]:
            if ix4[0,i] == icx4 and iy4[0,i] == icy4:
                occupancy_map[ix4[0, i]][iy4[0, i]] = 0.0  
                occupancy_map[ix4[0, i] + 1][iy4[0, i]] = 0.0  
                occupancy_map[ix4[0, i]][iy4[0, i] + 1] = 0.0  
                occupancy_map[ix4[0, i] + 1][iy4[0, i] + 1] = 0.0  
            else:
                occupancy_map[ix4[0, i]][iy4[0, i]] = 1.0  # occupied area 1.0
                occupancy_map[ix4[0, i] + 1][iy4[0, i]] = 1.0  # extend the occupied area
                occupancy_map[ix4[0, i]][iy4[0, i] + 1] = 1.0  # extend the occupied area
                occupancy_map[ix4[0, i] + 1][iy4[0, i] + 1] = 1.0  # extend the occupied area

    occupancy_map[icx2][icy2] = -0.4   # 로봇의 Location plotting
    occupancy_map[icx3][icy3] = -0.4   # 로봇의 Location plotting
    occupancy_map[icx4][icy4] = -0.4   # 로봇의 Location plotting

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


def turn_L(degree, mc, scf):
    rate = - 360 / 10
    turning = True
    
    mc.stop()
    if degree > 0:
        # scf.cf.commander.send_hover_setpoint(0.3, 0.8, rate, 0.4)
        mc._set_vel_setpoint(0.2, 0.6, 0.0, rate)
        trace = True    # 왼쪽 following 드론은 왼쪽 돌 때만 true
        # time.sleep(0.2)
        
    elif degree <= 0:
        mc.start_turn_right()
        trace = False # 오른쪽은 false
        # time.sleep(0.2)
        rate = 360 / 10
    
    return rate, turning, trace

def turn_R(degree, mc, scf):
    rate = 360 / 10
    turning = True
    
    mc.stop()
    if degree > 0:
        # mc._set_vel_setpoint(0.3, 0.6, 0.0, rate)
        mc.start_turn_left()
        trace = False
        # time.sleep(0.1) 
        
    elif degree <= 0:
        
        # scf.cf.commander.send_hover_setpoint(0.3, -0.8, rate, 0.4)
        mc._set_vel_setpoint(0.2, -0.6, 0.0, rate)
        trace = True 
        # time.sleep(0.2)
        rate = 360 / 10
    
    return rate, turning, trace


def log_position(scf, ppp):
    global params
    uri = scf.cf.link_uri

    # # Figure 초기값 설정
    if uri == "radio://0/100/2M/E7E7E7E701":
        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        ims = []
        hposes2 = np.zeros((1,3))
        hox2 = np.zeros((1,4))
        hoy2 = np.zeros((1,4))
        hposes3 = np.zeros((1,3))
        hox3 = np.zeros((1,4))
        hoy3 = np.zeros((1,4))
        hposes4 = np.zeros((1,3))
        hox4 = np.zeros((1,4))
        hoy4 = np.zeros((1,4))

        #Occupancy Grid 초기값 설정
        xy_resolution = 0.05  # x-y grid resolution
        min_x, min_y, max_x, max_y, x_w, y_w = calc_grid_map_config(xy_resolution)
        occupancy_map = np.ones((x_w, y_w)) / 2  # Occupancy map Initializing


    with MotionCommander(scf) as mc:
        with State(scf) as Ste:
            with Multiranger(scf) as multiranger:
                with Battery(scf) as Bat:

                    keep_flying = True
                    turning = False
                    trace = True
                    # TT = True
                    velocity = 0.7
                    velocity_x = 0.0
                    velocity_y = 0.0
                    rate = 0
                    if uri != "radio://0/100/2M/E7E7E7E701":
                        if keep_flying:
                                mc.up(0.1)
                                time.sleep(1)


                    while keep_flying:
                        # if uri != "radio://0/100/2M/E7E7E7E701":
                            front = multiranger.front
                            left = multiranger.left
                            back = multiranger.back
                            right = multiranger.right
                            up = multiranger.up

                            Xpos = Ste.Xpos
                            Ypos = Ste.Ypos
                            Yaw = Ste.Yawpos

                            if uri == "radio://0/70/2M/E7E7E7E704":
                                Xpos = Xpos - 1.8

                            ppp['measurement'] = [front, left, back, right]
                            ppp['position'] = [Xpos, Ypos, Yaw]
                            
                            # print("First params ={}".format(params, uri))
                            # bat = Bat.bat
                            # bat_lev = Bat._bat_lev
                            if front != None and left != None\
                                    and back != None and right != None:
                                if front < 0.5:
                                    velocity_x = -0.3
                                    velocity_y =  0.0
                                if back < 0.3:
                                    velocity_x = +0.2
                                    velocity_y =  0.0
                                if left < 0.3:
                                    velocity_y = -0.3
                                if right < 0.3:
                                    velocity_y = +0.3


                            if uri == "radio://0/80/2M/E7E7E7E702": # main은 motion commander
                                mc._set_vel_setpoint(velocity_x, velocity_y, 0, rate)

                            else: # 나머지는 commander
                                scf.cf.commander.send_hover_setpoint(velocity_x, velocity_y, rate, 0.4)

                            
                            if trace == True:
                                ppp['trace_x'].append(Xpos)
                                ppp['trace_y'].append(Ypos)

                                # params[URI2][0]['trace_y'][-1] = (params[URI2][0]['trace_y'])[-1] -1.8
                                params[URI4][0]['trace_x'][-1] = (params[URI4][0]['trace_x'])[-1] -1.8
                                trace = False
                                time.sleep(0.1)
                            

                            if turning:
                                if velocity_x != 0 or velocity_y != 0: 
                                    time.sleep(0.2) # 충돌방지 작동
                                
                                if uri == "radio://0/80/2M/E7E7E7E702" or uri == "radio://0/70/2M/E7E7E7E704":
                                    if left < 0.7:
                                        mc.stop()
                                        turning = False
                                        rate = 0
                                if uri == "radio://0/90/2M/E7E7E7E703":
                                    if right < 0.7:
                                        mc.stop()
                                        turning = False
                                        rate = 0

                            # if up < 0.2 or bat < 3.0:  # 드론 위를 가리면 Landing
                            if up < 0.2 :
                                    keep_flying = False
                                    trace = True
                                    # trace = True

                            else:
                                if uri == "radio://0/80/2M/E7E7E7E702" or uri == "radio://0/70/2M/E7E7E7E704":
                                    if left > 0.8: # 왼쪽 벽이 멀면


                                        # # for i in [
                                        # #     #'radio://0/100/2M/E7E7E7E701',
                                        # #     # 'radio://0/90/2M/E7E7E7E702',
                                        # #     'radio://0/80/2M/E7E7E7E703',
                                        # #     'radio://0/70/2M/E7E7E7E704'
                                        #         # ]:
                                        if uri == "radio://0/80/2M/E7E7E7E702":
                                            for j in range(len(params['radio://0/70/2M/E7E7E7E704'][0]['trace_x'])):
                                                if (
                                                        (Xpos-params['radio://0/70/2M/E7E7E7E704'][0]['trace_x'][j])**2
                                                        +
                                                        (Ypos-params['radio://0/70/2M/E7E7E7E704'][0]['trace_y'][j])**2
                                                        )**(1/2) < 0.8:

                                                    # print(abs((
                                                    #     (Xpos-params['radio://0/80/2M/E7E7E7E703'][0]['trace_x'][j])**2
                                                    #     +
                                                    #     (Ypos-params['radio://0/80/2M/E7E7E7E703'][0]['trace_y'][j])**2
                                                    #     )**(1/2)), uri)

                                                    velocity_x = velocity + 0.3
                                                    velocity_y = 0.0
                                                    # mc.stop()
                                                    turning = False
                                                    trace = False
                                                    rate = 0
                                                    # mc.land()
                                                    
                                                    break
                                                else:
                                                    
                                                    rate, turning, trace = turn_L(90, mc, scf)
                                                    velocity_x = 0
                                            print(string, uri)
                                        else :
                                            for j in range(len(params['radio://0/80/2M/E7E7E7E702'][0]['trace_x'])):
                                                if (
                                                        (Xpos-params['radio://0/80/2M/E7E7E7E702'][0]['trace_x'][j])**2
                                                        +
                                                        (Ypos-params['radio://0/80/2M/E7E7E7E702'][0]['trace_y'][j])**2
                                                        )**(1/2) < 0.8:

                                                    # print(abs((
                                                    #     (Xpos-params['radio://0/80/2M/E7E7E7E703'][0]['trace_x'][j])**2
                                                    #     +
                                                    #     (Ypos-params['radio://0/80/2M/E7E7E7E703'][0]['trace_y'][j])**2
                                                    #     )**(1/2)), uri)

                                                    velocity_x = velocity + 0.3
                                                    velocity_y = 0.0
                                                    # mc.stop()
                                                    turning = False
                                                    trace = False
                                                    rate = 0
                                                    # mc.land()
                                                    
                                                    break
                                                else:
                                                    
                                                    rate, turning, trace = turn_L(90, mc, scf)
                                                    velocity_x = 0
                                            print(string, uri)
                                                


                                    else: # 왼쪽 벽이 가까우면
                                        if front < 0.5: # 앞에 장애물이 있으면
                                            # print(front)
                                            rate, turning, trace = turn_L(-90, mc, scf)
                                            velocity_x = 0
                                        else: # 앞에 장애물이 없으면
                                            if left < 0.3:
                                                velocity_x = velocity
                                                velocity_y = -0.3
                                                # print('close')
                                            if 0.3 <= left < 0.5:
                                                velocity_x = velocity
                                                velocity_y = 0.0
                                                # print('center')
                                            if 0.5 <= left < 0.8:
                                                velocity_x = velocity
                                                velocity_y = +0.3
                                                # print('far')

                                if uri == "radio://0/90/2M/E7E7E7E703":
                                    if right > 0.8: # 오른쪽 벽이 멀면
                                        rate, turning, trace = turn_R(-90, mc, scf)
                                        velocity_x = 0
                                    else: #오른쪽 벽이 가까우면
                                        if front < 0.5: # 앞에 장애물이 있으면
                                            rate, turning, trace = turn_R(90, mc, scf)
                                            velocity_x = 0
                                        else: # 앞에 장애물이 없으면
                                            if right < 0.3:
                                                velocity_x = velocity
                                                velocity_y = +0.3
                                            if 0.3 <= right < 0.5:
                                                velocity_x = velocity
                                                velocity_y = 0.0
                                            if 0.5 <= right < 0.8:
                                                velocity_x = velocity
                                                velocity_y = -0.3
                                                
                                if uri == "radio://0/100/2M/E7E7E7E701":
                                    print("Second params ={}".format(params, uri))
                                    # params[URI2][0]['position'][1] = (params[URI2][0]['position'][1]) -1.8
                                    # params[URI4][0]['position'][0] = (params[URI4][0]['position'][0]) -1.8

                                    
                                    # for i in range(len(params[URI2][0]['trace_y'])):
                                    #     params[URI2][0]['trace_y'][i] = (params[URI2][0]['trace_y'])[i] -1.8
                                    print("Second params ={}".format(params, uri))
                                    occupancy_map = \
                                        generate_ray_casting_grid_map(params, xy_resolution, occupancy_map)
                                    im = ax1.imshow(occupancy_map.T, origin='lower', animated=True, cmap="PuOr")
                                    print("params ={}".format(params, uri))
                                    
                                    ims.append([im])
                                    time.sleep(0.05)


                                
                                        # Real-time plotting

                                    poses2 = np.array([
                                                        params[URI2][0]['position'][0],
                                                        params[URI2][0]['position'][1]-1.8,
                                                        params[URI2][0]['position'][2]
                                                        ])
                                    poses3 = np.array([
                                                        params[URI3][0]['position'][0],
                                                        params[URI3][0]['position'][1],
                                                        params[URI3][0]['position'][2]
                                                        ])

                                    poses4 = np.array([
                                                        params[URI4][0]['position'][0]-1.8,
                                                        params[URI4][0]['position'][1],
                                                        params[URI4][0]['position'][2]
                                                        ])

                                    measures2 = np.array([
                                                        params[URI2][0]['measurement'][0],
                                                        params[URI2][0]['measurement'][1],
                                                        params[URI2][0]['measurement'][2],
                                                        params[URI2][0]['measurement'][3]
                                                        ])
                                    measures3 = np.array([
                                                        params[URI3][0]['measurement'][0],
                                                        params[URI3][0]['measurement'][1],
                                                        params[URI3][0]['measurement'][2],
                                                        params[URI3][0]['measurement'][3]
                                                        ])

                                    measures4 = np.array([
                                                        params[URI4][0]['measurement'][0],
                                                        params[URI4][0]['measurement'][1],
                                                        params[URI4][0]['measurement'][2],
                                                        params[URI4][0]['measurement'][3]
                                                        ])
                                    hposes2 = np.vstack((hposes2, poses2))
                                    hposes3 = np.vstack((hposes3, poses3))
                                    hposes4 = np.vstack((hposes4, poses4))

                                    ox2, oy2 = ox_and_oy(poses2, measures2)
                                    ox3, oy3 = ox_and_oy(poses3, measures3)
                                    ox4, oy4 = ox_and_oy(poses4, measures4)

                                    hox2, hoy2 = np.vstack((hox2, ox2)), np.vstack((hoy2, oy2))
                                    hox3, hoy3 = np.vstack((hox3, ox3)), np.vstack((hoy3, oy3))
                                    hox4, hoy4 = np.vstack((hox4, ox4)), np.vstack((hoy4, oy4))

                                    plt.cla()
                                    ax2.plot(hposes2[:, 0],  # plot ground truth
                                            hposes2[:, 1], "-r", label='position2')
                                    ax2.plot(hposes3[:, 0],  # plot ground truth
                                            hposes3[:, 1], "-r", label='position3')
                                    ax2.plot(hposes4[:, 0],  # plot ground truth
                                            hposes4[:, 1], "-r", label='position4')
                                    
                                    for i in range(4):
                                        ax2.plot(hox2[:, i],    # plot pose(with noisy input)
                                                hoy2[:, i], ".k", label='measurements2')
                                    for i in range(4):
                                        ax2.plot(hox3[:, i],    # plot pose(with noisy input)
                                                hoy3[:, i], ".k", label='measurements3')
                                    for i in range(4):
                                        ax2.plot(hox4[:, i],    # plot pose(with noisy input)
                                                hoy4[:, i], ".k", label='measurements4')

                                    for i in [
                                            #'radio://0/100/2M/E7E7E7E701',
                                            'radio://0/90/2M/E7E7E7E702',
                                            'radio://0/80/2M/E7E7E7E703',
                                            'radio://0/70/2M/E7E7E7E704'
                                                ]:

                                        ax2.plot(params[i][0]['trace_x'], params[i][0]['trace_y'], "xg", label='trace')
                            
                                    ax2.legend()
                                    ax2.axis("equal")
                                    ax2.grid(True)
                                    plt.pause(0.001)
                        
                        

                    # for i in range(len(params["radio://0/80/2M/E7E7E7E703"][0]['trace_y'])):
                    #     print(params["radio://0/80/2M/E7E7E7E703"][0]['trace_x'][i],
                    #             params["radio://0/80/2M/E7E7E7E703"][0]['trace_y'][i])

                    mc.land()

                    if uri == "radio://0/100/2M/E7E7E7E701":
                        # plt.close(fig2)

                        ani = animation.ArtistAnimation(fig1, ims, interval=100, repeat_delay=2000)
                        ani.save('test7.gif', fps=5000, dpi=80)   # .gif 로 저장
                        plt.show()


def reset_estimator(scf):
    cf = scf.cf
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(2)

if __name__ == '__main__':
    # Initialize drivers
    cflib.crtp.init_drivers(enable_debug_driver=False)
    


    URI1 = 'radio://0/100/2M/E7E7E7E701'
    URI2 = 'radio://0/80/2M/E7E7E7E702'
    URI3 = 'radio://0/80/2M/E7E7E7E703'   
    URI4 = 'radio://0/70/2M/E7E7E7E704'

    uris = [
            URI1,
            URI2,
            URI3,
            URI4
            ]

    params1 = {'trace': True, 'trace_x':[], 'trace_y':[]}
    params2 = {'trace': True, 'trace_x':[], 'trace_y':[]}
    params3 = {'trace': True, 'trace_x':[], 'trace_y':[]}
    params4 = {'trace': True, 'trace_x':[], 'trace_y':[]}

    params = {
        URI1: [params1],
        URI2: [params2],
        URI3: [params3],
        URI4: [params4],
    }


    factory = CachedCfFactory(rw_cache='./cache')
    with Swarm(uris, factory=factory) as swarm:
        swarm.parallel(reset_estimator)
        swarm.parallel_safe(log_position, args_dict=params)