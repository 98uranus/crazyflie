# 3대 동시 비행
# 2,4번은 Left following 3번은 Right
# 4번이 Main

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


# from a_star import *

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
    min_x = round(-10 - Sensor_Maxrange )
    min_y = round(-10 - Sensor_Maxrange )
    max_x = round(10 + Sensor_Maxrange )
    max_y = round(10 + Sensor_Maxrange )
    xw = int(round((max_x - min_x) / xy_resolution))
    yw = int(round((max_y - min_y) / xy_resolution))
    # print("The grid map is ", xw, "x", yw, ".")
    return min_x, min_y, max_x, max_y, xw, yw

def ox_and_oy(poses, measures):
    OX = np.zeros((1,4))   # OX = [ox1, ox2, ox3, ox4]
    OY = np.zeros((1,4))
    for i in [1, 2, 3, 4]:
        OX[0,i-1] = np.cos(np.deg2rad(poses[2]+90*(i-1))) * measures[i-1] + poses[0]
        OY[0,i-1] = np.sin(np.deg2rad(poses[2]+90*(i-1))) * measures[i-1] + poses[1]
    return OX, OY

def cell_number(poses, ox, oy, min_x, min_y, xy_resolution):
    icx = int(round((poses[0] - min_x)/xy_resolution))
    icy = int(round((poses[1] - min_y)/xy_resolution))

    IX = np.zeros((1,4))  
    IY = np.zeros((1,4))
    for i in [1, 2, 3, 4]:
        IX[0,i-1] = int(round((ox[0,i-1] - min_x) / xy_resolution))   # IX = [ix1, ix2, ix3, ix4]
        IY[0,i-1] = int(round((oy[0,i-1] - min_y) / xy_resolution))
        IX = IX.astype('int32') # 정수로 변환
        IY = IY.astype('int32')
    return icx, icy, IX, IY

def generate_ray_casting_grid_map(params, xy_resolution, occupancy_map):
    min_x, min_y, max_x, max_y, x_w, y_w = calc_grid_map_config(xy_resolution)

    # poses2 = [params[URI2][0]['position'][i] for i in range(3)]
    # poses3 = [params[URI3][0]['position'][i] for i in range(3)]
    poses4 = [params[URI4][0]['position'][i] for i in range(3)]
    # measures2 = [params[URI2][0]['measurement'][i] for i in range(4)]
    # measures3 = [params[URI3][0]['measurement'][i] for i in range(4)]
    measures4 = [params[URI4][0]['measurement'][i] for i in range(4)]


    # ox1, oy1 = ox_and_oy(poses1, measures1)
    # ox2, oy2 = ox_and_oy(poses2, measures2)
    # ox3, oy3 = ox_and_oy(poses3, measures3)
    ox4, oy4 = ox_and_oy(poses4, measures4)

    # icx1, icy1, ix1, iy1 = \
    #     cell_number(poses1, ox1, oy1, min_x, min_y, xy_resolution)
    # icx2, icy2, ix2, iy2 = \
    #     cell_number(poses2, ox2, oy2, min_x, min_y, xy_resolution)
    # icx3, icy3, ix3, iy3 = \
    #     cell_number(poses3, ox3, oy3, min_x, min_y, xy_resolution)
    icx4, icy4, ix4, iy4 = \
        cell_number(poses4, ox4, oy4, min_x, min_y, xy_resolution)

    # laser_beams2 = np.empty((1, 2)).astype('int32')
    # for i in range(4):
    #     laser = bresenham((icx2, icy2), (ix2[0, i], iy2[0, i]))
    #     laser_beams2 = np.vstack((laser_beams2, laser))

    # laser_beams3 = np.empty((1, 2)).astype('int32')
    # for i in range(4):
    #     laser = bresenham((icx3, icy3), (ix3[0, i], iy3[0, i]))
    #     laser_beams3 = np.vstack((laser_beams3, laser))

    laser_beams4 = np.empty((1, 2)).astype('int32')
    for i in range(4):
        laser = bresenham((icx4, icy4), (ix4[0, i], iy4[0, i]))
        laser_beams4 = np.vstack((laser_beams4, laser))


    # for i in range(1,len(laser_beams2)):
    #     for j in range(3):
    #         for k in range(3):
    #             occupancy_map[laser_beams2[i,0] + j, laser_beams2[i,1] + k] = 0.0
    # for i in range(1,len(laser_beams3)):
    #     for j in range(3):
    #         for k in range(3):
    #             occupancy_map[laser_beams3[i,0] + j, laser_beams3[i,1] + k] = 0.0
    for i in range(1,len(laser_beams4)):
        for j in range(3):
            for k in range(3):
                occupancy_map[laser_beams4[i,0] + j, laser_beams4[i,1] + k] = 0.0


    #     # 4번 반복
    # for i in [0, 1, 2, 3]:
    #         if ix2[0,i] == icx2 and iy2[0,i] == icy2:
    #             occupancy_map[ix2[0, i]][iy2[0, i]] = 0.0  
    #             occupancy_map[ix2[0, i] + 1][iy2[0, i]] = 0.0  
    #             occupancy_map[ix2[0, i]][iy2[0, i] + 1] = 0.0  
    #             occupancy_map[ix2[0, i] + 1][iy2[0, i] + 1] = 0.0  
    #         else:
    #             occupancy_map[ix2[0, i]][iy2[0, i]] = 1.0  # occupied area 1.0
    #             occupancy_map[ix2[0, i] + 1][iy2[0, i]] = 1.0  # extend the occupied area
    #             occupancy_map[ix2[0, i]][iy2[0, i] + 1] = 1.0  # extend the occupied area
    #             occupancy_map[ix2[0, i] + 1][iy2[0, i] + 1] = 1.0  # extend the occupied area

    # for i in [0, 1, 2, 3]:
    #         if ix3[0,i] == icx3 and iy3[0,i] == icy3:
    #             occupancy_map[ix3[0, i]][iy3[0, i]] = 0.0  
    #             occupancy_map[ix3[0, i] + 1][iy3[0, i]] = 0.0  
    #             occupancy_map[ix3[0, i]][iy3[0, i] + 1] = 0.0  
    #             occupancy_map[ix3[0, i] + 1][iy3[0, i] + 1] = 0.0  
    #         else:
    #             occupancy_map[ix3[0, i]][iy3[0, i]] = 1.0  # occupied area 1.0
    #             occupancy_map[ix3[0, i] + 1][iy3[0, i]] = 1.0  # extend the occupied area
    #             occupancy_map[ix3[0, i]][iy3[0, i] + 1] = 1.0  # extend the occupied area
    #             occupancy_map[ix3[0, i] + 1][iy3[0, i] + 1] = 1.0  # extend the occupied area

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
    

    # occupancy_map[icx2][icy2] = -0.4   # 로봇의 Location plotting
    # occupancy_map[icx3][icy3] = -0.4   # 로봇의 Location plotting
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
    rate = - 360 / 20
    turning = True
    if degree > 0:
        # scf.cf.commander.send_hover_setpoint(0.3, 0.8, rate, 0.4)
        mc._set_vel_setpoint(0.0, 1.0, 0.0, rate)
        trace = True    # 왼쪽 following 드론은 왼쪽 돌 때만 true
        time.sleep(0.2)
        
    elif degree <= 0:
        mc.start_turn_right()
        trace = False # 오른쪽은 false
        rate = 360 / 10
    
    return rate, turning, trace

def turn_R(degree, mc, scf):
    rate = 360 / 10
    turning = True
    if degree > 0:
        mc.start_turn_left()
        trace = False
        
    elif degree <= 0:
        # scf.cf.commander.send_hover_setpoint(0.3, -0.8, rate, 0.4)
        mc._set_vel_setpoint(0.1, -0.7, 0.0, rate)
        trace = True 
        rate = 360 / 10
    
    return rate, turning, trace

def log_position(scf, ppp):
    global params
    uri = scf.cf.link_uri

    # # Figure 초기값 설정
    if uri == "radio://0/70/2M/E7E7E7E704":
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
                    velocity = 0.5
                    # if uri == "radio://0/70/2M/E7E7E7E704":
                    #     velocity = 0.4
                    velocity_x = 0.0
                    velocity_y = 0.0
                    rate = 0

                    if keep_flying:
                            mc.up(0.1)
                            time.sleep(1)

                    while keep_flying:
                        front = multiranger.front
                        left = multiranger.left
                        back = multiranger.back
                        right = multiranger.right
                        up = multiranger.up

                        Xpos = Ste.Xpos
                        Ypos = Ste.Ypos
                        Yaw = Ste.Yawpos
                        
                        # if uri == "radio://0/80/2M/E7E7E7E703":
                        #     Xpos = -(Xpos - 0.9)
                        #     Ypos = -Ypos
                        #     Yaw = Yaw + 180

                        # if uri == "radio://0/80/2M/E7E7E7E7E7":
                        #     Xpos = (Xpos + 2.6)
                        #     Ypos = Ypos
                        #     Yaw = Yaw

                        ppp['measurement'] = [front, left, back, right]
                        ppp['position'] = [Xpos, Ypos, Yaw]
                        
                        if front != None and left != None\
                                and back != None and right != None:
                            if front < 0.3:
                                velocity_x = -0.4
                                velocity_y =  0.0
                            if back < 0.3:
                                velocity_x = +0.4
                                velocity_y =  0.0
                            if left < 0.3:
                                velocity_y = -0.4
                            if right < 0.3:
                                velocity_y = +0.4

                        if uri == "radio://0/70/2M/E7E7E7E704": # main은 motion commander
                            mc._set_vel_setpoint(velocity_x, velocity_y, 0, rate)

                        else: # 나머지는 commander
                            scf.cf.commander.send_hover_setpoint(velocity_x, velocity_y, rate, 0.4)

                        # if up < 0.2 or bat < 3.0:  # 드론 위를 가리면 Landing
                        if up < 0.2 :
                                keep_flying = False
                                trace = True
                                mc.stop()
                        
                        if turning:
                            if velocity_x != 0 or velocity_y != 0: 
                                time.sleep(0.2) # 충돌방지 작동
                            
                            if uri == "radio://0/80/2M/E7E7E7E7E7" or uri == "radio://0/70/2M/E7E7E7E704":
                                if left < 0.7:
                                    mc.stop()
                                    turning = False
                                    rate = 0

                            if uri == "radio://0/80/2M/E7E7E7E703":
                                if right < 0.7:
                                    mc.stop()
                                    turning = False
                                    rate = 0

                        else:
                            if uri == "radio://0/80/2M/E7E7E7E7E7" or uri == "radio://0/70/2M/E7E7E7E704":
                                if left > 0.8: # 왼쪽 벽이 멀면
                                    # if uri == "radio://0/80/2M/E7E7E7E7E7":
                                    #     for j in range(len(params['radio://0/70/2M/E7E7E7E704'][0]['trace_x'])):
                                    #         if (
                                    #                 (Xpos-params['radio://0/70/2M/E7E7E7E704'][0]['trace_x'][j])**2
                                    #                 +
                                    #                 (Ypos-params['radio://0/70/2M/E7E7E7E704'][0]['trace_y'][j])**2
                                    #                 )**(1/2) < 1.2:

                                    #             velocity_x = velocity + 0.3
                                    #             velocity_y = -0.2
                                    #             trace = False
                                    #             rate = 0
                                    #             break
                                    #         else:
                                    #             rate, turning, trace = turn_L(90, mc, scf)
                                    #             velocity_x = 0
                                    #             velocity_y = 0
                                    # else :
                                    #     for j in range(len(params["radio://0/80/2M/E7E7E7E7E7"][0]['trace_x'])):
                                    #         if (
                                    #                 (Xpos-params["radio://0/80/2M/E7E7E7E7E7"][0]['trace_x'][j])**2
                                    #                 +
                                    #                 (Ypos-params["radio://0/80/2M/E7E7E7E7E7"][0]['trace_y'][j])**2
                                    #                 )**(1/2) < 1.2:
                                    #             if front < 0.4: # 앞에 장애물이 있으면
                                    #                 rate, turning, trace = turn_L(-90, mc, scf)
                                    #                 velocity_x = 0
                                    #                 velocity_y = 0
                                    #             else:
                                    #                 velocity_x = velocity + 0.3
                                    #                 velocity_y = -0.2
                                    #                 trace = False
                                    #                 rate = 0
                                    #                 break
                                    #         else:
                                                rate, turning, trace = turn_L(90, mc, scf)
                                                velocity_x = 0
                                                velocity_y = 0
                                    
                                else: # 왼쪽 벽이 가까우면
                                    if front < 0.5: # 앞에 장애물이 있으면
                                        rate, turning, trace = turn_L(-90, mc, scf)
                                        velocity_x = 0
                                        velocity_y = 0
                                    else: # 앞에 장애물이 없으면
                                        if left < 0.2:
                                            velocity_x = velocity
                                            velocity_y = -0.2
                                            # print('close')
                                        if 0.2 <= left < 0.5:
                                            velocity_x = velocity
                                            velocity_y = 0.0
                                            # print('center')
                                        if 0.5 <= left < 0.8:
                                            velocity_x = velocity
                                            velocity_y = +0.2
                                            # print('far')

                            if uri == "radio://0/80/2M/E7E7E7E703":
                                if right > 0.8: # 오른쪽 벽이 멀면
                                    rate, turning, trace = turn_R(-90, mc, scf)
                                    velocity_x = 0
                                    velocity_y = 0
                                else: #오른쪽 벽이 가까우면
                                    if front < 0.5: # 앞에 장애물이 있으면
                                        rate, turning, trace = turn_R(90, mc, scf)
                                        velocity_x = 0
                                        velocity_y = 0
                                    else: # 앞에 장애물이 없으면
                                        if right < 0.2:
                                            velocity_x = velocity
                                            velocity_y = +0.2
                                        if 0.2 <= right < 0.5:
                                            velocity_x = velocity
                                            velocity_y = 0.0
                                        if 0.5 <= right < 0.8:
                                            velocity_x = velocity
                                            velocity_y = -0.2

                        if trace == True:
                            ppp['trace_x'].append(Xpos)
                            ppp['trace_y'].append(Ypos)
                            trace = False

                        if front != None and left != None\
                                and back != None and right != None:

                            if uri == "radio://0/70/2M/E7E7E7E704": # 4번 드론만 Grid Map 작성 



                                start = time.time()  # 시작 시간 저장
                                
                                # params[URI2][0]['position'][1] = (params[URI2][0]['position'][1]) -1.8
                                # params[URI4][0]['position'][0] = (params[URI4][0]['position'][0]) -1.8

                                
                                # for i in range(len(params[URI2][0]['trace_y'])):
                                #     params[URI2][0]['trace_y'][i] = (params[URI2][0]['trace_y'])[i] -1.8

                                occupancy_map = \
                                            generate_ray_casting_grid_map(params, xy_resolution, occupancy_map)
                                im = ax1.imshow(occupancy_map.T, origin='lower', animated=True, cmap="PuOr")
                                
                                ims.append([im])

                                ppp['oc'] = occupancy_map
                                
                                time.sleep(0.05)


                                # Real-time plotting
                                # poses2 = [params[URI2][0]['position'][i] for i in range(3)]
                                # poses3 = [params[URI3][0]['position'][i] for i in range(3)]
                                # poses4 = [params[URI4][0]['position'][i] for i in range(3)]
                                # measures2 = [params[URI2][0]['measurement'][i] for i in range(4)]
                                # measures3 = [params[URI3][0]['measurement'][i] for i in range(4)]
                                # measures4 = [params[URI4][0]['measurement'][i] for i in range(4)]

                            
                                # poses2 = np.array([
                                #                     params[URI2][0]['position'][0],
                                #                     params[URI2][0]['position'][1],
                                #                     params[URI2][0]['position'][2]
                                #                     ])
                                # poses3 = np.array([
                                #                     params[URI3][0]['position'][0],
                                #                     params[URI3][0]['position'][1],
                                #                     params[URI3][0]['position'][2]
                                #                     ])
                                poses4 = np.array([
                                                    params[URI4][0]['position'][0],
                                                    params[URI4][0]['position'][1],
                                                    params[URI4][0]['position'][2]
                                                    ])
                                # measures2 = np.array([
                                #                     params[URI2][0]['measurement'][0],
                                #                     params[URI2][0]['measurement'][1],
                                #                     params[URI2][0]['measurement'][2],
                                #                     params[URI2][0]['measurement'][3]
                                #                     ])
                                # measures3 = np.array([
                                #                     params[URI3][0]['measurement'][0],
                                #                     params[URI3][0]['measurement'][1],
                                #                     params[URI3][0]['measurement'][2],
                                #                     params[URI3][0]['measurement'][3]
                                #                     ])
                                measures4 = np.array([
                                                    params[URI4][0]['measurement'][0],
                                                    params[URI4][0]['measurement'][1],
                                                    params[URI4][0]['measurement'][2],
                                                    params[URI4][0]['measurement'][3]
                                                    ])
                                # hposes2 = np.vstack((hposes2, poses2))
                                # hposes3 = np.vstack((hposes3, poses3))
                                hposes4 = np.vstack((hposes4, poses4))

                                # ox2, oy2 = ox_and_oy(poses2, measures2)
                                # ox3, oy3 = ox_and_oy(poses3, measures3)
                                ox4, oy4 = ox_and_oy(poses4, measures4)

                                # hox2, hoy2 = np.vstack((hox2, ox2)), np.vstack((hoy2, oy2))
                                # hox3, hoy3 = np.vstack((hox3, ox3)), np.vstack((hoy3, oy3))
                                hox4, hoy4 = np.vstack((hox4, ox4)), np.vstack((hoy4, oy4))

                                plt.cla()
                                ax2.plot(hposes2[:, 0],  # plot ground truth
                                        hposes2[:, 1], ".r", label='position2')
                                ax2.plot(hposes3[:, 0],  # plot ground truth
                                        hposes3[:, 1], ".b", label='position3')
                                ax2.plot(hposes4[:, 0],  # plot ground truth
                                        hposes4[:, 1], ".y", label='position4')
                                
                                for i in range(4):
                                    ax2.plot(hox2[:, i],    # plot pose(with noisy input)
                                            hoy2[:, i], ".k",)
                                for i in range(4):
                                    ax2.plot(hox3[:, i],    # plot pose(with noisy input)
                                            hoy3[:, i], ".k")
                                for i in range(4):
                                    ax2.plot(hox4[:, i],    # plot pose(with noisy input)
                                            hoy4[:, i], ".k")

                                for i in [
                                        #'radio://0/100/2M/E7E7E7E701',
                                        # 'radio://0/80/2M/E7E7E7E7E7',
                                        # 'radio://0/80/2M/E7E7E7E703',
                                        'radio://0/70/2M/E7E7E7E704'
                                            ]:

                                    ax2.plot(params[i][0]['trace_x'], params[i][0]['trace_y'], "xg", label='trace')
                        
                                ax2.legend()
                                ax2.axis("equal")
                                ax2.grid(True)
                                plt.pause(0.001)
                                print("time :", time.time() - start)  # 현재시각 - 시작시간 = 실행 시간

                    mc.land()

                    

                    if uri == "radio://0/70/2M/E7E7E7E704":
                        np.savetxt("C:/Users/kangmin/Desktop/PythonWorkspace/20211102.csv",occupancy_map,delimiter=",")
                        plt.close(fig2)

                        ani = animation.ArtistAnimation(fig1, ims, interval=100, repeat_delay=2000)
                        ani.save('20201102.gif', fps=2000, dpi=80)   # .gif 로 저장
                        plt.show()




                # print("Homing start!!")

                # final_xpos = params[uri][0]['trace_x'][-1] # 종료점(Homing 시작지점) 좌표
                # final_ypos = params[uri][0]['trace_y'][-1]

                # initial_xpos = params[uri][0]['trace_x'][0]  # 출발점(t = 0) 좌표
                # initial_ypos = params[uri][0]['trace_y'][0]

                # occupancy_map = params["radio://0/90/2M/E7E7E7E7E7"][0]['oc']



                



                # TARGET_ALTITUDE = 0.3 / 0.0254

                # st = time.time()
            
                # # # occupancy_map은 바로 numpy로 불러오기 (저장X)
                # # occupancy_map = np.loadtxt('C:\python\FCND-projects-crazyflie-port-master\FCND-projects-crazyflie-port-master\map_obstacle_course2.csv', delimiter=',',
                # #                 dtype='float64')

                # G = construct_road_map_crazyflie(occupancy_map,3600, 8)
                        
                # time_taken = time.time() - st
                # print(f'create_graph_and_edges() took: {time_taken} seconds')

                # # start, goal 좌표 추가
                # final_xcel = int(round((final_xpos - min_x)/xy_resolution))
                # final_ycel = int(round((final_ypos - min_y)/xy_resolution))
                # initial_xcel = int(round((initial_xpos - min_x)/xy_resolution))
                # initial_ycel = int(round((initial_ypos - min_y)/xy_resolution))

                # occupancy_map_start = (final_xcel, final_ycel, 0.5)
                # occupancy_map_goal = (initial_xcel, initial_ycel, 0.5)
                # # Find closest node on the graph
                
                # g_start, g_goal = find_start_goal(G, occupancy_map_start, occupancy_map_goal) 
                # print("Start and Goal location:", occupancy_map_start, occupancy_map_goal)
                # print("Start and Goal location on graph:", g_start, g_goal)
                # path, _ = a_star_graph(G, heuristic, g_start, g_goal)
                # path.append(occupancy_map_goal)

                # new_path = []
                # new_path.append(occupancy_map_start)
                # new_path.append(g_start)
                # new_path.extend(path)

                # unique_path = []
                # for p in new_path:
                #     if p not in unique_path:
                #         unique_path.append(p)

                # waypoints = []

                # for p in unique_path:
                #     print(np.array(p) - np.array(occupancy_map_start))
                #     offset = (np.array(p) - np.array(occupancy_map_start)) * 0.0254
                #     offset[2] = TARGET_ALTITUDE * 0.0254
                #     waypoints.append(list(offset))

                # # visualize_prob_road_map_crazyflie(occupancy_map, G, occupancy_map_start,
                # #                                 occupancy_map_goal, unique_path,
                # #                                 all_nodes=False)
                # reduced_path = condense_waypoints_crazyflie(occupancy_map, unique_path)
                # print(f'reduced_path: {reduced_path}')
                # visualize_prob_road_map_crazyflie(occupancy_map, G, occupancy_map_start,
                #                                 occupancy_map_goal, reduced_path,
                #                                 all_nodes=False)
                # print(f'All waypoints: {waypoints}',uri)



                # # waypoints = [[y1,x1,z1],[y2,x2,z2],[],[],...]


                # for n in range(len(waypoints)):
                #     x = np.cos(np.deg2rad(Yaw)) * ( waypoints[n+1][0] - waypoints[n][0] ) + np.sin(np.deg2rad(Yaw)) * ( waypoints[n+1][1] - waypoints[n][1] )   # from 절대좌표계 to 회전좌표계
                #     y = -np.sin(np.deg2rad(Yaw)) * ( waypoints[n+1][0] - waypoints[n][0] ) + np.cos(np.deg2rad(Yaw)) * ( waypoints[n+1][1] - waypoints[n][1] )
                #     mc.move_distance(x, y, 0, velocity=0.4)

                #     print(x,y,uri)



                # mc.land()








def reset_estimator(scf):
    cf = scf.cf
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    time.sleep(2)

if __name__ == '__main__':
    # Initialize drivers
    cflib.crtp.init_drivers(enable_debug_driver=False)
    


    # URI1 = 'radio://0/100/2M/E7E7E7E701'
    # URI2 = 'radio://0/80/2M/E7E7E7E7E7'
    # URI3 = 'radio://0/80/2M/E7E7E7E703'   
    URI4 = 'radio://0/70/2M/E7E7E7E704'


    uris = [
            # URI1,
            # URI2,
            # URI3,
            URI4
            ]

    # params1 = {'trace': True, 'trace_x':[], 'trace_y':[], 'U': 'radio://0/100/2M/E7E7E7E701'}
    # params2 = {'trace': True, 'trace_x':[], 'trace_y':[], 'U':'radio://0/80/2M/E7E7E7E7E7'}
    # params3 = {'trace': True, 'trace_x':[], 'trace_y':[], 'U':'radio://0/80/2M/E7E7E7E703'}
    params4 = {'trace': True, 'trace_x':[], 'trace_y':[], 'U':'radio://0/70/2M/E7E7E7E704'}

    params = {
        # URI1: [params1],
        # URI2: [params2],
        # URI3: [params3],
        URI4: [params4]
    }


    factory = CachedCfFactory(rw_cache='./cache')
    with Swarm(uris, factory=factory) as swarm:
        swarm.parallel(reset_estimator)
        swarm.parallel_safe(log_position, args_dict=params)

    # uri = scf.cf.link_uri
    # if uri == "radio://0/100/2M/E7E7E7E701":
    #     print(params)