#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#     ||          ____  _ __
#  +------+      / __ )(_) /_______________ _____  ___
#  | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#   ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
#  Copyright (C) 2018 Bitcraze AB
#
#  Crazyflie Python Library
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA  02110-1301, USA.
"""
Example script that plots the output ranges from the Multiranger and Flow
deck in a 3D plot.
When the application is started the Crazyflie will hover at 0.3 m. The
Crazyflie can then be controlled by using keyboard input:
 * Move by using the arrow keys (left/right/forward/backwards)
 * Adjust the right with w/s (0.1 m for each keypress)
 * Yaw slowly using a/d (CCW/CW)
 * Yaw fast using z/x (CCW/CW)
There's additional setting for (see constants below):
 * Plotting the downwards sensor
 * Plotting the estimated Crazyflie position
 * Max threshold for sensors
 * Speed factor that set's how fast the Crazyflie moves
The demo is ended by either closing the graph window.
For the example to run the following hardware is needed:
 * Crazyflie 2.0
 * Crazyradio PA
 * Flow deck
 * Multiranger deck
"""
import logging
import math
import sys
import time

import numpy as np
from vispy import scene
from vispy.scene import visuals
from vispy.scene.cameras import TurntableCamera

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper
from PyQt5 import QtCore, QtWidgets
# from sip import setapi

from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.utils.multiranger import Multiranger

import matplotlib.pyplot as plt
import matplotlib.animation as animation

try:
    from sip import setapi
    setapi('QVariant', 2)
    setapi('QString', 2)
except ImportError:
    pass

# from PyQt5 import QtCore, QtWidgets

logging.basicConfig(level=logging.INFO)

URI = uri_helper.uri_from_env(default='radio://0/70/2M/E7E7E7E704')

if len(sys.argv) > 1:
    URI = sys.argv[1]

logging.basicConfig(level=logging.ERROR)

# Grid map 전역변수
EXTEND_AREA = 1.0
Sensor_Maxrange = 2.0

# Enable plotting of Crazyflie
PLOT_CF = False
# Enable plotting of down sensor
PLOT_SENSOR_DOWN = False
# Set the sensor threshold (in mm)
SENSOR_TH = 2000
# Set the speed factor for moving and rotating
SPEED_FACTOR = 0.3


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, URI):
        QtWidgets.QMainWindow.__init__(self)

        self.resize(700, 500)
        self.setWindowTitle('Multi-ranger point cloud')

        self.canvas = Canvas(self.updateHover)
        self.canvas.create_native()
        self.canvas.native.setParent(self)

        self.setCentralWidget(self.canvas.native)

        cflib.crtp.init_drivers()
        self.cf = Crazyflie(ro_cache=None, rw_cache='cache')

        # Connect callbacks from the Crazyflie API
        self.cf.connected.add_callback(self.connected)
        self.cf.disconnected.add_callback(self.disconnected)

        # Connect to the Crazyflie
        self.cf.open_link(URI)

        self.hover = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'yaw': 0.0, 'height': 0.3}
        self.hoverTimer = QtCore.QTimer()
        self.hoverTimer.timeout.connect(self.sendHoverCommand)
        self.hoverTimer.setInterval(100)
        self.hoverTimer.start()
        

    def sendHoverCommand(self):
        self.cf.commander.send_hover_setpoint(
            self.hover['x'], self.hover['y'], self.hover['yaw'],
            self.hover['height'])

    def updateHover(self, k, v):
        if (k != 'height'):
            self.hover[k] = v * SPEED_FACTOR
        else:
            self.hover[k] += v

    def disconnected(self, URI):
        print('Disconnected')

    def connected(self, URI):
        print('We are now connected to {}'.format(URI))

        # The definition of the logconfig can be made before connecting
        lpos = LogConfig(name='Position', period_in_ms=100)
        lpos.add_variable('stateEstimate.x')
        lpos.add_variable('stateEstimate.y')
        lpos.add_variable('stateEstimate.z')

        try:
            self.cf.log.add_config(lpos)
            lpos.data_received_cb.add_callback(self.pos_data)
            lpos.start()
        except KeyError as e:
            print('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Position log config, bad configuration.')

        lmeas = LogConfig(name='Meas', period_in_ms=100)
        lmeas.add_variable('range.front')
        lmeas.add_variable('range.back')
        lmeas.add_variable('range.up')
        lmeas.add_variable('range.left')
        lmeas.add_variable('range.right')
        lmeas.add_variable('range.zrange')
        lmeas.add_variable('stabilizer.roll')
        lmeas.add_variable('stabilizer.pitch')
        lmeas.add_variable('stabilizer.yaw')

        try:
            self.cf.log.add_config(lmeas)
            lmeas.data_received_cb.add_callback(self.meas_data)
            lmeas.start()
        except KeyError as e:
            print('Could not start log configuration,'
                  '{} not found in TOC'.format(str(e)))
        except AttributeError:
            print('Could not add Measurement log config, bad configuration.')

    def pos_data(self, timestamp, data, logconf):
        position = [
            data['stateEstimate.x'],
            data['stateEstimate.y'],
            data['stateEstimate.z']
        ]
        self.canvas.set_position(position)

    def meas_data(self, timestamp, data, logconf):
        measurement = {
            'roll': data['stabilizer.roll'],
            'pitch': data['stabilizer.pitch'],
            'yaw': data['stabilizer.yaw'],
            'front': data['range.front'],
            'back': data['range.back'],
            'up': data['range.up'],
            'down': data['range.zrange'],
            'left': data['range.left'],
            'right': data['range.right']
        }
        self.canvas.set_measurement(measurement)

    def closeEvent(self, event):
        if (self.cf is not None):
            self.cf.close_link()


class Canvas(scene.SceneCanvas):
    def __init__(self, keyupdateCB):
        scene.SceneCanvas.__init__(self, keys=None)
        self.size = 1000, 1000
        self.unfreeze()
        self.view = self.central_widget.add_view()
        self.view.bgcolor = '#ffffff'
        self.view.camera = TurntableCamera(
            fov = 10.0, distance = 30.0, up = '+z', center = (0.0, 0.0, 0.0))
        self.last_pos = [0, 0, 0]
        self.pos_markers = visuals.Markers()
        self.meas_markers = visuals.Markers()
        self.pos_data = np.array([0, 0, 0], ndmin=2)
        self.meas_data = np.array([0, 0, 0], ndmin=2)
        self.lines = []

        self.view.add(self.pos_markers)
        self.view.add(self.meas_markers)
        for i in range(6):
            line = visuals.Line()
            self.lines.append(line)
            self.view.add(line)

        self.keyCB = keyupdateCB

        self.freeze()

        scene.visuals.XYZAxis(parent=self.view.scene)

    def on_key_press(self, event):
        if (not event.native.isAutoRepeat()):
            if (event.native.key() == QtCore.Qt.Key_Left):
                self.keyCB('y', 1)
            if (event.native.key() == QtCore.Qt.Key_Right):
                self.keyCB('y', -1)
            if (event.native.key() == QtCore.Qt.Key_Up):
                self.keyCB('x', 1)
            if (event.native.key() == QtCore.Qt.Key_Down):
                self.keyCB('x', -1)
            if (event.native.key() == QtCore.Qt.Key_A):
                self.keyCB('yaw', -70)
            if (event.native.key() == QtCore.Qt.Key_D):
                self.keyCB('yaw', 70)
            if (event.native.key() == QtCore.Qt.Key_Z):
                self.keyCB('yaw', -200)
            if (event.native.key() == QtCore.Qt.Key_X):
                self.keyCB('yaw', 200)
            if (event.native.key() == QtCore.Qt.Key_W):
                self.keyCB('height', 0.1)
            if (event.native.key() == QtCore.Qt.Key_S):
                self.keyCB('height', -0.1)

    def on_key_release(self, event):
        if (not event.native.isAutoRepeat()):
            if (event.native.key() == QtCore.Qt.Key_Left):
                self.keyCB('y', 0)
            if (event.native.key() == QtCore.Qt.Key_Right):
                self.keyCB('y', 0)
            if (event.native.key() == QtCore.Qt.Key_Up):
                self.keyCB('x', 0)
            if (event.native.key() == QtCore.Qt.Key_Down):
                self.keyCB('x', 0)
            if (event.native.key() == QtCore.Qt.Key_A):
                self.keyCB('yaw', 0)
            if (event.native.key() == QtCore.Qt.Key_D):
                self.keyCB('yaw', 0)
            if (event.native.key() == QtCore.Qt.Key_W):
                self.keyCB('height', 0)
            if (event.native.key() == QtCore.Qt.Key_S):
                self.keyCB('height', 0)
            if (event.native.key() == QtCore.Qt.Key_Z):
                self.keyCB('yaw', 0)
            if (event.native.key() == QtCore.Qt.Key_X):
                self.keyCB('yaw', 0)

    def set_position(self, pos):
        self.last_pos = pos
        if (PLOT_CF):
            self.pos_data = np.append(self.pos_data, [pos], axis=0)
            self.pos_markers.set_data(self.pos_data, face_color='red', size=5)

    def rot(self, roll, pitch, yaw, origin, point):
        cosr = math.cos(math.radians(roll))
        cosp = math.cos(math.radians(pitch))
        cosy = math.cos(math.radians(yaw))

        sinr = math.sin(math.radians(roll))
        sinp = math.sin(math.radians(pitch))
        siny = math.sin(math.radians(yaw))

        roty = np.array([[cosy, -siny, 0],
                         [siny, cosy, 0],
                         [0, 0,    1]])

        rotp = np.array([[cosp, 0, sinp],
                         [0, 1, 0],
                         [-sinp, 0, cosp]])

        rotr = np.array([[1, 0,   0],
                         [0, cosr, -sinr],
                         [0, sinr,  cosr]])

        rotFirst = np.dot(rotr, rotp)

        rot = np.array(np.dot(rotFirst, roty))

        tmp = np.subtract(point, origin)
        tmp2 = np.dot(rot, tmp)
        return np.add(tmp2, origin)

    def rotate_and_create_points(self, m):
        data = []
        o = self.last_pos
        roll = m['roll']
        pitch = -m['pitch']
        yaw = m['yaw']

        if (m['up'] < SENSOR_TH):
            up = [o[0], o[1], o[2] + m['up'] / 1000.0]
            data.append(self.rot(roll, pitch, yaw, o, up))

        if (m['down'] < SENSOR_TH and PLOT_SENSOR_DOWN):
            down = [o[0], o[1], o[2] - m['down'] / 1000.0]
            data.append(self.rot(roll, pitch, yaw, o, down))

        if (m['left'] < SENSOR_TH):
            left = [o[0], o[1] + m['left'] / 1000.0, o[2]]
            data.append(self.rot(roll, pitch, yaw, o, left))

        if (m['right'] < SENSOR_TH):
            right = [o[0], o[1] - m['right'] / 1000.0, o[2]]
            data.append(self.rot(roll, pitch, yaw, o, right))

        if (m['front'] < SENSOR_TH):
            front = [o[0] + m['front'] / 1000.0, o[1], o[2]]
            data.append(self.rot(roll, pitch, yaw, o, front))

        if (m['back'] < SENSOR_TH):
            back = [o[0] - m['back'] / 1000.0, o[1], o[2]]
            data.append(self.rot(roll, pitch, yaw, o, back))

        return data

    def set_measurement(self, measurements):
        data = self.rotate_and_create_points(measurements)
        o = self.last_pos
        for i in range(6):
            
            if (i < len(data)):
                o = self.last_pos
                self.lines[i].set_data(np.array([o, data[i]]))
            else:
                self.lines[i].set_data(np.array([o, o]))

            if (len(data) > 0):
                self.meas_data = np.append(self.meas_data, data, axis=0)
                self.meas_markers.set_data(self.meas_data, face_color='blue', size=5)


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
    min_x = round(-12 - Sensor_Maxrange )
    min_y = round(-12 - Sensor_Maxrange )
    max_x = round(12 + Sensor_Maxrange )
    max_y = round(12 + Sensor_Maxrange )
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
        if occupancy_map[laser_beams[i,0], laser_beams[i,1]] != -0.2: # 레이저 빔 사이에 자취가 없으면
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

    return occupancy_map
   

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
        with State(scf) as Ste:
            with MotionCommander(scf) as mc: # 이 라인을 활성화할 경우 갑자기 이륙할 수 있으므로 주의
                with Multiranger(scf) as multiranger:
                    keep_flying = True

                    while keep_flying:
                        front = multiranger.front
                        left = multiranger.left
                        back = multiranger.back
                        right = multiranger.right
                        up = multiranger.up
                        
                        Xpos = Ste.Xpos
                        Ypos = Ste.Ypos
                        Yaw = Ste.Yawpos
                        
                        print('Xpos = {}'.format(Xpos))
                        print('Ypos = {}'.format(Ypos))
                        print('Yaw = {}'.format(Yaw))
                        print('up = {}'.format(up))

                        # appQt = QtWidgets.QApplication(sys.argv)
                        # win = MainWindow(URI)
                        # win.show()
                        # appQt.exec_()

                        if Xpos != None and Ypos != None\
                                and Yaw != None:
                                poses = np.array([Xpos, Ypos, Yaw])
                                hpose = np.vstack((hpose, poses))
                            
                        if front != None and left != None\
                            and back != None and right != None:
                            measures = np.array([front, left, back, right])

                            # Grid map
                            occupancy_map, trace = \
                                    generate_ray_casting_grid_map(poses, measures, xy_resolution, occupancy_map)
                            im = ax1.imshow(occupancy_map.T, origin='lower', animated=True, cmap="PuOr")  # Occupancy map을 이미지 파일로 변환
                        else:
                            occupancy_map = np.full((600,600),0.5)
                            im = ax1.imshow(occupancy_map.T, origin='lower', animated=True, cmap="PuOr")
                        
                        ims.append([im])

                        if Xpos > 0.5 :
                            break
                    
                        appQt = QtWidgets.QApplication(sys.argv)
                        win = MainWindow(URI)
                        win.show()
                        appQt.exec_()
                        

                    mc.land()  
                    ani = animation.ArtistAnimation(fig1, ims, interval=50, repeat_delay=2000)
                    ani.save('181818181818.gif', fps=10000, dpi=80)   # .gif 로 저장
                    plt.show()