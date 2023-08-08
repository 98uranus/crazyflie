# -* - coding : utf -8 -* -
#
# || ____ _ __
# + - - - - - -+ / __ )(_) / _______________ _____ ___
# | 0xBC | / __ / / __/ ___ / ___/ __ ‘/_ / / _ \
# + - - - - - -+ / /_/ / / /_/ /__/ / / /_/ / / /_/ __/
# || || / _____ /_/\ __ /\ ___ /_/ \__ ,_/ /___ /\ ___/
#
# Copyright (C) 2017 Bitcraze AB
#
# Crazyflie Nano Quadcopter Client
#
# This program is free software ; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation ; either version 2
# of the License , or (at your option ) any later version .
#
# This program is distributed in the hope that it will be useful ,
# but WITHOUT ANY WARRANTY ; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE . See the
# GNU General Public License for more details .
# You should have received a copy of the GNU General Public License
# along with this program ; if not , write to the Free Software
# Foundation , Inc . , 51 Franklin Street , Fifth Floor , Boston ,
# MA 02110 -1301 , USA .
"""
This script flies one crazyflie autonomously through an obstacle course .
The coordinates are read from a file , placed in a directed graph and the
shortest path through the course is calculated and sent to the drone .
The drone can detect obstacles 50 cm away (def in is_close ) and can handle
obstacles at splitting points and dead ends .
The script is designed for the floe deck .
"""

import logging
import time
import csv
import os
import sys
import networkx as nx
import numpy as np
import math
import cflib.crtp
from cflib.utils import uri_helper
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
# from MotionCommander import MotionCommander
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils.multiranger import Multiranger


URI = 'radio://0/70/2M/E7E7E7E704'

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

def is_close ( range ) :
    MIN_DISTANCE = 0.5 # m
    if range is None :
        return False
    else :
        return range < MIN_DISTANCE

def angles_path ( nodes_shortest ) :
    angles = []
    for i in range (1 , len( nodes_shortest ) -1) :
        p0 = nodes_shortest [ i ]
        p1 = nodes_shortest [i -1]
        p2 = nodes_shortest [ i +1]
        v1 = np.array ([ p1 [0] - p0 [0] , p1 [1] - p0 [1]])
        v2 = np.array ([ p2 [0] - p0 [0] , p2 [1] - p0 [1]])
        angle = np.math.atan2 ( np.linalg.det ([ v1 , v2 ]) , np.dot ( v1 , v2 ) )
        angles.append (180 - np.degrees ( angle ) )
    angles.append (0)
    print ( angles )
    return angles

def angle_nodes ( current , next , last ) :
    v1 = np.array ([ last [0] - current [0] , last [1] - current [1]])
    v2 = np.array ([ next [0] - current [0] , next [1] - current [1]])
    angle = np.degrees ( np.math.atan2 ( np.linalg.det ([ v2 , v1 ]) , np.dot ( v2 , v1 ) ) )
    return angle

def make_graph () :
    G = nx.DiGraph ()
    with open ('copycoordinates.csv', 'r') as file :
        reader = csv.reader ( file )
        line_count = 0
        for row in reader :
            nr_nodes = int( len ( row ) /3)
            nr_paths = nr_nodes -1
            x = []
            y = []
            z = []
            for i in range (0 , nr_nodes ) :
                x.append ( float ( row [0+3* i ]) )
                y.append ( float ( row [1+3* i ]) )
                z.append ( float   ( row [2+3* i ]) )
            for i in range (0 , nr_nodes -1) :
                Weight = abs( x [0] - x [ i +1]+ y [0] - y [ i +1]+ z [0] - z [ i +1])
                G.add_edge (( x [0] , y [0] , z [0]) ,( x [ i +1] , y [ i +1] , z [ i +1]) , weight = Weight )
    return G

def recalc_path ( current_coord , next_coord , end_coord ) : # Pathfinding
    G.remove_edge ( current_coord , next_coord )
    last_coord = next_coord
    nodes_shortest = nx.dijkstra_path (G, source = current_coord , target = end_coord ) # Weight = 1
    next_coord = nodes_shortest [1]
    angles = angles_path ( nodes_shortest )
    angle = angle_nodes ( current_coord , next_coord , last_coord )
    return next_coord , nodes_shortest , angles , angle

def change_startcoord () : 
    with open ('testpath.csv', 'r') as file :
        reader = csv.reader ( file )
        lines = list ( reader )
        lines [0][2] = float ( lines [0][2] ) - 0.0001
    file.close ()
    
    with open ('copycoordinates.csv', 'w') as file : 
        writer = csv.writer ( file )
        writer.writerows ( lines )
    file.close ()
    return lines

def fly (G , lines , mc , multiranger ) :
    nodes_shortest = nx.dijkstra_path (G , source =( float ( lines [0][0]) ,float ( lines [0][1]) ,
        float ( lines [0][2]) ) , target =( float ( lines [0][0]) ,float ( lines [0][1]) ,float ( lines
        [0][2]) +0.0001) )                                                                               # 최단경로 
    angles = angles_path ( nodes_shortest )                                                              # Pathfinding
    current_coord = nodes_shortest [0]                                                                   # p1
    next_coord = nodes_shortest [1]                                                                      # p0
    end_coord = nodes_shortest [ len( nodes_shortest ) -1]
    print ( current_coord )

    coord_index = 1
    traveled_path = []
    latest_angles = []
    traveled_path.append ( current_coord )

    while current_coord != end_coord :
        time.sleep (1)
        angle = angles [ coord_index -1]
        
        # Check if there is a obstacle placed at the splitting point
        if is_close ( multiranger.front ) :
            print ('Obstacle placed close to the splitting point')
            next_coord , nodes_shortest , angles , correction_angle = recalc_path (
                current_coord , next_coord , end_coord )
            
            #If obstacle is found turn to the coordinate next in line 장애물 회피
            if correction_angle > 180: # 다음 노드와의 각도, 시계방향으로 360도
                mc.turn_left (360 - correction_angle , 360.0/2.0)
                print ('turn left')
            else :
                mc.turn_right ( correction_angle , 360.0/2.0)
                print ('turn right')
            coord_index = 1

        #If no obstacle is found , go to next coordinate but stop before and check if it is a dead end
        else :
            distance_x = next_coord [0] - current_coord [0]
            distance_y = next_coord [1] - current_coord [1]
            distance_z = next_coord [2] - current_coord [2]
            x_distance = math.sqrt (( distance_x * distance_x ) +( distance_y * distance_y ) ) - 0.2 
           
            mc.move_distance ( x_distance -0.2 , 0.0 , distance_z )
            time.sleep (1)
            
            # Check to see if it is a dead end if it is move back to the latest splitting point
            if is_close ( multiranger.front ) :
                mc.move_distance ( -( x_distance -0.2) , 0 , -( distance_z ) )
                for i in range ( len( traveled_path ) ) :
                    if len ( G.adj [( traveled_path [ -( i +1) ]) ]) >=2:
                        splitting_point = traveled_path [ -( i +1) ]
                        break
                    else :
                        pass
                
                #If the last splitting point is the last coordinate go back to it
                if splitting_point == current_coord :
                    next_coord , nodes_shortest , angles , correction_angle = recalc_path
                    ( current_coord , next_coord , end_coord )
                    if correction_angle > 180:
                        mc.turn_left (360 - correction_angle , 360.0/2.0)
                    else :
                        mc.turn_right ( correction_angle ,360.0/2.0)
                    
                #If the last splitting point is several coordinates away , retrack to it
                else :
                    print ( current_coord )
                    retrack = nx.dijkstra_path (G , source = splitting_point , target =
                    current_coord )
                    print ( retrack )
                    G.remove_edge ( current_coord , next_coord )
                
                    for i in range ( len ( retrack ) -1) :
                        distance_x = retrack [ -( i +2) ][0] - current_coord [0]
                        distance_y = retrack [ -( i +2) ][1] - current_coord [1]
                        distance_z = retrack [ -( i +2) ][2] - current_coord [2]
                        x_distance = math.sqrt (( distance_x * distance_x ) +( distance_y *
                        distance_y ) )
                        latest_angle = latest_angles [ -( i +1) ]
                        if latest_angle > 180:
                            mc.turn_right (360 - latest_angle , 360.0/2.0)
                        
                        else :
                            mc.turn_left ( latest_angle ,360.0/2.0)
                            mc.move_distance ( x_distance , 0.0 , distance_z )
                            traveled_path.pop ()
                            current_coord = retrack [ -( i +2) ]
                            print ('current_coord')
                            print ( current_coord )
                    next_coord , nodes_shortest , angles , correction_angle = recalc_path
                    ( current_coord , retrack [1] , end_coord )
                    time.sleep (1)
                    if correction_angle > 180:
                        mc.turn_left (360 - correction_angle , 360.0/2.0)
                    else :
                        mc.turn_right ( correction_angle ,360.0/2.0)
                    time.sleep (1)
                    coord_index = 1
                
            #If no dead end is found , move all the way to the node 막다른 골목이 없으면 노드점까지 직진
            else :
                mc.move_distance (0.20 , 0.0 , 0.0)
                time.sleep (1)
            
                if angle > 180:
                    mc.turn_left (360 - angle , 360.0/2.0)
                    latest_angles.append ( angle )
                else :
                    mc.turn_right ( angle ,360.0/2.0)
                    latest_angles.append ( angle )
                
                
                time.sleep (1)
                
                # Update current coordinate to next coordinate in the shortest path
                coord_index += 1
                if coord_index < len( nodes_shortest ) :
                    current_coord = next_coord
                    traveled_path.append (( current_coord ) )
                    print ( current_coord )
                    next_coord = nodes_shortest [ coord_index ]
                else :
                    break   
    print ( next_coord )

# Only output errors from the logging framework
logging.basicConfig ( level = logging.ERROR )


if __name__ == '__main__ ':
    cflib.crtp.init_drivers()
    try:
        os.remove ('position.csv')
    except :
        print ('Already deleted .')
    lines = change_startcoord ()
    G = make_graph ()
    
    cflib.crtp.init_drivers ( enable_debug_driver = False )
    with SyncCrazyflie ( URI , cf = Crazyflie ( rw_cache ='./ cache ') ) as scf :
        time.sleep(1)
        start_position_printing ( scf ) # Log the drone’s path
        with MotionCommander ( scf ) as mc :
            with Multiranger ( scf ) as multiranger :
                keep_flying = True  
                mc.up (0.5)
                fly (G , lines , mc , multiranger )