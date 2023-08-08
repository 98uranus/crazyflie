# -*- coding: utf-8 -*-
"""
example on how to plot decoded sensor data from crazyflie
@author: jsschell
"""
import cfusdlog
import matplotlib.pyplot as plt
import re
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("log00")
args = parser.parse_args()

# decode binary log data
logData = cfusdlog.decode(args.log00)

#only focus on regular logging
logData = logData['fixedFrequency']

# set window background to white
plt.rcParams['figure.facecolor'] = 'w'
    
# number of columns and rows for suplot
plotCols = 1
plotRows = 1

# let's see which keys exists in current data set
keys = ""
for k, v in logData.items():
    keys += k

# get plot config from user
plotGyro = 0
if re.search('gyro', keys):
    inStr = input("plot gyro data? ([Y]es / [n]o): ")
    if ((re.search('^[Yy]', inStr)) or (inStr == '')):
        plotGyro = 1
        plotRows += 1

plotAccel = 0
if re.search('acc', keys):
    inStr = input("plot accel data? ([Y]es / [n]o): ")
    if ((re.search('^[Yy]', inStr)) or (inStr == '')):
        plotAccel = 1
        plotRows += 1
 
# current plot for simple subplot usage
plotCurrent = 00

# new figure
plt.figure(0)

if plotGyro:
    plotCurrent += 1
    plt.subplot(plotRows, plotCols, plotCurrent)
    plt.plot(logData['timestamp'], logData['gyro.x'], '-', label='X')
    plt.plot(logData['timestamp'], logData['gyro.y'], '-', label='Y')
    plt.plot(logData['timestamp'], logData['gyro.z'], '-', label='Z')
    plt.xlabel('timestamp [ms]')
    plt.ylabel('Gyroscope [°/s]')
    plt.legend(loc=9, ncol=3, borderaxespad=0.)
 
if plotAccel:
    plotCurrent += 1
    plt.subplot(plotRows, plotCols, plotCurrent)
    plt.plot(logData['timestamp'], logData['acc.x'], '-', label='X')
    plt.plot(logData['timestamp'], logData['acc.y'], '-', label='Y')
    plt.plot(logData['timestamp'], logData['acc.z'], '-', label='Z')
    plt.xlabel('timestamp [ms]')
    plt.ylabel('Accelerometer [g]')
    plt.legend(loc=9, ncol=3, borderaxespad=0.)

plt.show()