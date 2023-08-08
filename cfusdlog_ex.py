# -*- coding: utf-8 -*-
"""
example on how to plot decoded sensor data from crazyflie
@author: jsschell
"""
import cfusdlog
import matplotlib.pyplot as plt
import re
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("sd_log00.bin")
args = parser.parse_args()

# decode binary log data
logData = cfusdlog.decode(args.sd_log00.bin)

#only focus on regular logging
logData = logData['fixedFrequency']


logData = pd.DataFrame(logData)
logData.to_csv('D:\대학원\학부논문2022\학부논문2021\박범준\Python')
