# -*- coding: utf-8 -*-
#
#     ||          ____  _ __
#  +------+      / __ )(_) /_______________ _____  ___
#  | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
#  +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
#   ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
#
#  Copyright (C) 2017 Bitcraze AB
#
#  Crazyflie Nano Quadcopter Client
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
import logging
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.utils import uri_helper

# URI to the Crazyflie to connect to
uri = uri_helper.uri_from_env(default='radio://0/70/2M/E7E7E7E704')

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)






class Battery:
    bat = 'pm.vbat'

    def __init__(self):
        

        lg_stab = LogConfig(name='Stabilizer', period_in_ms=10)
        lg_stab.add_variable('pm.vbat', 'float')

        self.simple_log_async(scf, lg_stab)

    def simple_log_async(self, scf, logconf):
        cf = scf.cf
        cf.log.add_config(logconf)
        logconf.data_received_cb.add_callback(self.log_stab_callback)
        logconf.start()
        time.sleep(5)
        logconf.stop()

    def log_stab_callback(self, timestamp, data, logconf):
        # print('[%d][%s]: %s' % (timestamp, logconf.name, data))
        self._bat = data[self.bat]
        print(self._bat)


    # @property
    # def bat(self):
    #     return self._bat





if __name__ == '__main__':
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    


    with SyncCrazyflie(uri, cf=Crazyflie(rw_cache='./cache')) as scf:

        with Battery() as Bat:
            Bat()
            pass