import cflib.crtp
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.swarm import Swarm
from cflib.crazyflie.swarm import CachedCfFactory
from cflib.crazyflie.syncLogger import SyncLogger


def log_position(scf):
    lg_stab = LogConfig(name='position', period_in_ms=100)
    lg_stab.add_variable('stateEstimate.x', 'float')
    lg_stab.add_variable('stateEstimate.y', 'float')
    lg_stab.add_variable('stateEstimate.z', 'float')

    uri = scf.cf.link_uri
    with SyncLogger(scf, lg_stab) as logger:
        for log_entry in logger:
            data = log_entry[1]

            x = data['stateEstimate.x']
            y = data['stateEstimate.y']
            z = data['stateEstimate.z']

            print(uri, "is at", x, y, z)


if __name__ == '__main__':
    cflib.crtp.init_drivers(enable_debug_driver=False)

    uris = [
        'radio://0/70/2M/E7E7E7E704'
        'radio://0/90/2M/E7E7E7E702'
    ]

    factory = CachedCfFactory(rw_cache='./cache')
    with Swarm(uris, factory=factory) as swarm:
        swarm.parallel_safe(log_position)