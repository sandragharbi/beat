import os
from beat import heart
from beat.models import GeometryOptimizer

from pyrocko.guts import load
from pyrocko import model

import logging

logger = logging.getLogger('BEAT')

name = 'Aqaba'
year = 1995

project_dir = '/data3TB/' + name + str(year)
store_superdir = '/data3TB/Teleseism/Greensfunctions/Aqaba1995GFS/'
seismic_datadir = '/data3TB/Teleseism/autokiwi/events/Aqaba1995/kiwi/data/'

geodetic_datadir = '/data/SAR_data/Aqaba1995/subsampled/'
tracks = ['A_T114do', 'A_T114up', 'A_T343co',
          'A_T343up', 'D_T254co', 'D_T350co']

n_variations = 20
sample_rate = 1.0
channels = ['Z', 'T']


def init():
    config = heart.init_nonlin(name, year,
        project_dir=project_dir,
        store_superdir=store_superdir,
        sample_rate=sample_rate,
        n_variations=n_variations,
        channels=channels,
        geodetic_datadir=geodetic_datadir,
        seismic_datadir=seismic_datadir,
        tracks=tracks)
    return config


def build_geo_gfs():

    config_fn = os.path.join(project_dir, 'config.yaml')
    config = load(filename=config_fn)

    n_mods = len(config.crust_inds)

    logger.info('Building geodetic Greens Functions for %i varied crustal'
                'velocity models! \n'
                'Building stores in: %s \n' % (n_mods, config.store_superdir))

    eventname = os.path.join(config.seismic_datadir, 'event.txt')
    event = model.load_one_event(eventname)

    for crust_ind in config.crust_inds:
        heart.geo_construct_gf(event, store_superdir,
             source_distance_min=0., source_distance_max=100.,
             source_depth_min=0., source_depth_max=50.,
             source_spacing=0.5, earth_model='ak135-f-average.m',
             crust_ind=crust_ind, execute=True)
        logger.info('Done building model %i / %i \n' % (crust_ind + 1, n_mods))


def check_model_setup():
    config_fn = os.path.join(project_dir, 'config')
    config = load(filename=config_fn)

    problem = GeometryOptimizer(config)
    problem.built_model()
    test_logp = problem.model.logp.eval()
    print('The test probability is %f') % test_logp
    return problem

if __name__ == '__main__':
    config = init()
    build_geo_gfs()
#    check_model_setup()
