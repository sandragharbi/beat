import unittest
from beat import heart, models, config

import numpy as num
from numpy.testing import assert_allclose
from tempfile import mkdtemp
import os
import logging
import shutil

from pyrocko import util, trace
from pyrocko import plot, orthodrome


logger = logging.getLogger('test_seis_composites')
km = 1000.


class RundirectoryError(Exception):
    pass


def get_run_directory():
    cwd = os.getcwd()
    if os.path.basename(cwd) != 'beat':
        raise RundirectoryError(
            'The test suite has to be run in the beat main-directory! '
            'Current work directory: %s' % cwd)
    else:
        return cwd


def load_problem(dirname, mode):
    beat_dir = get_run_directory()
    project_dir = os.path.join(beat_dir, 'data/examples', dirname)

    c = config.load_config(project_dir, mode, update=False)
    c.project_dir = project_dir
    print project_dir

    pc = c.problem_config

    if pc.mode in models.problem_catalog.keys():
        problem = models.problem_catalog[pc.mode](c, False)

    problem.built_model()
    return problem


def _get_mt_source_params():
    source_point = {
        'magnitude': 4.8,
        'mnn': 0.84551376,
        'mee': -0.75868967,
        'mdd': -0.08682409,
        'mne': 0.51322155,
        'mnd': 0.14554675,
        'med': -0.25767963,
        'east_shift': 10.,
        'north_shift': 20.,
        'depth': 8.00,
        'time': -2.5,
        'duration': 5.,
        }
    return {k:num.atleast_1d(num.asarray(v)) for k, v in source_point.iteritems()}


class TestGeoComposite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dirname = 'Mogi'
        mode = 'geometry'
        cls.problem = load_problem(dirname, mode)
        cls.gc = cls.problem.composites['geodetic']

    @classmethod
    def tearDownClass(slc):
        cls.problem = None
        cls.gc = None
        dirname = None

    def test_synths(self):
        logger.info('Test synth')
        synths = self.gc.get_synthetics(
            self.problem.model.test_point, outmode='stacked_arrays')

        for st, ds in zip(synths, self.gc.datasets):
            assert_allclose(st, ds, rtol=1e-03, atol=0)

    def test_results(self):
        logger.info('Test results')
        results = self.gc.assemble_results(self.problem.model.test_point)

        for result in results:
            assert_allclose(result.processed_obs,
                            result.processed_syn, rtol=1e-05, atol=0)

    def test_weights(self):
        logger.info('Test weights')
        for w, d in zip(self.gc.weights, self.gc.datasets):
            assert_allclose(
                w.get_value(), d.covariance.chol_inverse,
                rtol=1e-08, atol=0)


#class TestSeisComposite(unittest.TestCase):

#    @classmethod
#    def setUpClass(cls):
#        dirname = 'FullMT'
#        mode = 'geometry'
#        cls.problem = load_problem(dirname, mode)
#        cls.sc = cls.problem.composites['seismic']

#    def test_synths(self):
#        logger.info('Test synth')
#        synths, obs = self.sc.get_synthetics(
#            self.problem.model.test_point, outmode='data')

#        for st, ot in zip(synths, obs):
#            assert_allclose(st.ydata, ot.ydata, rtol=1e-05, atol=0)

#    def test_results(self):
#        logger.info('Test results')
#        results = self.sc.assemble_results(self.problem.model.test_point)

#        for result in results:
#            assert_allclose(result.processed_obs.ydata,
#                            result.processed_syn.ydata, rtol=1e-05, atol=0)
#            assert_allclose(result.filtered_obs.ydata,
#                            result.filtered_syn.ydata, rtol=1e-05, atol=0)

#    def test_weights(self):
#        logger.info('Test weights')
#        for wmap in self.sc.wavemaps:
#            for w, d in zip(wmap.weights, wmap.datasets):
#                assert_allclose(
#                    w.get_value(), d.covariance.chol_inverse,
#                    rtol=1e-08, atol=0)


if __name__ == "__main__":
    util.setup_logging('test_heart', 'warning')
    unittest.main()
