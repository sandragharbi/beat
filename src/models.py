import os
import time
import copy
import shutil

import pymc3 as pm
from pymc3 import Metropolis

from pyrocko import gf, util, trace
from pyrocko.guts import Object

import numpy as num

import theano.tensor as tt
from theano import config as tconfig
from theano import shared
from theano.printing import Print

from beat import theanof, heart, utility, smc, backend, metropolis
from beat import covariance as cov
from beat import config as bconfig
from beat.interseismic import geo_backslip_synthetics, seperate_point

import logging


km = 1000.


logger = logging.getLogger('models')

__all__ = ['GeometryOptimizer', 'DistributionOptimizer',
           'sample', 'load_model']


def multivariate_normal(datasets, weights, hyperparams, residuals):
    """
    Calculate posterior Likelihood of a Multivariate Normal distribution.
    Uses plain inverse of the covariances.
    DEPRECATED! Is currently not being used in beat.
    Can only be executed in a `with model context`.

    Parameters
    ----------
    datasets : list
        of :class:`heart.SeismicDataset` or :class:`heart.GeodeticDataset`
    weights : list
        of :class:`theano.shared`
        Square matrix of the inverse of the covariance matrix as weights
    hyperparams : dict
        of :class:`theano.`
    residual : list or array of model residuals

    Returns
    -------
    array_like
    """
    n_t = len(datasets)

    logpts = tt.zeros((n_t), tconfig.floatX)

    for l, data in enumerate(datasets):
        M = tt.cast(shared(data.samples, borrow=True), 'int16')
        hp_name = '_'.join(('h', data.typ))

        logpts = tt.set_subtensor(logpts[l:l + 1],
            (-0.5) * (data.covariance.slnf + \
            (M * 2 * hyperparams[hp_name]) + \
            (1 / tt.exp(hyperparams[hp_name] * 2)) * \
            (residuals[l].dot(weights[l]).dot(residuals[l].T))
                     )
                                 )

    return logpts


def multivariate_normal_chol(
        datasets, weights, hyperparams, residuals, hp_specific=False):
    """
    Calculate posterior Likelihood of a Multivariate Normal distribution.
    Assumes weights to be the inverse cholesky decomposed lower triangle
    of the Covariance matrix.
    Can only be executed in a `with model context`.

    Parameters
    ----------
    datasets : list
        of :class:`heart.SeismicDataset` or :class:`heart.GeodeticDataset`
    weights : list
        of :class:`theano.shared`
        Square matrix of the inverse of the lower triangular matrix of a
        cholesky decomposed covariance matrix
    hyperparams : dict
        of :class:`theano.`
    residual : list or array of model residuals
    hp_specific : boolean
        if true, the hyperparameters have to be arrays size equal to
        the number of datasets, if false size: 1.

    Returns
    -------
    array_like
    """
    n_t = len(datasets)
    logpts = tt.zeros((n_t), tconfig.floatX)
    count = utility.Counter()

    for l, data in enumerate(datasets):
        M = tt.cast(shared(data.samples, borrow=True), 'int16')
        hp_name = '_'.join(('h', data.typ))

        if hp_specific:
            hp = hyperparams[hp_name][count(hp_name)]
        else:
            hp = hyperparams[hp_name]

        tmp = weights[l].dot(residuals[l])

        logpts = tt.set_subtensor(logpts[l:l + 1],
            (-0.5) * (data.covariance.slnf + \
            (M * 2 * hp) + \
            (1 / tt.exp(hp * 2)) * \
            (tt.dot(tmp, tmp))
                     )
                                 )

    return logpts


def hyper_normal(datasets, hyperparams, llks, hp_specific=False):
    """
    Calculate posterior Likelihood only dependent on hyperparameters.

    Parameters
    ----------
    datasets : list
        of :class:`heart.SeismicDatset` or :class:`heart.GeodeticDataset`
    hyperparams : dict
        of :class:`theano.`
    llks : posterior likelihoods
    hp_specific : boolean
        if true, the hyperparameters have to be arrays size equal to
        the number of datasets, if false size: 1.

    Returns
    -------
    array_like
    """
    n_t = len(datasets)
    logpts = tt.zeros((n_t), tconfig.floatX)
    count = utility.Counter()

    for k, data in enumerate(datasets):
        M = data.samples
        hp_name = '_'.join(('h', data.typ))

        if hp_specific:
            hp = hyperparams[hp_name][count(hp_name)]
        else:
            hp = hyperparams[hp_name]

        logpts = tt.set_subtensor(logpts[k:k + 1],
            (-0.5) * (data.covariance.slnf + \
            (M * 2 * hp) + \
            (1 / tt.exp(hp * 2)) * \
                llks[k]
                     )
                                 )

    return logpts


def get_ramp_displacement(slocx, slocy, ramp):
    """
    Get synthetic residual plane in azimuth and range direction of the
    satellite.

    Parameters
    ----------
    slocx : shared array-like :class:`numpy.ndarray`
        local coordinates [km] in east direction
    slocy : shared array-like :class:`numpy.ndarray`
        local coordinates [km] in north direction
    ramp : :class:`theano.tensor.Tensor`
        vector of 2 variables with ramp parameters in azimuth[0] & range[1]
    """
    return slocy * ramp[0] + slocx * ramp[1]


class Composite(Object):
    """
    Class that comprises the rules to formulate the problem. Has to be
    used by an overarching problem object.
    """

    def __init__(self):

        self.input_rvs = {}
        self.fixed_rvs = {}
        self.hierarchicals = None
        self.name = None
        self._like_name = None
        self.config = None

    def get_hyper_formula(self, hyperparams, problem_config):
        """
        Get likelihood formula for the hyper model built. Has to be called
        within a with model context.

        problem_config : :class:`config.ProblemConfig`
        """

        hp_specific = problem_config.dataset_specific_residual_noise_estimation
        logpts = hyper_normal(
            self.datasets, hyperparams, self._llks,
            hp_specific=hp_specific)
        llk = pm.Deterministic(self._like_name, logpts)
        return llk.sum()

    def apply(self, composite):
        """
        Update composite weight matrixes (in place) with weights in given
        composite.

        Parameters
        ----------
        composite : :class:`Composite`
            containing weight matrixes to use for updates
        """

        for i, weight in enumerate(composite.weights):
            A = weight.get_value()
            self.weights[i].set_value(A)


class GeodeticComposite(Composite):
    """
    Comprises data structure of the geodetic composite.

    Parameters
    ----------
    gc : :class:`config.GeodeticConfig`
        configuration object containing seismic setup parameters
    project_dir : str
        directory of the model project, where to find the data
    event : :class:`pyrocko.model.Event`
        contains information of reference event, coordinates of reference
        point and source time
    hypers : boolean
        if true initialise object for hyper parameter optimization
    """

    def __init__(self, gc, project_dir, event, hypers=False):

        super(GeodeticComposite, self).__init__()

        self.event = event

        logger.debug('Setting up geodetic structure ...\n')
        self.name = 'geodetic'
        self._like_name = 'geo_like'

        geodetic_data_path = os.path.join(
            project_dir, bconfig.geodetic_data_name)

        self.datasets = utility.load_objects(geodetic_data_path)

        logger.info('Number of geodetic datasets: %i ' % self.n_t)

        if gc.calc_data_cov:
            logger.warn('Covariance estimation not implemented (yet)!'
                        ' Using imported covariances!')
        else:
            logger.info('No data-covariance estimation! Using imported'
                        ' covariances \n')

        self.weights = []
        for data in self.datasets:
            if int(data.covariance.data.sum()) == data.ncoords:
                logger.warn('Data covariance is identity matrix!'
                            ' Please double check!!!')

            choli = data.covariance.chol_inverse
            self.weights.append(shared(choli, borrow=True))

        if gc.fit_plane:
            logger.info('Fit residual ramp selected!')
            self._slocx = []
            self._slocy = []
            for data in self.datasets:
                if isinstance(data, heart.DiffIFG):
                    locy, locx = data.update_local_coords(self.event)
                    self._slocx.append(shared(
                        locx.astype(tconfig.floatX) / km, borrow=True))
                    self._slocy.append(shared(
                        locy.astype(tconfig.floatX) / km, borrow=True))
                else:
                    logger.debug('Appending placeholder for non-SAR data!')
                    self._slocx.append(None)
                    self._slocy.append(None)

        self.config = gc

        if hypers:
            self._llks = []
            for t in range(self.n_t):
                self._llks.append(shared(num.array([1.]), borrow=True))

    @property
    def n_t(self):
        return len(self.datasets)

    def assemble_results(self, point):
        """
        Assemble geodetic data for given point in solution space.

        Parameters
        ----------
        point : :func:`pymc3.Point`
            Dictionary with model parameters

        Returns
        -------
        List with :class:`heart.GeodeticResult`
        """

        logger.debug('Assembling geodetic data ...')

        processed_synts = self.get_synthetics(point, outmode='stacked_arrays')

        results = []
        for i, data in enumerate(self.datasets):
            res = data.displacement - processed_synts[i]

            results.append(heart.GeodeticResult(
                processed_obs=data.displacement,
                processed_syn=processed_synts[i],
                processed_res=res))

        return results

    def init_hierarchicals(self):

        self.hierarchicals = {}
        if self.config.fit_plane:
            logger.info('Estimating ramp for each dataset...')
            for i, data in enumerate(self.datasets):
                if isinstance(data, heart.DiffIFG):

                    kwargs = dict(
                        name=data.name + '_ramp',
                        shape=(2,),
                        lower=bconfig.default_bounds['ramp'][0],
                        upper=bconfig.default_bounds['ramp'][1],
                        testval=0.,
                        transform=None,
                        dtype=tconfig.floatX)
                    try:
                        self.hierarchicals[data.name] = pm.Uniform(**kwargs)

                    except TypeError:
                        kwargs.pop('name')
                        self.hierarchicals[data.name] = \
                            pm.Uniform.dist(**kwargs)

        logger.info(
            'Initialized %i hierarchical parameters '
            '(ramps).' % len(self.hierarchicals.keys()))

    def remove_ramps(self, residuals):
        """
        Remove an orbital ramp from the residual displacements
        """

        for i, data in enumerate(self.datasets):
            if isinstance(data, heart.DiffIFG):
                residuals[i] -= get_ramp_displacement(
                    self._slocx[i], self._slocy[i],
                    self.hierarchicals[data.name])

        return residuals

    def update_llks(self, point):
        """
        Update posterior likelihoods (in place) of the composite w.r.t.
        one point in the solution space.

        Parameters
        ----------
        point : dict
            with numpy array-like items and variable name keys
        """
        results = self.assemble_results(point)
        for l, result in enumerate(results):
            choli = self.datasets[l].covariance.chol_inverse
            tmp = choli.dot(result.processed_res)
            _llk = num.asarray([num.dot(tmp, tmp)])
            self._llks[l].set_value(_llk)


class GeodeticSourceComposite(GeodeticComposite):
    """
    Comprises how to solve the non-linear geodetic forward model.

    Parameters
    ----------
    gc : :class:`config.GeodeticConfig`
        configuration object containing seismic setup parameters
    project_dir : str
        directory of the model project, where to find the data
    sources : list
        of :class:`pyrocko.gf.seismosizer.Source`
    event : :class:`pyrocko.model.Event`
        contains information of reference event, coordinates of reference
        point and source time
    hypers : boolean
        if true initialise object for hyper parameter optimization
    """

    def __init__(self, gc, project_dir, sources, event, hypers=False):

        super(GeodeticSourceComposite, self).__init__(
            gc, project_dir, event, hypers=hypers)

        self.engine = gf.LocalEngine(
            store_superdirs=[gc.gf_config.store_superdir])

        self.targets = heart.init_geodetic_targets(
            datasets=self.datasets,
            earth_model_name=gc.gf_config.earth_model_name,
            interpolation='multilinear',
            crust_inds=[gc.gf_config.reference_model_idx],
            sample_rate=gc.gf_config.sample_rate)

        self.sources = sources

        _disp_list = [data.displacement for data in self.datasets]
        _odws_list = [data.odw for data in self.datasets]
        _lv_list = [data.update_los_vector() for data in self.datasets]

        # merge geodetic data to calculate residuals on single array
        ordering = utility.ListArrayOrdering(_disp_list, intype='numpy')
        self.Bij = utility.ListToArrayBijection(ordering, _disp_list)

        odws = self.Bij.fmap(_odws_list)

        logger.info(
            'Number of geodetic data points: %i ' % ordering.size)

        self.wdata = shared(self.Bij.fmap(_disp_list) * odws, borrow=True)
        self.lv = shared(self.Bij.f3map(_lv_list), borrow=True)
        self.odws = shared(odws, borrow=True)

    def point2sources(self, point):
        """
        Updates the composite source(s) (in place) with the point values.
        """
        tpoint = copy.deepcopy(point)
        tpoint = utility.adjust_point_units(tpoint)

        # remove hyperparameters from point
        hps = self.config.get_hypernames()

        for hyper in hps:
            if hyper in tpoint:
                tpoint.pop(hyper)

        source_params = self.sources[0].keys()

        for param in tpoint.keys():
            if param not in source_params:
                tpoint.pop(param)

        source_points = utility.split_point(tpoint)

        for i, source in enumerate(self.sources):
            utility.update_source(source, **source_points[i])
            # reset source time may result in store error otherwise
            source.time = 0.

    def get_formula(
            self, input_rvs, fixed_rvs, hyperparams, problem_config):
        """
        Get geodetic likelihood formula for the model built. Has to be called
        within a with model context.
        Part of the pymc3 model.

        Parameters
        ----------
        input_rvs : dict
            of :class:`pymc3.distribution.Distribution`
        fixed_rvs : dict
            of :class:`numpy.array`
        hyperparams : dict
            of :class:`pymc3.distribution.Distribution`
        problem_config : :class:`config.ProblemConfig`

        Returns
        -------
        posterior_llk : :class:`theano.tensor.Tensor`
        """
        hp_specific = problem_config.dataset_specific_residual_noise_estimation

        self.input_rvs = input_rvs
        self.fixed_rvs = fixed_rvs

        logger.info(
            'Geodetic optimization on: \n '
            '%s' % ', '.join(self.input_rvs.keys()))

        self.input_rvs.update(fixed_rvs)

        t0 = time.time()
        disp = self.get_synths(self.input_rvs)
        t1 = time.time()
        logger.debug(
            'Geodetic forward model on test model takes: %f' % \
                (t1 - t0))

        los = (disp[:, 0] * self.lv[:, 0] + \
               disp[:, 1] * self.lv[:, 1] + \
               disp[:, 2] * self.lv[:, 2]) * self.odws

        residuals = self.Bij.srmap(
            tt.cast((self.wdata - los), tconfig.floatX))

        self.init_hierarchicals()
        if self.config.fit_plane:
            residuals = self.remove_ramps(residuals)

        logpts = multivariate_normal_chol(
            self.datasets, self.weights, hyperparams, residuals,
            hp_specific=hp_specific)

        llk = pm.Deterministic(self._like_name, logpts)
        return llk.sum()


class GeodeticGeometryComposite(GeodeticSourceComposite):

    def __init__(self, gc, project_dir, sources, event, hypers=False):

        super(GeodeticGeometryComposite, self).__init__(
            gc, project_dir, sources, event, hypers=hypers)

        # synthetics generation
        logger.debug('Initialising synthetics functions ... \n')
        self.get_synths = theanof.GeoSynthesizer(
            engine=self.engine,
            sources=self.sources,
            targets=self.targets)

    def get_synthetics(self, point, **kwargs):
        """
        Get synthetics for given point in solution space.

        Parameters
        ----------
        point : :func:`pymc3.Point`
            Dictionary with model parameters
        kwargs especially to change output of the forward model

        Returns
        -------
        list with :class:`numpy.ndarray` synthetics for each target
        """
        self.point2sources(point)

        displacements = heart.geo_synthetics(
            engine=self.engine,
            targets=self.targets,
            sources=self.sources,
            **kwargs)

        synths = []
        for disp, data in zip(displacements, self.datasets):
            synths.append((
                disp[:, 0] * data.los_vector[:, 0] + \
                disp[:, 1] * data.los_vector[:, 1] + \
                disp[:, 2] * data.los_vector[:, 2]))

        return synths

    def update_weights(self, point, n_jobs=1, plot=False):
        """
        Updates weighting matrixes (in place) with respect to the point in the
        solution space.

        Parameters
        ----------
        point : dict
            with numpy array-like items and variable name keys
        """
        gc = self.config

        self.point2sources(point)

        for i, data in enumerate(self.datasets):
            crust_targets = heart.init_geodetic_targets(
                datasets=[data],
                earth_model_name=gc.gf_config.earth_model_name,
                interpolation='nearest_neighbor',
                crust_inds=range(*gc.gf_config.n_variations),
                sample_rate=gc.gf_config.sample_rate)

            logger.debug('Track %s' % data.name)
            cov_pv = cov.geodetic_cov_velocity_models(
                engine=self.engine,
                sources=self.sources,
                targets=crust_targets,
                dataset=data,
                plot=plot,
                event=self.event,
                n_jobs=1)

            cov_pv = utility.ensure_cov_psd(cov_pv)

            data.covariance.pred_v = cov_pv
            choli = data.covariance.chol_inverse
            self.weights[i].set_value(choli)
            data.covariance.update_slnf()


class GeodeticInterseismicComposite(GeodeticSourceComposite):

    def __init__(self, gc, project_dir, sources, event, hypers=False):

        super(GeodeticInterseismicComposite, self).__init__(
            gc, project_dir, sources, event, hypers=hypers)

        for source in sources:
            if not isinstance(source, gf.RectangularSource):
                raise TypeError('Sources have to be RectangularSources!')

        self._lats = self.Bij.fmap([data.lats for data in self.datasets])
        self._lons = self.Bij.fmap([data.lons for data in self.datasets])

        self.get_synths = theanof.GeoInterseismicSynthesizer(
            lats=self._lats,
            lons=self._lons,
            engine=self.engine,
            targets=self.targets,
            sources=sources,
            reference=event)

    def get_synthetics(self, point, **kwargs):
        """
        Get synthetics for given point in solution space.

        Parameters
        ----------
        point : :func:`pymc3.Point`
            Dictionary with model parameters
        kwargs especially to change output of the forward model

        Returns
        -------
        list with :class:`numpy.ndarray` synthetics for each target
        """
        tpoint = copy.deepcopy(point)
        tpoint.update(self.fixed_rvs)
        spoint, bpoint = seperate_point(tpoint)

        self.point2sources(spoint)

        synths = []
        for target, data in zip(self.targets, self.datasets):
            disp = geo_backslip_synthetics(
                engine=self.engine,
                sources=self.sources,
                targets=[target],
                lons=target.lons,
                lats=target.lats,
                reference=self.event,
                **bpoint)
            synths.append((
                disp[:, 0] * data.los_vector[:, 0] + \
                disp[:, 1] * data.los_vector[:, 1] + \
                disp[:, 2] * data.los_vector[:, 2]))

        return synths

    def update_weights(point, n_jobs=1, plot=False):
        logger.warning('Not implemented yet!')
        raise Exception('Not implemented yet!')


class SeismicComposite(Composite):
    """
    Comprises how to solve the non-linear seismic forward model.

    Parameters
    ----------
    sc : :class:`config.SeismicConfig`
        configuration object containing seismic setup parameters
    event: :class:`pyrocko.model.Event`
    project_dir : str
        directory of the model project, where to find the data
    hypers : boolean
        if true initialise object for hyper parameter optimization
    """
    _datasets = None
    _weights = None
    _targets = None

    def __init__(self, sc, event, project_dir, hypers=False):

        super(SeismicComposite, self).__init__()

        logger.debug('Setting up seismic structure ...\n')
        self.name = 'seismic'
        self._like_name = 'seis_like'

        if sc.station_corrections:
            self.correction_name = 'time_shift'

        self.event = event
        self.engine = gf.LocalEngine(
            store_superdirs=[sc.gf_config.store_superdir])

        seismic_data_path = os.path.join(
            project_dir, bconfig.seismic_data_name)
        stations, data_traces = utility.load_objects(
            seismic_data_path)

        wavenames = sc.get_waveform_names()

        target_deltat = 1. / sc.gf_config.sample_rate

        targets = heart.init_seismic_targets(
            stations,
            earth_model_name=sc.gf_config.earth_model_name,
            channels=sc.get_unique_channels(),
            sample_rate=sc.gf_config.sample_rate,
            crust_inds=[sc.gf_config.reference_model_idx],
            reference_location=sc.gf_config.reference_location,
            blacklist=sc.blacklist)

        self.datahandler = heart.DataWaveformCollection(stations, wavenames)
        self.datahandler.add_datasets(
            data_traces, location=sc.gf_config.reference_model_idx)
        self.datahandler.downsample_datasets(target_deltat)
        self.datahandler.add_targets(targets)
        self.datahandler.station_blacklisting(sc.blacklist)

        self.wavemaps = []
        for wc in sc.waveforms:
            if wc.include:
                wmap = self.datahandler.get_waveform_mapping(
                    wc.name, channels=wc.channels)
                wmap.config = wc

                wmap.station_distance_weeding(event, wc.distances)
                wmap.update_interpolation(wc.interpolation)

                logger.info('Number of seismic datasets for %s: %i ' % (
                    wmap.name, wmap.n_data))

                if sc.calc_data_cov:
                    logger.info(
                        'Estimating seismic data-covariances '
                        'for %s ...\n' % wmap.name)

                    cov_ds_seismic = cov.seismic_data_covariance(
                        data_traces=wmap.datasets,
                        filterer=wc.filterer,
                        sample_rate=sc.gf_config.sample_rate,
                        arrival_taper=wc.arrival_taper,
                        engine=self.engine,
                        event=self.event,
                        targets=wmap.targets)
                else:
                    logger.info('No data-covariance estimation, using imported'
                                ' covariances...\n')

                    cov_ds_seismic = []
                    at = wc.arrival_taper
                    n_samples = int(num.ceil(
                        at.duration * sc.gf_config.sample_rate))

                    for trc in wmap.datasets:
                        if trc.covariance is None:
                            logger.warn(
                                'No data covariance given/estimated! '
                                'Setting default: eye')
                            cov_ds_seismic.append(num.eye(n_samples))
                        else:
                            data_cov = trc.covariance.data
                            if data_cov.shape[0] != n_samples:
                                raise ValueError(
                                    'Imported covariance %i does not agree '
                                    ' with taper duration %i!' % (
                                        data_cov.shape[0], n_samples))
                            cov_ds_seismic.append(data_cov)

                weights = []
                for t, trc in enumerate(wmap.datasets):
                    trc.covariance = heart.Covariance(data=cov_ds_seismic[t])
                    if int(trc.covariance.data.sum()) == trc.data_len():
                        logger.warn('Data covariance is identity matrix!'
                                    ' Please double check!!!')
                    icov = trc.covariance.chol_inverse
                    weights.append(shared(icov, borrow=True))

                wmap.add_weights(weights)

                self.wavemaps.append(wmap)
            else:
                logger.info(
                    'The waveform defined in "%s" config is not '
                    'included in the optimization!' % wc.name)

        if hypers:
            self._llks = []
            for t in range(self.n_t):
                self._llks.append(shared(num.array([1.]), borrow=True))

    def init_hierarchicals(self):
        """
        Initialise random variables for temporal station corrections.
        """
        self.hierarchicals = {}
        if self.config.station_corrections:
            logger.info(
                'Estimating time shift for each station,'
                ' channel and phase ...')
            kwargs = dict(
                name=self.correction_name,
                shape=self.n_t,
                lower=bconfig.default_bounds[self.correction_name][0],
                upper=bconfig.default_bounds[self.correction_name][1],
                testval=0.,
                transform=None,
                dtype=tconfig.floatX)

            try:
                station_corrs_rv = pm.Uniform(**kwargs)

            except TypeError:
                kwargs.pop('name')
                station_corrs_rv = pm.Uniform.dist(**kwargs)

            nhierarchs = station_corrs_rv.shape.prod()
            self.hierarchicals[self.correction_name] = station_corrs_rv
        else:
            nhierarchs = 0

        logger.info(
            'Initialized %i hierarchical parameters for '
            'station corrections.' % nhierarchs)

    def get_unique_stations(self):
        sl = [wmap.stations for wmap in self.wavemaps]
        us = []
        map(us.extend, sl)
        return list(set(us))

    @property
    def n_t(self):
        return sum(wmap.n_t for wmap in self.wavemaps)

    @property
    def datasets(self):
        if self._datasets is None:
            ds = []
            for wmap in self.wavemaps:
                ds.extend(wmap.datasets)

            self._datasets = ds
        return self._datasets

    @property
    def weights(self):
        if self._weights is None:
            ws = []
            for wmap in self.wavemaps:
                ws.extend(wmap.weights)

            self._weights = ws
        return self._weights

    @property
    def targets(self):
        if self._targets is None:
            ts = []
            for wmap in self.wavemaps:
                ts.extend(wmap.targets)

            self._targets = ts
        return self._targets

    def assemble_results(self, point):
        """
        Assemble seismic traces for given point in solution space.

        Parameters
        ----------
        point : :func:`pymc3.Point`
            Dictionary with model parameters

        Returns
        -------
        List with :class:`heart.SeismicResult`
        """
        logger.debug('Assembling seismic waveforms ...')

        self.point2sources(point)

        syn_proc_traces, obs_proc_traces = self.get_synthetics(
            point, outmode='stacked_traces')

        syn_filt_traces, obs_filt_traces = self.get_synthetics(
            point, outmode='stacked_traces', taper_tolerance_factor=2.)

        ats = []
        for wmap in self.wavemaps:
            wc = wmap.config
            ats.extend(wmap.n_t * [wc.arrival_taper])

        results = []
        for i, (obs_tr, at) in enumerate(zip(obs_proc_traces, ats)):
            dtrace = obs_tr.copy()
            dtrace.set_ydata(
                (obs_tr.get_ydata() - syn_proc_traces[i].get_ydata()))

            taper = at.get_pyrocko_taper(
                float(obs_tr.tmin + num.abs(at.a)))

            results.append(heart.SeismicResult(
                processed_obs=obs_tr,
                processed_syn=syn_proc_traces[i],
                processed_res=dtrace,
                filtered_obs=obs_filt_traces[i],
                filtered_syn=syn_filt_traces[i],
                taper=taper))

        return results

    def update_llks(self, point):
        """
        Update posterior likelihoods of the composite with respect to one point
        in the solution space.

        Parameters
        ----------
        point : dict
            with numpy array-like items and variable name keys
        """
        results = self.assemble_results(point)
        for k, result in enumerate(results):
            choli = self.datasets[k].covariance.chol_inverse
            tmp = choli.dot(result.processed_res.ydata)
            _llk = num.asarray([num.dot(tmp, tmp)])
            self._llks[k].set_value(_llk)


class SeismicGeometryComposite(SeismicComposite):
    """
    Comprises how to solve the non-linear seismic forward model.

    Parameters
    ----------
    sc : :class:`config.SeismicConfig`
        configuration object containing seismic setup parameters
    project_dir : str
        directory of the model project, where to find the data
    sources : list
        of :class:`pyrocko.gf.seismosizer.Source`
    event : :class:`pyrocko.model.Event`
        contains information of reference event, coordinates of reference
        point and source time
    hypers : boolean
        if true initialise object for hyper parameter optimization
    """

    def __init__(self, sc, project_dir, sources, event, hypers=False):

        super(SeismicGeometryComposite, self).__init__(
            sc, event, project_dir, hypers=hypers)

        self.synthesizers = {}
        self.choppers = {}

        self.sources = sources

        # syntetics generation
        logger.debug('Initialising synthetics functions ... \n')
        for wmap in self.wavemaps:
            wc = wmap.config

            self.synthesizers[wc.name] = theanof.SeisSynthesizer(
                engine=self.engine,
                sources=self.sources,
                targets=wmap.targets,
                event=self.event,
                arrival_taper=wc.arrival_taper,
                wavename=wmap.name,
                filterer=wc.filterer,
                pre_stack_cut=sc.pre_stack_cut)

            self.choppers[wc.name] = theanof.SeisDataChopper(
                sample_rate=sc.gf_config.sample_rate,
                traces=wmap.datasets,
                arrival_taper=wc.arrival_taper,
                filterer=wc.filterer)

        self.config = sc

    def point2sources(self, point):
        """
        Updates the composite source(s) (in place) with the point values.
        """
        tpoint = copy.deepcopy(point)
        tpoint = utility.adjust_point_units(tpoint)

        # remove hyperparameters from point
        hps = self.config.get_hypernames()

        for hyper in hps:
            if hyper in tpoint:
                tpoint.pop(hyper)

        source = self.sources[0]
        source_params = source.keys() + source.stf.keys()

        for param in tpoint.keys():
            if param not in source_params:
                tpoint.pop(param)

        tpoint['time'] += self.event.time

        source_points = utility.split_point(tpoint)

        for i, source in enumerate(self.sources):
            utility.update_source(source, **source_points[i])

    def get_formula(
            self, input_rvs, fixed_rvs, hyperparams, problem_config):
        """
        Get seismic likelihood formula for the model built. Has to be called
        within a with model context.

        Parameters
        ----------
        input_rvs : list
            of :class:`pymc3.distribution.Distribution`
        fixed_rvs : dict
            of :class:`numpy.array`
        hyperparams : dict
            of :class:`pymc3.distribution.Distribution`
        problem_config : :class:`config.ProblemConfig`

        Returns
        -------
        posterior_llk : :class:`theano.tensor.Tensor`
        """
        hp_specific = problem_config.dataset_specific_residual_noise_estimation

        self.input_rvs = input_rvs
        self.fixed_rvs = fixed_rvs

        logger.info(
            'Seismic optimization on: \n '
            ' %s' % ', '.join(self.input_rvs.keys()))

        t2 = time.time()
        wlogpts = []

        self.init_hierarchicals()

        for wmap in self.wavemaps:
            synths, tmins = self.synthesizers[wmap.name](self.input_rvs)

            if self.correction_name in self.hierarchicals.keys():
                tmins += self.hierarchicals[
                    self.correction_name][wmap.station_correction_idxs]

            data_trcs = self.choppers[wmap.name](tmins)
            residuals = data_trcs - synths

            logpts = multivariate_normal_chol(
                wmap.datasets, wmap.weights, hyperparams, residuals,
                hp_specific=hp_specific)

            wlogpts.append(logpts)

        t3 = time.time()
        logger.debug(
            'Teleseismic forward model on test model takes: %f' %
            (t3 - t2))

        llk = pm.Deterministic(self._like_name, tt.concatenate((wlogpts)))
        return llk.sum()

    def get_synthetics(self, point, **kwargs):
        """
        Get synthetics for given point in solution space.

        Parameters
        ----------
        point : :func:`pymc3.Point`
            Dictionary with model parameters
        kwargs especially to change output of seismic forward model
            outmode = 'traces'/ 'array' / 'data'

        Returns
        -------
        default: array of synthetics for all targets
        """
        print point
        self.point2sources(point)

        sc = self.config
        synths = []
        obs = []
        for wmap in self.wavemaps:
            wc = wmap.config

            synthetics, tmins = heart.seis_synthetics(
                engine=self.engine,
                sources=self.sources,
                targets=wmap.targets,
                arrival_taper=wc.arrival_taper,
                wavename=wmap.name,
                filterer=wc.filterer,
                pre_stack_cut=sc.pre_stack_cut,
                **kwargs)
            synths.extend(synthetics)

            if self.config.station_corrections:
                tmins += point[
                    self.correction_name][wmap.station_correction_idxs]

            obs.extend(heart.taper_filter_traces(
                wmap.datasets,
                arrival_taper=wc.arrival_taper,
                filterer=wc.filterer,
                tmins=tmins,
                **kwargs))

        return synths, obs

    def update_weights(self, point, n_jobs=1, plot=False):
        """
        Updates weighting matrixes (in place) with respect to the point in the
        solution space.

        Parameters
        ----------
        point : dict
            with numpy array-like items and variable name keys
        """
        sc = self.config

        self.point2sources(point)

        for wmap in self.wavemaps:
            wc = wmap.config

            for channel in wmap.channels:
                datasets = wmap.get_datasets([channel])
                weights = wmap.get_weights([channel])

                for station, dataset, weight in zip(
                    wmap.stations, datasets, weights):

                    logger.debug('Channel %s of Station %s ' % (
                        channel, station.station))

                    crust_targets = heart.init_seismic_targets(
                        stations=[station],
                        earth_model_name=sc.gf_config.earth_model_name,
                        channels=channel,
                        sample_rate=sc.gf_config.sample_rate,
                        crust_inds=range(*sc.gf_config.n_variations),
                        reference_location=sc.gf_config.reference_location)

                    cov_pv = cov.seismic_cov_velocity_models(
                        engine=self.engine,
                        sources=self.sources,
                        targets=crust_targets,
                        wavename=wmap.name,
                        arrival_taper=wc.arrival_taper,
                        filterer=wc.filterer,
                        plot=plot, n_jobs=n_jobs)
                    cov_pv = utility.ensure_cov_psd(cov_pv)

                    self.engine.close_cashed_stores()

                    dataset.covariance.pred_v = cov_pv

                    t0 = time.time()
                    choli = dataset.covariance.chol_inverse
                    t1 = time.time()
                    logger.debug('Calculate weight time %f' % (t1 - t0))
                    weight.set_value(choli)
                    dataset.covariance.update_slnf()


class GeodeticDistributerComposite(GeodeticComposite):
    """
    Comprises how to solve the geodetic (static) linear forward model.
    Distributed slip
    """

    def __init__(self, gc, project_dir, event, hypers=False):

        super(GeodeticDistributerComposite, self).__init__(
            gc, project_dir, event, hypers=hypers)

        self.gfs = {}
        self.sgfs = {}
        self.gf_names = {}

        self._mode = 'static'
        self.gfpath = os.path.join(project_dir, self._mode,
                         bconfig.linear_gf_dir_name)

        self.shared_data = [
            shared(data.displacement.astype(tconfig.floatX), borrow=True) \
            for data in self.datasets]
        self.shared_odws = [
            shared(data.odw.astype(tconfig.floatX), borrow=True) \
            for data in self.datasets]

    def load_gfs(self, crust_inds=None, make_shared=True):
        """
        Load Greens Function matrixes for each variable to be inverted for.
        Updates gfs and gf_names attributes.

        Parameters
        ----------
        crust_inds : list
            of int to indexes of Green's Functions
        make_shared : bool
            if True transforms gfs to :class:`theano.shared` variables
        """

        if crust_inds is None:
            crust_inds = range(*self.config.gf_config.n_variations)

        for crust_ind in crust_inds:
            gfpath = os.path.join(self.gfpath,
                str(crust_ind) + '_' + bconfig.geodetic_linear_gf_name)

            self.gf_names[crust_ind] = gfpath
            gfs = utility.load_objects(gfpath)[0]

            if make_shared:
                self.sgfs[crust_ind] = {param: [
                    shared(gf.astype(tconfig.floatX), borrow=True) \
                        for gf in gfs[param]] \
                            for param in gfs.keys()}
            else:
                self.gfs[crust_ind] = gfs

    def load_fault_geometry(self):
        """
        Load fault-geometry, i.e. discretized patches.

        Returns
        -------
        :class:`heart.FaultGeometry`
        """
        return utility.load_objects(
            os.path.join(self.gfpath, bconfig.fault_geometry_name))[0]

    def get_formula(self, input_rvs, fixed_rvs, hyperparams):
        """
        Formulation of the distribution problem for the model built. Has to be
        called within a with-model-context.

        Parameters
        ----------
        input_rvs : list
            of :class:`pymc3.distribution.Distribution`
        hyperparams : dict
            of :class:`pymc3.distribution.Distribution`

        Returns
        -------
        llk : :class:`theano.tensor.Tensor`
            log-likelihood for the distributed slip
        """
        self.input_rvs = input_rvs
        self.fixed_rvs = fixed_rvs

        residuals = [None for i in range(self.n_t)]
        for t in range(self.n_t):

            mu = tt.zeros_like(self.data[t], tconfig.floatX)
            for var, rv in input_rvs.iteritems():
                mu += tt.dot(self.sgfs[0][var][t], rv)

            residuals[t] = self.shared_odws[t] * (self.shared_data[t] - mu)

        if self.config.fit_plane:
            residuals = self.remove_ramps(residuals)

        logpts = multivariate_normal_chol(
            self.datasets, self.weights, hyperparams, residuals)

        llk = pm.Deterministic(self._like_name, logpts)

        return llk.sum()

    def get_synthetics(self, point, outmode='data'):
        """
        Get synthetics for given point in solution space.

        Parameters
        ----------
        point : :func:`pymc3.Point`
            Dictionary with model parameters
        kwargs especially to change output of the forward model

        Returns
        -------
        list with :class:`numpy.ndarray` synthetics for each target
        """
        if len(self.gfs.keys()) == 0:
            self.load_gfs(
                crust_inds=[self.config.gf_config.reference_model_idx],
                make_shared=False)

        tpoint = copy.deepcopy(point)

        hps = self.config.get_hypernames()

        for hyper in hps:
            if hyper in tpoint:
                tpoint.pop(hyper)

        gf_params = self.gfs[0].keys()

        for param in tpoint.keys():
            if param not in gf_params:
                tpoint.pop(param)

        synthetics = []
        for i, data in enumerate(self.datasets):

            mu = num.zeros_like(data.displacement)
            for var, rv in tpoint.iteritems():
                mu += num.dot(self.gfs[0][var][i], rv)
                synthetics.append(mu)

        return synthetics


geometry_composite_catalog = {
    'seismic': SeismicGeometryComposite,
    'geodetic': GeodeticGeometryComposite}


distributer_composite_catalog = {
    'geodetic': GeodeticDistributerComposite,
    }

interseismic_composite_catalog = {
    'geodetic': GeodeticInterseismicComposite,
    }


class InconsistentNumberHyperparametersError(Exception):

    context = 'Configuration file has to be updated!' + \
              ' Hyperparameters have to be re-estimated. \n' + \
              ' Please run "beat sample <project_dir> --hypers"'

    def __init__(self, errmess=''):
        self.errmess = errmess

    def __str__(self):
        return '\n%s\n%s' % (self.errmess, self.context)


class Problem(object):
    """
    Overarching class for the optimization problems to be solved.

    Parameters
    ----------
    config : :class:`beat.BEATConfig`
        Configuration object that contains the problem definition.
    """

    def __init__(self, config, hypers=False):
        self.event = None
        self.model = None

        self._like_name = 'like'

        self.fixed_params = {}
        self.composites = {}
        self.hyperparams = {}

        logger.info('Analysing problem ...')
        logger.info('---------------------\n')

        self.config = config

        mode = self.config.problem_config.mode

        outfolder = os.path.join(self.config.project_dir, mode)

        if hypers:
            outfolder = os.path.join(outfolder, 'hypers')

        self.outfolder = outfolder
        util.ensuredir(self.outfolder)

    def init_sampler(self, hypers=False):
        """
        Initialise the Sampling algorithm as defined in the configuration file.
        """

        if hypers:
            sc = self.config.hyper_sampler_config
        else:
            sc = self.config.sampler_config

        if self.model is None:
            raise Exception(
                'Model has to be built before initialising the sampler.')

        with self.model:
            if sc.name == 'Metropolis':
                logger.info(
                    '... Initiate Metropolis ... \n'
                    ' proposal_distribution %s, tune_interval=%i,'
                    ' n_jobs=%i \n' % (
                    sc.parameters.proposal_dist, sc.parameters.tune_interval,
                    sc.parameters.n_jobs))

                t1 = time.time()
                if hypers:
                    step = Metropolis(
                        tune_interval=sc.parameters.tune_interval,
                        proposal_dist=smc.proposal_dists[
                            sc.parameters.proposal_dist])
                else:
                    step = smc.SMC(
                        n_chains=sc.parameters.n_jobs,
                        tune_interval=sc.parameters.tune_interval,
                        likelihood_name=self._like_name,
                        proposal_name=sc.parameters.proposal_dist)
                t2 = time.time()
                logger.info('Compilation time: %f' % (t2 - t1))

            elif sc.name == 'SMC':
                logger.info(
                    '... Initiate Sequential Monte Carlo ... \n'
                    ' n_chains=%i, tune_interval=%i, n_jobs=%i \n' % (
                        sc.parameters.n_chains, sc.parameters.tune_interval,
                        sc.parameters.n_jobs))

                t1 = time.time()
                step = smc.SMC(
                    n_chains=sc.parameters.n_chains,
                    tune_interval=sc.parameters.tune_interval,
                    coef_variation=sc.parameters.coef_variation,
                    proposal_dist=sc.parameters.proposal_dist,
                    likelihood_name=self._like_name)
                t2 = time.time()
                logger.info('Compilation time: %f' % (t2 - t1))

        return step

    def built_model(self):
        """
        Initialise :class:`pymc3.Model` depending on problem composites,
        geodetic and/or seismic data are included. Composites also determine
        the problem to be solved.
        """

        logger.info('... Building model ...\n')

        pc = self.config.problem_config

        with pm.Model() as self.model:

            self.rvs, self.fixed_params = self.get_random_variables()

            self.hyperparams = self.get_hyperparams()

            total_llk = tt.zeros((1), tconfig.floatX)

            for datatype, composite in self.composites.iteritems():
                input_rvs = utility.weed_input_rvs(
                    self.rvs, pc.mode, datatype=datatype)
                fixed_rvs = utility.weed_input_rvs(
                    self.fixed_params, pc.mode, datatype=datatype)

                total_llk += composite.get_formula(
                    input_rvs, fixed_rvs, self.hyperparams, pc)

            # deterministic RV to write out llks to file
            like = pm.Deterministic('tmp', total_llk)

            # will overwrite deterministic name ...
            llk = pm.Potential(self._like_name, like)
            logger.info('Model building was successful!')

    def built_hyper_model(self):
        """
        Initialise :class:`pymc3.Model` depending on configuration file,
        geodetic and/or seismic data are included. Estimates initial parameter
        bounds for hyperparameters.
        """

        logger.info('... Building Hyper model ...\n')

        pc = self.config.problem_config

        point = self.get_random_point(include=['hierarchicals', 'prioŕs'])
        for param in pc.priors.values():
            point[param.name] = param.testvalue

        self.update_llks(point)

        with pm.Model() as self.model:

            self.hyperparams = self.get_hyperparams()

            total_llk = tt.zeros((1), tconfig.floatX)

            for composite in self.composites.itervalues():
                total_llk += composite.get_hyper_formula(self.hyperparams, pc)

            like = pm.Deterministic('tmp', total_llk)
            llk = pm.Potential(self._like_name, like)
            logger.info('Hyper model building was successful!')

    def get_random_point(self, include=['priors', 'hierarchicals', 'hypers']):
        """
        Get random point in solution space.
        """
        pc = self.config.problem_config

        point = {}

        if 'hierarchicals' in include:
            if self.hierarchicals is None:
                self.init_hierarchicals()

            for name, param in self.hierarchicals.items():
                point[name] = param.random()

        if 'priors' in include:
            dummy = {
                param.name: param.random() for param in pc.priors.values()}
            point.update(dummy)

        if 'hypers' in include:
            if len(self.hyperparams) == 0:
                self.hyperparams = self.get_hyperparams()

            hps = {hp_name: param.random()
                   for hp_name, param in self.hyperparams.iteritems()}

            point.update(hps)

        return point

    def get_random_variables(self):
        """
        Evaluate problem setup and return random variables dictionary.
        Has to be executed in a "with model context"!
        """
        pc = self.config.problem_config

        logger.debug('Optimization for %i sources', pc.n_sources)

        rvs = dict()
        fixed_params = dict()
        for param in pc.priors.itervalues():
            if not num.array_equal(param.lower, param.upper):
                rvs[param.name] = pm.Uniform(
                    param.name,
                    shape=param.dimension,
                    lower=param.lower,
                    upper=param.upper,
                    testval=param.testvalue,
                    transform=None,
                    dtype=tconfig.floatX)
            else:
                logger.info(
                    'not solving for %s, got fixed at %s' % (
                        param.name,
                        utility.list_to_str(param.lower.flatten())))
                fixed_params[param.name] = param.lower

        return rvs, fixed_params

    def get_hyperparams(self):
        """
        Evaluate problem setup and return hyperparameter dictionary.
        Has to be executed in a "with model context"!
        """
        pc = self.config.problem_config
        hyperparameters = copy.deepcopy(pc.hyperparameters)

        hyperparams = {}
        n_hyp = 0
        for datatype, composite in self.composites.items():
            hypernames = composite.config.get_hypernames()

            for hp_name in hypernames:
                if hp_name in hyperparameters.keys():
                    hyperpar = hyperparameters.pop(hp_name)

                    if pc.dataset_specific_residual_noise_estimation:
                        ndata = len(composite.get_unique_stations())
                    else:
                        ndata = 1

                else:
                    raise InconsistentNumberHyperparametersError(
                        'Datasets and -types require additional '
                        ' hyperparameter(s): %s!' % hp_name)

                if not num.array_equal(hyperpar.lower, hyperpar.upper):
                    dimension = hyperpar.dimension * ndata

                    kwargs = dict(
                        name=hyperpar.name,
                        shape=dimension,
                        lower=num.repeat(hyperpar.lower, ndata),
                        upper=num.repeat(hyperpar.upper, ndata),
                        testval=num.repeat(hyperpar.testvalue, ndata),
                        dtype=tconfig.floatX,
                        transform=None)

                    try:
                        hyperparams[hp_name] = pm.Uniform(**kwargs)

                    except TypeError:
                        kwargs.pop('name')
                        hyperparams[hp_name] = pm.Uniform.dist(**kwargs)

                    n_hyp += dimension

                else:
                    logger.info(
                        'not solving for %s, got fixed at %s' % (
                            hyperpar.name,
                            utility.list_to_str(hyperpar.lower.flatten())))
                    hyperparams[hyperpar.name] = hyperpar.lower

        if len(hyperparameters) > 0:
            raise InconsistentNumberHyperparametersError(
                'There are hyperparameters in config file, which are not'
                ' covered by datasets/datatypes.')

        logger.info('Optimization for %i hyperparemeters in total!', n_hyp)
        return hyperparams

    def update_llks(self, point):
        """
        Update posterior likelihoods of each composite of the problem with
        respect to one point in the solution space.

        Parameters
        ----------
        point : dict
            with numpy array-like items and variable name keys
        """
        for composite in self.composites.itervalues():
            composite.update_llks(point)

    def apply(self, problem):
        """
        Update composites in problem object with given composites.
        """
        for composite in problem.composites.values():
            self.composites[composite.name].apply(composite)

    def point2sources(self, point):
        """
        Update composite sources(in place) with values from given point.

        Parameters
        ----------
        point : :func:`pymc3.Point`
            Dictionary with model parameters, for which the covariance matrixes
            with respect to velocity model uncertainties are calculated
        """
        for composite in self.composites.values():
            self.composites[composite.name].point2sources(point)

    def update_weights(self, point, n_jobs=1, plot=False):
        """
        Calculate and update model prediction uncertainty covariances of
        composites due to uncertainty in the velocity model with respect to
        one point in the solution space. Shared variables are updated in place.

        Parameters
        ----------
        point : :func:`pymc3.Point`
            Dictionary with model parameters, for which the covariance matrixes
            with respect to velocity model uncertainties are calculated
        n_jobs : int
            Number of processors to use for calculation of seismic covariances
        plot : boolean
            Flag for opening the seismic waveforms in the snuffler
        """
        for composite in self.composites.itervalues():
            composite.update_weights(point, n_jobs=n_jobs)

    def get_synthetics(self, point, **kwargs):
        """
        Get synthetics for given point in solution space.

        Parameters
        ----------
        point : :func:`pymc3.Point`
            Dictionary with model parameters
        kwargs especially to change output of seismic forward model
            outmode = 'traces'/ 'array' / 'data'

        Returns
        -------
        Dictionary with keys according to composites containing the synthetics
        as lists.
        """

        d = dict()

        for composite in self.composites.itervalues():
            d[composite.name] = composite.get_synthetics(point, outmode='data')

        return d

    def init_hierarchicals(self):
        """
        Initialise hierarchical random variables of all composites.
        """
        for composite in self.composites.values():
            composite.init_hierarchicals()
            print 'ch', composite.hierarchicals

    @property
    def hierarchicals(self):
        """
        Return dictionary of all hierarchical variables of the problem.
        """
        d = {}
        for composite in self.composites.values():
            if composite.hierarchicals is not None:
                d.update(composite.hierarchicals)
            else:
                d = None

        return d


class SourceOptimizer(Problem):
    """
    Defines the base-class setup involving non-linear fault geometry.

    Parameters
    ----------
    config : :class:'config.BEATconfig'
        Contains all the information about the model setup and optimization
        boundaries, as well as the sampler parameters.
    """

    def __init__(self, config, hypers=False):

        super(SourceOptimizer, self).__init__(config, hypers)

        pc = config.problem_config

        # Load event
        if config.event is None:
            logger.warn('Found no event information!')
        else:
            self.event = config.event

        # Init sources
        self.sources = []
        for i in range(pc.n_sources):
            if self.event:
                source = \
                    bconfig.source_catalog[pc.source_type].from_pyrocko_event(
                        self.event)

                source.stf = bconfig.stf_catalog[pc.stf_type](
                    duration=self.event.duration)

                # hardcoded inversion for hypocentral time
                if source.stf is not None:
                    source.stf.anchor = -1.
            else:
                source = bconfig.source_catalog[pc.source_type]()

            self.sources.append(source)


class GeometryOptimizer(SourceOptimizer):
    """
    Defines the model setup to solve for the non-linear fault geometry.

    Parameters
    ----------
    config : :class:'config.BEATconfig'
        Contains all the information about the model setup and optimization
        boundaries, as well as the sampler parameters.
    """

    def __init__(self, config, hypers=False):
        logger.info('... Initialising Geometry Optimizer ... \n')

        super(GeometryOptimizer, self).__init__(config, hypers)

        pc = config.problem_config

        dsources = utility.transform_sources(
            self.sources,
            pc.datatypes,
            pc.decimation_factors)

        for datatype in pc.datatypes:
            self.composites[datatype] = geometry_composite_catalog[datatype](
                config[datatype + '_config'],
                config.project_dir,
                dsources[datatype],
                self.event,
                hypers)

        self.config = config

        # updating source objects with values in bounds
        point = self.get_random_point()
        self.point2sources(point)


class InterseismicOptimizer(SourceOptimizer):
    """
    Uses the backslip-model in combination with the blockmodel to formulate an
    interseismic model.

    Parameters
    ----------
    config : :class:'config.BEATconfig'
        Contains all the information about the model setup and optimization
        boundaries, as well as the sampler parameters.
    """

    def __init__(self, config, hypers=False):
        logger.info('... Initialising Interseismic Optimizer ... \n')

        super(InterseismicOptimizer, self).__init__(config, hypers)

        pc = config.problem_config

        if pc.source_type == 'RectangularSource':
            dsources = utility.transform_sources(
                self.sources,
                pc.datatypes)
        else:
            raise TypeError('Interseismic Optimizer has to be used with'
                            ' RectangularSources!')

        for datatype in pc.datatypes:
            self.composites[datatype] = \
                interseismic_composite_catalog[datatype](
                    config[datatype + '_config'],
                    config.project_dir,
                    dsources[datatype],
                    self.event,
                    hypers)

        self.config = config

        # updating source objects with fixed values
        point = self.get_random_point()
        self.point2sources(point)


class DistributionOptimizer(Problem):
    """
    Defines the model setup to solve the linear slip-distribution and
    returns the model object.

    Parameters
    ----------
    config : :class:'config.BEATconfig'
        Contains all the information about the model setup and optimization
        boundaries, as well as the sampler parameters.
    """

    def __init__(self, config, hypers=False):
        logger.info('... Initialising Distribution Optimizer ... \n')

        super(DistributionOptimizer, self).__init__(config, hypers)

        for datatype in config.problem_config.datatypes:
            data_config = config[datatype + '_config']
            composite = distributer_composite_catalog[datatype](
                data_config,
                config.project_dir,
                self.event,
                hypers)

            # do the optimization only on the reference velocity model
            logger.info("Loading %s Green's Functions" % datatype)
            composite.load_gfs(crust_inds=[data_config.gf_config.reference_model_idx])
            self.composites[datatype] = composite

        self.config = config


problem_catalog = {
    bconfig.modes_catalog.keys()[0]: GeometryOptimizer,
    bconfig.modes_catalog.keys()[1]: DistributionOptimizer,
    bconfig.modes_catalog.keys()[3]: InterseismicOptimizer}


def sample(step, problem):
    """
    Sample solution space with the previously initalised algorithm.

    Parameters
    ----------

    step : :class:`SMC` or :class:`pymc3.metropolis.Metropolis`
        from problem.init_sampler()
    problem : :class:`Problem` with characteristics of problem to solve
    """

    sc = problem.config.sampler_config
    pa = sc.parameters

    if pa.update_covariances:
        update = problem
    else:
        update = None

    if sc.name == 'Metropolis':
        logger.info('... Starting Metropolis ...\n')

        name = problem.outfolder
        util.ensuredir(name)

        metropolis.Metropolis_sample(
            n_stages=pa.n_stages,
            n_steps=pa.n_steps,
            stage=pa.stage,
            step=step,
            progressbar=sc.progressbar,
            trace=problem.outfolder,
            burn=pa.burn,
            thin=pa.thin,
            model=problem.model,
            n_jobs=pa.n_jobs,
            update=update,
            rm_flag=pa.rm_flag)

    elif sc.name == 'SMC':
        logger.info('... Starting ATMIP ...\n')

        smc.ATMIP_sample(
            pa.n_steps,
            step=step,
            progressbar=sc.progressbar,
            model=problem.model,
            n_jobs=pa.n_jobs,
            stage=pa.stage,
            update=update,
            homepath=problem.outfolder,
            rm_flag=pa.rm_flag)


def estimate_hypers(step, problem):
    """
    Get initial estimates of the hyperparameters
    """
    logger.info('... Estimating hyperparameters ...')

    pc = problem.config.problem_config
    sc0 = problem.config.sampler_config
    sc = problem.config.hyper_sampler_config
    pa = sc.parameters

    name = problem.outfolder
    util.ensuredir(name)

    mtraces = []
    for stage in range(pa.n_stages):
        logger.info('Metropolis stage %i' % stage)

        point = problem.get_random_point(
            include=['priors', 'hierarchicals'])

        if stage == 0:
            # put testpoint in there
            for param in pc.priors.values():
                point[param.name] = param.testvalue

        problem.outfolder = os.path.join(name, 'stage_%i' % stage)
        start = problem.get_random_point(include=['hypers'])

        if pa.rm_flag:
            shutil.rmtree(problem.outfolder, ignore_errors=True)

        if not os.path.exists(problem.outfolder):
            logger.debug('Sampling ...')
            if sc0.parameters.update_covariances:
                logger.info('Updating Covariances ...')
                problem.update_weights(point)

            problem.update_llks(point)
            with problem.model as hmodel:
                mtraces.append(pm.sample(
                    draws=pa.n_steps,
                    step=step,
                    trace=pm.backends.Text(
                        name=problem.outfolder,
                        model=hmodel),
                    start=start,
                    model=hmodel,
                    chain=stage * pa.n_jobs,
                    njobs=pa.n_jobs,
                    ))

        else:
            logger.debug('Loading existing results!')
            mtraces.append(pm.backends.text.load(
                name=problem.outfolder, model=problem.model))

    mtrace = pm.backends.base.merge_traces(mtraces)
    outname = os.path.join(name, 'stage_final')

    if not os.path.exists(outname):
        util.ensuredir(outname)
        pm.backends.text.dump(name=outname, trace=mtrace)

    n_steps = pa.n_steps

    for v, i in pc.hyperparameters.iteritems():
        d = mtrace.get_values(
            v, combine=True, burn=int(n_steps * pa.burn),
            thin=pa.thin, squeeze=True)
        print d
!        lower = float(num.floor(d.min(axis=0)) - 2.)
!        upper = float(num.ceil(d.max(axis=0)) + 2.)
        logger.info('Updating hyperparameter %s from %f, %f to %f, %f' % (
            v, i.lower, i.upper, lower, upper))
        pc.hyperparameters[v].lower = lower
        pc.hyperparameters[v].upper = upper
        pc.hyperparameters[v].testvalue = (upper + lower) / 2.

    config_file_name = 'config_' + pc.mode + '.yaml'
    conf_out = os.path.join(problem.config.project_dir, config_file_name)

    problem.config.problem_config = pc
    bconfig.dump(problem.config, filename=conf_out)


def load_model(project_dir, mode, hypers=False, nobuild=False):
    """
    Load config from project directory and return BEAT problem including model.

    Parameters
    ----------
    project_dir : string
        path to beat model directory
    mode : string
        problem name to be loaded
    hypers : boolean
        flag to return hyper parameter estimation model instead of main model.
    built : boolean
        flag to do not build models

    Returns
    -------
    problem : :class:`Problem`
    """

    config = bconfig.load_config(project_dir, mode, update=hypers)

    pc = config.problem_config

    if hypers and len(pc.hyperparameters) == 0:
        raise Exception('No hyperparameters specified!'
        ' option --hypers not applicable')

    if pc.mode in problem_catalog.keys():
        problem = problem_catalog[pc.mode](config, hypers)
    else:
        logger.error('Modeling problem %s not supported' % pc.mode)
        raise Exception('Model not supported')

    if not nobuild:
        if hypers:
            problem.built_hyper_model()
        else:
            problem.built_model()

    return problem


class Stage(object):
    """
    Stage, containing sampling results and intermediate sampler
    parameters.
    """
    number = None
    path = None
    step = None
    updates = None
    mtrace = None

    def __init__(self, handler=None, homepath=None, stage_number=-1):

        if handler is not None:
            self.handler = handler
        elif handler is None and homepath is not None:
            self.handler = backend.TextStage(homepath)
        else:
            raise TypeError('Either handler or homepath have to be not None')

        self.number = stage_number

    def load_results(self, model=None, stage_number=None, load='trace'):
        """
        Load stage results from sampling.

        Parameters
        ----------
        model : :class:`pymc3.model.Model`
        stage_number : int
            Number of stage to load
        load : str
            what to load and return 'full', 'trace', 'params'
        """
        if stage_number is None:
            stage_number = self.number

        self.path = self.handler.stage_path(stage_number)

        if not os.path.exists(self.path):
            stage_number = self.handler.highest_sampled_stage()

            logger.info(
                'Stage results %s do not exist! Loading last completed'
                ' stage %s' % (self.path, stage_number))
            self.path = self.handler.stage_path(stage_number)

        self.number = stage_number

        if load == 'full':
            to_load = ['params', 'trace']
        else:
            to_load = [load]

        with model:
            if 'trace' in to_load:
                self.mtrace = self.handler.load_multitrace(
                    stage_number, model=model)

            if 'params' in to_load:
                self.step, self.updates = self.handler.load_sampler_params(
                stage_number)
