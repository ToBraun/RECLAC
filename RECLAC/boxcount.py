#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 11:22:17 2021

@author: tobraun
"""

import numpy as np
import scipy.stats as stats
import scipy.optimize as opt

from .recurrence_plot import RP


class Boxcount(RP):
    
    def __init__(self, x, boxes, glide = False, gridpos = 0, compute_rp = True, **kwargs):
        """
        Class Boxcount for computing recurrence quantification measures from a box-counting analysis.
        Currently supports the computation of the box-counting dimension and recurrence lacunarity
        for recurrence plots. Recurrence plots are computed based on the RP-class. 
        
        Box-counting can be applied through a (computationally faster) static box-counting algorithm
        or a gliding box algorithm (deprecated). For the computation of the box-counting dimension
        and regression parameters for recurrence lacunarity, a basic least-squares or a more robust 
        maximum-likelihood regression of the log-counts against the log-sizes are provided.
        Confidence intervals can be returned based on a bootstrap resampling scheme.
        
        If no recurrence plot should be computed but instead is directly given to the box-counting as an input,
        use 'compute_rp = False'.
        
    Parameters
    ----------
        x: 1D array (float)
            The time series to be analyzed, can be scalar or multi-dimensional.
        boxes : 1D array (int)
            array of boxes sizes (ascending order)
        method: str
             estimation method for the vicinity threshold 
             epsilon (`distance`, `frr`, `stdev`, `fan`)
        thresh: float
            threshold parameter for the vicinity threshold,
            depends on the specified method (`epsilon`, 
            `recurrence rate`, `multiple of standard deviation`,
            `fixed fraction of neighbours`)
        glide: bool, optional
            static or gliding box algorithm
        gridpos: float, optional
            number of 90° rotations to apply to the RP (0,1,2,3)
        compute_rp: bool, optional
            compute an RP based on RP class or feed one as input

    Examples
    --------
    
    - Create an instance of Boxcount for a realization of a random walk. Generate a recurrence plot
      and analyse it with the static box-counting algorithm and a logarithmically increasing set of boxes.
      Symmetry of the RP is used.
        
        >>> np.random.seed(123)
        >>> x = np.cumsum(np.random.normal(0,1,1000))
        >>> wmin, wmax, N = 0, 2, 100
        >>> a_boxes = 2*np.unique(np.logspace(wmin, wmax, N, dtype=int))
        >>> BC = bc.Boxcount(x, method='distance', thresh=2, boxes=a_boxes, glide=False, sym=True)
        >>> BC
        <RECLAC.boxcount.Boxcount at 0x7f047c1330b8>

    - Create an instance of Boxcount with fixed recurrence rate of 10% & without embedding and analyse it 
      (after a 90° rotation) with the static box-counting algorithm.
      Return the box counts.

        >>> BC = bc.Boxcount(x, method='frr', thresh=0.1, boxes=a_boxes, glide=False, gridpos=1)
        >>> _, l_counts = BC.boxcounts()
        >>> l_counts[-1]
            array([[   2,  146, 8799, 5022, 8738],
                   [  32,  144, 5397, 7288, 5022],
                   [   0,   95, 9574, 5397, 8799],
                   [8484, 8780,   95,  144,  146],
                   [9378, 8484,    0,   32,    2]])
        """
        if compute_rp:
            method = kwargs.get('method')
            thresh = kwargs.get('thresh')
            dim = kwargs.get('dim')
            tau = kwargs.get('tau')
            if (method is None) or (thresh is None):
                raise NameError("Please provide a method for thresholding and a threshold.")
            assert (x.ndim == 1), "If no recurrence matrix is provided, please provide a 1-dimensional time series."
            #  Initialize the underlying RecurrencePlot object
            RP.__init__(self, x=x, method=method, thresh=thresh, dim=dim, tau=tau)
            array = self.R.copy()
        else:
            assert (x.ndim == 2), "If a recurrence matrix is provided, it has to be a 2-dimensional array."
            assert ~np.all(np.isnan(x)), "Recurrence matrix only contains NaNs."
            array = x
        
        #  Define attributes
        self.gridpos = gridpos
        self.array = np.rot90(array, gridpos)
        assert boxes.dtype == 'int64', "Please provide an array of integer box sizes for 'boxes'."
        self.boxes = boxes
        #  Obtain boolean values for variables: gliding box? 
        self.glide = glide
        self.N = boxes.size
        # Run the box counting
        l_boxc = []
        for n in range(self.N):
            k = boxes[n]
            if self.glide == True:
                Boxcount._glide_count(self, k, sym=kwargs.get('sym'))
            else:
                Boxcount._stat_count(self, k)
            l_boxc.append(self.S)
        self.counts = l_boxc
        # 'back-up' (private) variable for self.counts to restore value when self.counts is altered
        self._counts = np.copy(np.array(self.counts, dtype=object))
          
        
        
    
    # setter
    def _stat_count(self, k):
        """
        Static box-counting algorithm. Computes the box counts for a fixed grid
        with a given (square) box size 'k'.
        
        .. note::
           Sets a value for the current box count attribute 'S'. No return value!
           
    Parameters
    ----------
        k: int
            box width
        """
        # only evaluate that part of the RP that can be covered without allowing for 
        # 'edge boxes' with different size
        nmod = int(self.array.shape[0]//k*k)
        array = self.array[:nmod, :nmod]
        # box-counting (src: https://github.com/rougier/numpy-100 (#87))
        S = np.add.reduceat(np.add.reduceat(array, np.arange(0, array.shape[0], k), axis=0, dtype=int),
                               np.arange(0, array.shape[1], k), axis=1, dtype=int)
        self.S = S
    
    
    # setter
    def _glide_count(self, k, sym):
        """
        Gliding box-counting algorithm. Computes the box counts for a given (square) box size 'k'
        by gliding over the recurrence plot with a step size of 1. 
        
        .. note::
           Sets a value for the current box count attribute 'S'. No return value!
           Deprecated due to high computational times. A more efficient algorithm will be
           implemented.
        
    Parameters
    ----------
        k: int
            box width
        sym: bool
            use symmetry of the recurrence matrix
        """
        # only evaluate that part of the RP that can be covered without allowing for 
        # 'edge boxes' with different size
        nmod = int(self.array.shape[0]//k*k)
        array = self.array[:nmod, :nmod]
        T = array.shape[0]
        S = np.zeros((T, T), dtype=int)
        # should symmetry be used for saving computational time?
        # (e.g. not reasonable for RPs based on FAN)
        if sym:
            for i in range(T):
                for j in range(i,T):
                    tmp_box = self.array[i:i+k, j:j+k]
                    s = tmp_box.sum()
                    S[i,j], S[j,i] = s, s
        else:
            for i in range(T):
                for j in range(T):
                    tmp_box = self.array[i:i+k, j:j+k]
                    S[i,j] = tmp_box.sum()
        self.S = S
            



    def box_dimension(self, regression, verb=False, **kwargs):
        """
        Returns the box-counting dimension (if 'regression' is not None). 
        Uses the box counts vs box sizes to compute the slope from a log-log-regression.
        
    Parameters
    ----------
        regression: str
            method for loglog-regression: LS (least-squares), ML (maximum likelihood) or None
        verb : bool
            print all regression parameters

    Returns
    ------- 
        : float tuple/dictionary/number (float) depending on 'regression' argument
            one of a) box sizes and box counts, b) box-counting dimension or c) full set of regression parameters

    Examples
    --------
    - Create an instance of Boxcount for a realization of a random walk. Generate a recurrence plot
      and analyse it with the static box-counting algorithm and a logarithmically increasing set of boxes.
      Compute the box-counting dimension with maximum likelihood regression and print regression parameters.
        
        >>> np.random.seed(123)
        >>> x = np.cumsum(np.random.normal(0,1,1000))
        >>> wmin, wmax, N = 0, 2, 100
        >>> a_boxes = 2*np.unique(np.logspace(wmin, wmax, N, dtype=int))
        >>> # run the box-counting algorithm
        >>> BC = bc.Boxcount(x, method='distance', thresh=2, boxes=a_boxes, glide=False)
        >>> # compute the box-counting dimension:
        >>> BC.box_dimension(regression='ML', verb = True)
            {'yinterc': 5.007707585461798, 'slope': -1.5877932599535791, 
            'stderr': 0.04352819734245077, 'R': -0.9962111634654818}
            -1.5877932599535791
    
    - Return the number of boxes that are required to cover the RP for different box widths:

        >>> t_boxc = BC.box_dimension(regression=None, verb=False)
        >>> t_boxc[1]
        array([19450., 11169.,  ...,  23.,    23.])
        """
        # box-counting with increasing box size
        a_covered = np.zeros(self.N)
        for n in range(self.N):
            k, tmp_counts = self.boxes[n], self.counts[n]
            a_covered[n] = int(len(np.where((tmp_counts > 0) & (tmp_counts < k*k))[0]))
    
        # Fit the log(sizes) with log(counts)
        if regression is None:
            outp = (self.boxes, a_covered)
        else:
            outp = Boxcount.loglog_regress(x=self.boxes, y=a_covered, rmethod=regression, verb=verb, 
                                       regr_param=kwargs.get("regr_param"), estimate=kwargs.get("estimate"))
        return outp




    @staticmethod
    def loglog_regress(x, y, rmethod='LS', verb=True, estimate=None, regr_param = False):
        """
        Returns the result of a log-log regression. Uses either a least-squares or a (robust)
        maximum likelihood estimation procedure. The latter can be useful if a slope estimate
        should be obtained even though outliers exist. It requires an initial estimate on the
        regression parameters. 
 
    Parameters
    ----------
        x, y: 1D arrays (float)
            x- and y-coordinates (logarithmic box sizes and box counts)
        rmethod : str
            method for loglog-regression: LS (least-squares), ML (maximum likelihood) or None
        verb: bool
            print all regression parameters
        estimate: list (float)
            initial estimate on regression parameters for ML-regression
        regr_param: bool
            return all regression parameters as dictionary

    Returns
    -------
        d_res: dictionary/number (float)
            one of a) full set of regression parameters or b) slope of the lac.-regression 
        """
        if estimate is None:
            estimate = np.random.normal(0,1,3)
        # use regular least-squares regression (not robust)
        if rmethod == 'LS':
            tmp_res = stats.linregress(np.log10(x), np.log10(y))
            d_res = dict([('yinterc', tmp_res[1]), ('slope', tmp_res[0]), ('stderr', tmp_res[4]), ( 'R', tmp_res[2])])
    
        # use robust maximum-likelihood estimator
        elif rmethod == 'ML':
            a_fit = np.vstack([np.log10(x), np.log10(y)])
            a_fit = np.where(np.isnan(a_fit), 0, np.where(a_fit ==-np.inf, 0, np.where(a_fit==np.inf, 0, a_fit)))
            d_res = (Boxcount.MLEregr(x = a_fit[0,], y = a_fit[1,], estimate = estimate))
#        # results
        if verb: 
            print(d_res)
        if regr_param: 
            return d_res
        else:
            return d_res['slope']




    @staticmethod
    def MLEregr(x, y, estimate):
        """
        Returns the result of a maximum likelihood regression. This can be useful if a slope estimate
        should be obtained even though outliers exist. It requires an initial estimate on the
        regression parameters. 

    Parameters
    ----------
        x, y: 1D arrays (float)
            x- and y-coordinates (logarithmic box sizes and box counts)
        estimate: list (float)
            initial estimate on regression parameters for ML-regression

    Returns
    -------
        d_res: dictionary (float)
            regression parameters: 'y-intercept', 'slope', 'standard error', 'Rsqrd'
        """
        #src : https://towardsdatascience.com/a-gentle-introduction-to-maximum-likelihood-estimation-9fbff27ea12f
        n = len(y)
        # define the likelihood-function
        def _likelihood(estimate):
            # initial guess on parameters
            intercept, beta, sd = estimate[0], estimate[1], estimate[2] 
            yhat = intercept + beta*x 
            # compute likelihood of the observed values being normally distributed with estimated parameters
            negLL = -np.sum(stats.norm.logpdf(y, loc=yhat, scale=sd) )
            return(negLL)
            
        # run through the optimization
        a_res = opt.minimize(_likelihood, estimate, method = 'Nelder-Mead', options={'disp': False})['x'][0:2]
        a_pred = a_res[0] + a_res[1]*x
        
        # standard error
        stderr = np.sqrt(np.sum((a_pred - y)**2)/n)
        # R-value
        ssr = np.sum((a_pred - y)**2)
        sst = np.sum((y - np.mean(y))**2)
        # catch zero values:
        if sst == 0:
            sst = 1e-9
        rval = -(1 - (ssr/sst))
        # create dictionary for results
        d_res = dict([('yinterc', a_res[0]), ('slope', a_res[1]), ('stderr', stderr), ( 'R', rval)])
        return(d_res) 
      
        
        
        
    # getter
    def boxcounts(self):
        """
        Returns the box sizes and box counts.
        
    Returns
    -------
        : tuple (float)
            set ox boxes and corresponding counts
        """
        return (self.boxes, self.counts)




    # getter
    def get_rm(self):
        """
        Returns the recurrence plot that is analysed.
        
    Returns
    -------
        : 2D array (float)
            recurrence matrix
        """
        return self.R




    def lacunarity(self, regression, normalized=True, verb=False, **kwargs):
        """
        Returns the recurrence lacunarity, either fully or in terms of regression parameters (e.g. slope)
        computed from it. Normalization also considers the inverted RP.
      
    Parameters
    ----------
        regression: str
            method for loglog-regression: LS (least-squares), ML (maximum likelihood) or None
        normalized: bool
            return normalized lacunarity
        verb : bool
            print all regression parameters
        regr_param: bool
            return all regression parameters as dictionary
        estimate: list (float)
            initial estimate on regression parameters for ML-regression

    Returns
    -------
        : tuple/float number/dictionary
            one of a) box sizes and lacunarities, b) slope of the lac.-regression or c) full set of regression parameters

    References
    -------
        Braun, Tobias, et al. "Detection of dynamical regime transitions with lacunarity as a 
        multiscale recurrence quantification measure." Nonlinear Dynamics (2021): 1-19.

    Examples
    --------
    - Create an instance of Boxcount for a realization of a random walk. Generate a recurrence plot
      and analyse it with the static box-counting algorithm and a logarithmically increasing set of boxes.
      Compute the box-counting dimension with maximum likelihood regression and print regression parameters.
        
        >>> np.random.seed(123)
        >>> x = np.cumsum(np.random.normal(0,1,1000))
        >>> wmin, wmax, N = 0, 2, 100
        >>> a_boxes = 2*np.unique(np.logspace(wmin, wmax, N, dtype=int))
        >>> # run the box-counting algorithm
        >>> BC = bc.Boxcount(x, method='distance', thresh=2, boxes=a_boxes, glide=False)
        >>> # compute the scaling exponent of the lacunarity curve:
        >>> BC.lacunarity(regression='ML', verb=True)
            {'yinterc': -0.9993807492735187, 'slope': 0.47489832293698464, 'stderr': 0.3132253202577394, 'R': 7.104566369637574}
            0.47489832293698464
    
    - Return the lacunarity values obtained for each box size:
        
        >>> t_lac = BC.lacunarity(regression=None, verb=False)
        >>> t_lac[1]
        array([0.9631434 , 0.92429532, ..., 0.43207166])
        """
        a_lac = np.zeros(self.N)
        for n in range(self.N):
            tmp_counts = self.counts[n]
            # average and variance of counts
            mean, var = np.mean(tmp_counts), np.var(tmp_counts)
            # LACUNARITY
            lac = var/(mean**2) + 1
            a_lac[n] = lac
            # for the normlization, the inverted RP is evaluated
            if normalized:
                tmp_compl_counts = self.boxes[n]**2 - self.counts[n]
                mean, var = np.mean(tmp_compl_counts), np.var(tmp_compl_counts)
                clac = var/(mean**2) + 1
                a_lac[n] = 2 - (1/lac + 1/clac)
        
        # Fit the log(sizes) with log(lac)
        if regression is None:
            outp = (self.boxes, a_lac)
        else:
            outp = self.loglog_regress(x=self.boxes, y=a_lac, rmethod=regression, verb=verb, 
                                       regr_param=kwargs.get("regr_param"), estimate=kwargs.get("estimate"))
        return outp





    def resample_counts(self, output, Nb, regression, **kwargs):
        """
        Bootstrap resampling of recurrence plot box counts. Can be used to obtain confidence intervals
        for the box-counting dimension or recurrence lacunarity. 
        For more information, see
        
        
    .. note::
           This is not equivalent to the hypothesis test performed in (Braun et al 2020, see references) 
           as it is not a test against the hypothesis of stationarity!
           Moreover, in each boostrap runs 'M' boxes are sampled which can not be changed as of now.

        
    Parameters
    ----------
        output: str
            choose between box-counting dimension ('boxdim') or lacunarity ('lac')
        Nb: int
            number of bootstrap runs
        regression : bool
            print all regression parameters

    Returns
    -------
        : list (float)
            list (Nb elements) of either the bootstrapped box-counting dimensions, 
            recurrence lacunarities or regression parameters of the recurrence lacunarities

    References
    -------
        Braun, Tobias, et al. "Detection of dynamical regime transitions with lacunarity as a 
        multiscale recurrence quantification measure." Nonlinear Dynamics (2021): 1-19.


    Examples
    --------
    - Create an instance of Boxcount for a realization of a random walk. Generate a recurrence plot
      and analyse it with the static box-counting algorithm and a logarithmically increasing set of boxes.
      Compute the box-counting dimension with maximum likelihood regression and print regression parameters.
        
        >>> np.random.seed(123)
        >>> x = np.cumsum(np.random.normal(0,1,1000))
        >>> wmin, wmax, N = 0, 2, 100
        >>> a_boxes = 2*np.unique(np.logspace(wmin, wmax, N, dtype=int))
        >>> # run the box-counting algorithm
        >>> BC = bc.Boxcount(x, method='distance', thresh=2, boxes=a_boxes, glide=False)
        >>> # resample the distribution of box counts five times and compute scaling parameters 
        >>> # of the lacunarity ncurves based on least-squares regression.
        >>> BC.resample_counts(output='lac', Nb=100, regression='LS')
            [-0.39592862611100216, -0.3874059856063104, -0.3902924508180553, -0.3791413046318648, -0.3862428074579492]
    
        """
        ## iterate through 'Nb' bootstrap runs whereas each one samples the counts 'M' times
        l_stat_distr = []
        for nb in range(Nb):
            l_smpl = []
            # iterate through box sizes
            for n in range(self.N):
                tmp_counts = np.copy(self.counts[n])
                K = tmp_counts.shape[0]
                # number of resamples
                M = K**2
                # get a fresh seed
                np.random.seed(nb*n)
                # bootstrap (drawing with replacement from box counts)
                tmp_smpl = np.random.choice(tmp_counts.ravel(), M).reshape((K, K))
                l_smpl.append(tmp_smpl)
            # make sure that lacunarity computation runs on the resampled counts
            self.counts = l_smpl 
            ## For which statistic should the distribution be generated?
            if output == 'boxdim':
                tmp_stat = self.box_dimension(regression, verb=kwargs.get("verb"))
            elif output == 'lac':
                tmp_stat = self.lacunarity(regression, normalized=kwargs.get("normalized"), 
                                           verb=kwargs.get("verb"), regr_param=kwargs.get("regr_param"))
            # distribution of values:
            l_stat_distr.append(tmp_stat)
            # for next run, the counts must be the real counts again
            self.counts = np.copy(self._counts)
        return l_stat_distr
