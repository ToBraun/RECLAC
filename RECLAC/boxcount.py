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
#import sys
#sys.path.insert(0, "/home/tobraun/Desktop/PhD/projects/#1_lacunarity")
#from package_alpha.recurrence_plot import RP

class Boxcount(RP):
    
    """
    Class Boxcount for computing recurrence quantification measures from a box-counting analysis.

    Currently supports the computation of the box-counting dimension and recurrence lacunarity
    for recurrence plots. Recurrence plots are computed based on the RP-class. 
    
    Box-counting can be applied through a (computationally faster) static box-counting algorithm
    or a more comprehensive gliding box algorithm. For the computation of the box-counting dimension
    and regression parameters for recurrence lacunarity, a basic least-squares or a more robust 
    maximum-likelihood regression of the log-counts against the log-sizes are provided.
    Confidence intervals can be returned based on a bootstrap resampling scheme.
    
    If no recurrence plot should be computed but instead is directly given to the box-counting as an input,
    use 'compute_rp = False'.


    **Examples:**

     - Create an instance of Boxcount with fixed recurrence rate of 10% & without embedding and analyse it 
       with the static box-counting algorithm and a linearly increasing set of boxes. Symmetry is used.

            Boxcount(x, method='frr', thresh=0.1, boxes=np.arange(2, 200), glide=False, sym=True)

     - Create an instance of Boxcount with fixed recurrence rate of 10% & without embedding and analyse it 
       (after a 90° rotation) with the gliding box-counting algorithm. Return the box-counting dimension.

            bc = Boxcount(x, method='frr', thresh=0.1, boxes=np.arange(2, 200), glide=True, gridpos=1)
            bc.box_dimension(regression='LS')
           
       Directly give a RP as input and compute recurrence lacunarity with full output of regression parameters.
       Obtain the respective bootstrap distribution based on 2000 bootstrap runs.
           
            bc = Boxcount(x, boxes=np.arange(2, 200), compute_rp = False)
            bc.lacunarity(regression='ML', regr_param=True)
            bc.resample_counts(output = 'lac', Nb = 2000, regression = 'ML', regr_param = True)
    """
    
    def __init__(self, x, boxes, glide = False, gridpos = 0, compute_rp = True, **kwargs):
        """
        Initialize an instance of Boxcount.

        The following keywords are required: x, boxes, glide, gridpos, compute_rp
        The sym keyword and the recurrence parameters are optional.
        
        :type x: float array
         :arg x: either a 1D time series or a 2D recurrence matrix (compute_rp = False)
        :type boxes: float 1D array
         :arg boxes: array of boxes sizes (ascending/descending order)
        :type glide: bool
         :arg glide: static or gliding box algorithm
        :type gridpos: float number
         :arg gridpos: number of 90° rotations to apply to the RP (0,1,2,3)
        :type compute_rp: bool
         :arg compute_rp: compute an RP based on RP class or feed one as input
        """
        if compute_rp:
            method = kwargs.get('method')
            thresh = kwargs.get('thresh')
            if (method is None) or (thresh is None):
                raise NameError("Please provide a method for thresholding and a threshold.")
            assert (x.ndim == 1), "If no recurrence matrix is provided, please provide a 1-dimensional time series."
            #  Initialize the underlying RecurrencePlot object
            RP.__init__(self, x=x, method=method, thresh=thresh)
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
        self._counts = np.copy(self.counts)
            
    
    # setter
    def _stat_count(self, k):
        """
        Static box-counting algorithm. Computes the box counts for a given (square) box size 'k'.

        .. note::
           Sets a value for the current box count attribute 'S'. No return value!

        :type k: int number
         :arg k: box width
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

        :type k: int number
         :arg k: box width
        :type sym: bool
         :arg sym: symmetry of the recurrence marix (True/False)
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

        :type regression: str
         :arg regression: method for loglog-regression: LS (least-squares), ML (maximum likelihood) or None
        :type verb: bool
         :arg verb: print all regression parameters
         
        :rtype: (optional) float tuple/dictionary/number
         :return: one of a) box sizes and box counts, b) box-counting dimension or
                  c) full set of regression parameters
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
            outp = self.loglog_regress(x=self.boxes, y=a_covered, rmethod=regression, verb=verb, 
                                       regr_param=kwargs.get("regr_param"), estimate=kwargs.get("estimate"))
        return outp


    @staticmethod
    def loglog_regress(self, x, y, rmethod='LS', verb=True, estimate=None, regr_param = False):
        """
        Returns the result of a log-log regression. Uses either a least-squares or a (robust)
        maximum likelihood estimation procedure. The latter can be useful if a slope estimate
        should be obtained even though outliers exist. It requires an initial estimate on the
        regression parameters. 

        :type x, y: float array
         :arg x, y: x- and y-coordinates (logarithmic box sizes and box counts)
        :type rmethod: str
         :arg rmethod: method for loglog-regression: LS (least-squares), ML (maximum likelihood) or None
        :type verb: bool
         :arg verb: print all regression parameters
        :type estimate: float list
         :arg estimate: initial estimate on regression parameters for ML-regression
        :type regr_param: bool
         :arg regr_param: return all regression parameters as dictionary
        
        :rtype: (optional) float dictionary/float number
         :return: one of a) full set of regression parameters or b) slope of the lac.-regression 
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
            d_res = (self.MLEregr(x = a_fit[0,], y = a_fit[1,], estimate = estimate))
#        # results
        if verb: 
            print(d_res)
        if regr_param: 
            return d_res
        else:
            return d_res['slope']


    @staticmethod
    def MLEregr(self, x, y, estimate):
        """
        Returns the result of a maximum likelihood regression. This can be useful if a slope estimate
        should be obtained even though outliers exist. It requires an initial estimate on the
        regression parameters. 
    
        :type x, y: float array
         :arg x, y: x- and y-coordinates (logarithmic box sizes and box counts)
        :type estimate: float list
         :arg estimate: initial estimate on regression parameters for ML-regression
        
        :rtype: float dictionary
         :return: regression parameters: 'y-intercept', 'slope', 'standard error', 'Rsqrd'
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
        :rtype: float tuple
         :return: box sizes
        """
        return (self.boxes, self.counts)



    def lacunarity(self, regression, normalized=True, verb=False, **kwargs):
        """
        Returns the recurrence lacunarity, either fully or in terms of regression parameters (e.g. slope)
        computed from it. Normalization also considers the inverted RP, see
        >Braun, Tobias, et al. "Detection of dynamical regime transitions with lacunarity as a 
        multiscale recurrence quantification measure." Nonlinear Dynamics (2021): 1-19.<

        
        :type regression: str
         :arg regression: method for loglog-regression: LS (least-squares), ML (maximum likelihood) or None
        :type normalized: bool
         :arg normalized: return normalized lacunarity
        :type verb: bool
         :arg verb: print all regression parameters
        :type regr_param: bool
         :arg regr_param: return all regression parameters as dictionary
         
        :rtype: (optional) tuple/float number/dictionary
         :return: one of a) box sizes and lacunarities, b) slope of the lac.-regression or
                  c) full set of regression parameters
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
        >Braun, Tobias, et al. "Detection of dynamical regime transitions with lacunarity as a 
        multiscale recurrence quantification measure." Nonlinear Dynamics (2021): 1-19.<

        .. note::
           This is not equivalent to the hypothesis test performed in (Braun et al 2020) as it is
           not a test against the hypothesis of stationarity!
           Moreover, in each boostrap runs 'M' boxes are sampled which can not be changed as of now.

        
        :type output: str
         :arg output: choose between box-counting dimension ('boxdim') or lacunarity ('lac')
        :type Nb: int number
         :arg Nb: number of bootstrap runs
        :type regression: bool
         :arg regression: print all regression parameters

        :rtype: (optional) float list 
         :return: list (Nb elements) of either the bootstrapped box-counting dimensions,
                  recurrence lacunarities or regression parameters of the recurrence lacunarities

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
