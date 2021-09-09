#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 18 11:22:17 2021

@author: tobraun
"""

import numpy as np
from scipy.spatial import distance_matrix
import scipy

class RP:
    
    """
    Class RP for computing recurrence plots from univariate time series.

    The recurrence_plot class supports time-delay embedding of multi-dimensional time series with
    known embedding dimension and delay, computation of euclidean distance matrices and
    computation of recurrence matrices based on four common threshold selection criteria.
    
    Note that this is a very sparse implementation with limited functionality and more
    comprehensive implementations (incl. embedding, parameter estimation, RQA...) can be found elsewhere.
    For a comprehensive summary of the recurrence plot method refer to [Marwan2007].

    **Examples:**

     - Create an instance of RP with fixed recurrence rate of 10%
       and without embedding:

           RP(x=time_series, method='frr', thresh=0.1)

     - Create an instance of RP with a fixed recurrence threshold
       in units of the time series standard deviation (2*stdev) and with embedding:

           RP(x=time_series, dim=2, tau=3, method='stdev', thresh=2)
           
       Obtain the recurrence matrix:
           
           RP(x=time_series, dim=2, tau=3, method='stdev', thresh=2).rm()
    """
    
    def __init__(self, x, method, thresh, **kwargs):
        """
        Initialize an instance of RP.

        The following keywords are required: method, thresh
        The dim, tau keywords are optional.

        If given a univariate/scalar time series, embedding parameters may be specified to apply
        time-delay embedding. If the time series multi-dimensional (non-scalar), no embedding
        can be applied and the input is treated as an embedded time series.

        :type x: 2D array (time, dimension)
         :arg x: The time series to be analyzed, can be scalar or
            multi-dimensional.
        :type dim: int
         :arg int dim: embedding dimension (>1)
        :type tau: int
         :arg tau: embedding delay
        :type method: str
         :arg method: estimation method for the vicinity threshold 
                      epsilon (`distance`, `frr`, `stdev`, `fan`)
        :type thresh: float
         :arg thresh: threshold parameter for the vicinity threshold,
                     depends on the specified method (`epsilon`, 
                    `recurrence rate`, `multiple of standard deviation`,
                    `fixed fraction of neighbours`)
        """
        #  Store time series as float
        self.x = x.copy().astype("float32")
        self.method = method
        self.thresh = thresh 
        
        #  Apply time-delay embedding: get embedding dimension and delay from **kwargs
        self.dim = kwargs.get("dim")
        self.tau = kwargs.get("tau")
        if self.dim is not None and self.tau is not None:
            assert (self.dim > 0) and (self.tau>0), "Negative embedding parameter(s)!"
            #  Embed the time series
            self.embedding = self.embed()
        elif (self.dim is not None and self.tau is None) or (self.dim is None and self.tau is not None):
            raise NameError("Please specify either both or no embedding parameters.")
        else:
            if x.ndim > 1:
                self.embedding = self.x
            else: 
                self.embedding = self.x.reshape(x.size,1)
        
        # default metric: euclidean
        self.metric = kwargs.get("metric")
        if self.metric is None:
            self.metric = 'euclidean'
        assert (type(self.metric) is str), "'metric' must specified as string!"

        # Set threshold based on one of the four given methods (distance, fixed rr, fixed stdev, fan)
        # and compute recurrence matrix:
        self.R = self.apply_threshold()


    
    def apply_threshold(self):
        """
        Apply thresholding to the distance matrix by one of four methods:
            *  'distance': no method, expects value for vicinity threshold
            *  'frr': fixed recurrence rate, expects specification of distance-distr. quantile (.xx) 
            *  'stdev': standard deviation of time series, expects multiple of standard deviation
            *  'fan': fixed amounbt of neighbors, expects fraction of fixed neighbors

        :rtype: 2D array (integer)
         :return: recurrence matrix
        """
        # compute distance matrix
        dist = RP.dist_mat(self, metric=self.metric)
        # initialize recurrence matrix
        a_rm = np.zeros(dist.shape, dtype="int")
        # different methods
        method = self.method
        thresh = self.thresh
        if method == "distance":
            i = np.where(dist <= thresh)
            a_rm[i] = 1
        elif method == "frr":
            eps = np.nanquantile(dist, thresh)
            i = np.where(dist <= eps)
            a_rm[i] = 1
        elif method == "stdev":
            eps = thresh*np.nanstd(self.x)
            i = np.where(dist <= eps)
            a_rm[i] = 1
        elif method == "fan":
            nk = np.ceil(thresh * a_rm.shape[0]).astype("int")
            i = (np.arange(a_rm.shape[0]), np.argsort(dist, axis=0)[:nk])
            a_rm[i] = 1
        else:
            raise NameError("'method' must be one of 'distance', 'frr', 'stdev' or 'fan'.")
        return a_rm


    def dist_mat(self, metric):
        """
        Returns a square-distance matrix with some specified metric.
        The following metrics can be used:
        'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine', 'dice', 'euclidean',
        'hamming', 'jaccard', 'jensenshannon', 'kulsinski', 'mahalanobis', 'matching', 'minkowski',
        'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule'
        
        :rtype: 2D array (float)
         :return: distance matrix
        """
        z = self.embedding
        # using the scipy.spatial implementation:
        if z.ndim == 1:
            z = z.reshape((z.size, 1))
        a_dist = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(z, metric=metric), force='tomatrix')
        if np.all(np.isnan(a_dist)):
            raise ValueError("Distance matrix only contains NaNs.")
        return a_dist
    

    def embed(self):
        """
        Time-delay embedding: embeds a scalar time series 'x' in 'dim' dimensions with time delay 'tau'.

        :rtype: 2D array (float)
         :return: embedded time series
        """
        K = (self.dim-1)*self.tau
        assert (K<self.x.size), "Choice of embedding parameters exceeds time series length."
        # embedd time series:
        a_emb = np.asarray([np.roll(self.x, -d*self.tau)[:-K] for d in range(self.dim)]).T
        return a_emb
    

    def rm(self):
        """
        Returns the (square) recurrence matrix.

        :rtype: 2D array (int)
         :return: recurrence matrix
        """
        return self.R
    

