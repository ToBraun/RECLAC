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
    
    def __init__(self, x, method, thresh, compute_rp=True, **kwargs):        
        """
        Class RP for computing recurrence plots from univariate time series.

        The recurrence_plot class supports time-delay embedding of multi-dimensional time series with
        known embedding dimension and delay, computation of euclidean distance matrices and
        computation of recurrence matrices based on four common threshold selection criteria.
        Note that this is a very sparse implementation with limited functionality and more
        comprehensive implementations (incl. embedding, parameter estimation, RQA...) can be found elsewhere.
        For a comprehensive summary of the recurrence plot method refer to [Marwan2007].
            
        If given a univariate/scalar time series, embedding parameters may be specified to apply
        time-delay embedding. If the time series multi-dimensional (non-scalar), no embedding
        can be applied and the input is treated as an embedded time series.
        If a recurrence plot should be given as input, 'compute_rp' has to be set to False.
        
    Parameters
    ----------
        x: 2D array (time, dimension)
            The time series to be analyzed, can be scalar or multi-dimensional.
        dim : int, optional
            embedding dimension (>1)
        tau : int, optional
            embedding delay
        method : str
             estimation method for the vicinity threshold 
             epsilon (`distance`, `frr`, `stdev`, `fan`)
        thresh: float
            threshold parameter for the vicinity threshold,
            depends on the specified method (`epsilon`, 
            `recurrence rate`, `multiple of standard deviation`,
            `fixed fraction of neighbours`)
        
        
    Examples
    --------
        
    - Create an instance of RP with fixed recurrence rate of 10%
      and without embedding:

           >>> import RECLAC.recurrence_plot as rec
           >>> RP(x=time_series, method='frr', thresh=0.1)
           
    - Create an instance of RP with a fixed recurrence threshold
      in units of the time series standard deviation (2*stdev) and with embedding:

           >>> RP(x=time_series, dim=2, tau=3, method='stdev', thresh=2)
    
    - Obtain the recurrence matrix:
           
           >>> a_rm = RP(x=time_series, dim=2, tau=3, method='stdev', thresh=2).rm()
    """
        
        #  Store time series as float
        self.x = x.copy().astype("float32")
        self.method = method
        self.thresh = thresh 
        
        if compute_rp:
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
            # 'back-up' (private) variable for self.counts to restore value when self.counts is altered
            self._R = np.copy(self.R)
        # RP is passed as x argument
        else:
            assert (x.ndim == 2), "If a recurrence matrix is provided, it has to be a 2-dimensional array."
            assert ~np.all(np.isnan(x)), "Recurrence matrix only contains NaNs."
            self.R = x



    
    def apply_threshold(self):
        """
        Apply thresholding to the distance matrix by one of four methods:
            
        *  'distance': no method, expects value for vicinity threshold
        
        *  'frr': fixed recurrence rate, expects specification of distance-distr. quantile (.xx) 
        
        *  'stdev': standard deviation of time series, expects multiple of standard deviation
        
        *  'fan': fixed amounbt of neighbors, expects fraction of fixed neighbors

        
    Returns
    -------
        
        R: 2D array (integer)
            recurrence matrix
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
        
    Parameters
    ----------
        metric: str
            Metric that is used for distance computation.
        
    Returns
    -------
        
        R: 2D array (float)
            distance matrix
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

    Returns
    -------
        
        R: 2D array (float)
             embedded time series
        """
        K = (self.dim-1)*self.tau
        assert (K<self.x.size), "Choice of embedding parameters exceeds time series length."
        # embedd time series:
        a_emb = np.asarray([np.roll(self.x, -d*self.tau)[:-K] for d in range(self.dim)]).T
        return a_emb
    

    def rm(self):
        """
        Returns the (square) recurrence matrix.

    Returns
    -------
        
        R: 2D array (int)
             recurrence matrix
        """
        return self.R
    

    
    def line_hist(self, linetype, border=None):
        """
        Extracts all diagonal/vertical lines from a recurrence matrix. The 'linetype'
        arguments specifies which lines should be analysed. Returns all identified
        line lengths and the line length histogram.
        Since the length of border lines is generally unknown, they can be discarded or
        replaced by the mean/max line length.
        
    Parameters
    ----------
        linetype: str
            specifies whether diagonal ('diag') or vertical ('vert') lines
            should be extracted
        border: str, optional
            treatment of border lines: None, 'discard', 'kelo', 'mean' or 'max'
        
        
    Returns
    -------
        
        a_ll, a_bins, a_lhist: tuple of three 1D float arrays
            line lengths, bins, histogram
            
    Examples
    --------
        
    - Create an instance of RP with fixed recurrence rate of 10% for a noisy
      sinusoidal and obtain the diagonal line length histogram without border lines:

        >>> import RECLAC.recurrence_plot as rec
        >>> # sinusoidal with five periods and superimposed white Gaussian noise
        >>> a_ts = np.sin(2*math.pi*np.arange(100)/20) + np.random.normal(0, .25, 100)
        >>> # define a recurrence plot instance with a fixed recurrence rate of 10%
        >>> RP = rec.RP(a_ts, method='frr', dim=2, per=5, thresh=.1)
        >>> # obtain line histogram for diagonal lines while border lines are discarded
        >>> _, a_bins, a_freqs = RP.line_hist(linetype='diag', border='discard')
        >>> a_bins, a_freqs
        (array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]), array([152,  60,  16,  11,   1]))
           
    - Obtain the vertical line length histogram with no action on border lines:

        >>> _, a_vbins, a_vfreqs = RP.line_hist(linetype='vert', border=None)
        >>> a_vbins, a_vfreqs
        (array([0.5, 1.5, 2.5, 3.5, 4.5, 5.5]), array([268, 158,  71,  18,   5]))
        """
        N = self.R.shape[0]
        a_ll = np.array([])
        # 'counter' counts border lines
        counter = 0
        for n in range(1, N):
            # grab the n-th diagonal
            tmp_line = self._extract_line(n, linetype)
            # run length encoding
            tmp_rle = RP._rle(tmp_line)
            
            ## Border lines
            if border is not None:
                if tmp_rle[0][0] == 1:
                    tmp_rle = tmp_rle[1:,]
                    counter += 1
                try:
                    if tmp_rle[-1][0] == 1: 
                        tmp_rle = tmp_rle[:-1,]
                        counter += 1
                except IndexError:
                    tmp_rle = tmp_rle
                    
            ## Find diagonal lines
            tmp_ind = np.where(tmp_rle[:,0] == 1)
            # collect their lengths
            tmp_lengths = tmp_rle[tmp_ind, 1].ravel()
            if tmp_lengths.size > 0:
                a_ll = np.hstack([a_ll, tmp_lengths])
                ## Append border line substitutes (if desired)
                if border == 'mean':
                    avgll = np.mean(a_ll)
                    a_ll = np.hstack([a_ll, np.repeat(avgll, counter)])
                elif border == 'max':
                    maxll = np.max(a_ll)
                    a_ll = np.hstack([a_ll, np.repeat(maxll, counter)])
                elif border == 'kelo':
                    maxll = np.max(a_ll)
                    a_ll = np.hstack([a_ll, maxll])
        
        # any lines?
        if a_ll.size > 0:                
            a_bins = np.arange(0.5, np.max(a_ll) + 0.1 + 1, 1.)
            a_lhist, _ = np.histogram(a_ll, bins=a_bins)
            return a_ll, a_bins, a_lhist
        else:
            raise ValueError("No lines could be identified.")
            return None



    def _extract_line(self, n, linetype):
        """
        Extracts the n-th diagonal/column from a recurrence matrix, depending on
        whether diagonal (linetype='diag') or vertical (linetype='vert') lines are
        desired.
        
    Parameters
    ----------
        n: int
            index of diagonal/column of the RP (0 corresponds to LOI for diagonals)
        linetype: str
            specifies whether diagonal ('diag') or vertical ('vert') lines
            should be extracted
        
    Returns
    -------
        
        1D float array
            n-th diagonal/column of recurrence matrix
        """
        if linetype == 'diag':
            return np.diag(self.R, n)
        elif linetype == 'vert':
            return self.R[:,n]
        else:
            print("Specification error: 'linetype' must be one of 'diag' or 'vert'.")



    @staticmethod
    def _rle(sequence):
        """
        Run length encoding: count consecutive occurences of values in a sequence.
        Applied to binary sequences (diagonals/columns of recurrence matrix) to obtain
        line lengths.
        
    Parameters
    ----------
        sequence: 1D float array
            sequence of values (0s and 1s for recurrence plots)

    Returns
    -------
        
        2D float array
            run values (first column) and run lengths (second column)
        """
        #src:  https://github.com/alimanfoo
       ## Run length encoding: Find runs of consecutive items in an array.
    
        # ensure array
        x = np.asanyarray(sequence)
        if x.ndim != 1:
            raise ValueError('only 1D array supported')
        n = x.shape[0]
    
        # handle empty array
        if n == 0:
            return np.array([]), np.array([]), np.array([])
    
        else:
            # find run starts
            loc_run_start = np.empty(n, dtype=bool)
            loc_run_start[0] = True
            np.not_equal(x[:-1], x[1:], out=loc_run_start[1:])
            run_starts = np.nonzero(loc_run_start)[0]
    
            # find run values
            run_values = x[loc_run_start]
    
            # find run lengths
            run_lengths = np.diff(np.append(run_starts, n))

            # stack and return
            return np.vstack([run_values, run_lengths]).T


    @staticmethod
    def _fraction(bins, hist, lmin):
        """
        Returns the fraction of lines that are longer than 'lmin' based on the line
        length histogram. For diagonal (vertical) lines, this corresponds to DET (LAM).
        
    Parameters
    ----------
        bins: 1D float array
            bins of line length histogram
        hist: 1D float array
            frequencies of line lengths within each bin 
        lmin: int value
             minimum line length

    Returns
    -------
        
        rq: float value
            fraction of diagonal/vertical lines that exceed 'lmin' (DET/LAM)
        """
        # find fraction of lines larger than lmin
        a_Pl = hist.astype('float')
        a_l = (0.5 * (bins[:-1] + bins[1:])).astype('int')
        ind = a_l >= lmin
        # compute fraction
        a_ll_large = a_l[ind] * a_Pl[ind]
        a_ll_all = a_l * a_Pl
        rq = a_ll_large.sum() / a_ll_all.sum()
        return rq



    def RQA(self, lmin, measures = 'all', border=None):
        """
        Performs a recurrence quantification analysis (RQA) on a recurrence matrix based on
        a list of (traditional) recurrence quantification measures. Returns the following
        nine measures by default:
        - recurrence rate
        - DET
        - average diagonal line length
        - maximum diagonal line length
        - LAM
        - average vertical line length
        - maximum vertical line length
        - average white vertical line length(/recurrence time)
        - maximum white vertical line length(/recurrence time)
        
        If only quantifiers based on diagonal/vertical/white lines are desired, this can
        be restricted with the 'measure' argument.
        
        
    Parameters
    ----------
        lmin: int value
             minimum line length
        measures: str
            determines which recurrence quantification measures are computed
            ('all', 'diag', 'vert', 'white')
        border: str
            treatment of border lines: None, 'discard', 'mean' or 'max'
         
            
    Returns
    -------
        
        d_rqa: float dictionary
            recurrence quantification measures
            
    Examples
    --------
        
    - Create an instance of RP with fixed recurrence rate of 10% for a noisy
      sinusoidal and run a full recurrence quantification analysis:

        >>> import RECLAC.recurrence_plot as rec
        >>> # sinusoidal with five periods and superimposed white Gaussian noise
        >>> a_ts = np.sin(2*math.pi*np.arange(100)/20) + np.random.normal(0, .25, 100)
        >>> # define a recurrence plot instance with a fixed recurrence rate of 10%
        >>> RP = rec.RP(a_ts, method='frr', thresh=.1)
        >>> # compute all RQA measures with no border correction:
        >>> RP.RQA(lmin=2, measures='all', border=None)
        {'RR': 0.10005540166204986,
        'DET': 0.6064356435643564,'avgDL': 2.5257731958762886,'maxDL': 9,
         'LAM': 0.7002237136465325,'avgVL': 2.484126984126984,'maxVL': 5,
         'avgWVL': 15.614931237721022,'maxWVL': 86}


    - Run only a recurrence quantification analysis that considers diagonal measures
      on lines of minimum length 3 whereas border lines are set to the average diagonal 
      line length:

        >>> RP.RQA(lmin=3, measures='diag', border='mean')
        {'RR': 0.10005540166204986,
         'DET': 0.12262958280657396,'avgDL': 3.4642857142857144,'maxDL': 5,
         'LAM': None, 'avgVL': None, 'maxVL': None,
         'avgWVL': None, 'maxWVL': None} 
        """
        DET, avgDL, maxDL, LAM, avgVL, maxVL, avgWL, maxWL = np.repeat(None, 8)
        # recurrence rate
        rr = np.sum(self.R)/(self.R.size)
        # diagonal line structures
        if (measures == 'diag') or (measures == 'all'):
            a_ll, a_bins, a_nlines  = self.line_hist(linetype='diag', border=border)
            DET = RP._fraction(bins = a_bins, hist = a_nlines, lmin = lmin)
            a_llsub = a_ll[a_ll >= lmin]
            avgDL = np.mean(a_llsub)
            maxDL = np.max(a_llsub).astype(int)
        # vertical line structures
        if (measures == 'vert') or (measures == 'all'):
            a_ll, a_bins, a_nlines  = self.line_hist(linetype='vert', border=border)
            LAM = RP._fraction(bins = a_bins, hist = a_nlines, lmin = lmin)
            a_llsub = a_ll[a_ll >= lmin]
            avgVL = np.mean(a_llsub)
            maxVL = np.max(a_llsub).astype(int)
        # white vertical line structures/ recurrence times
        if (measures == 'white') or (measures == 'all'):
            self.R = 1 - self.R
            a_ll, a_bins, a_nlines  = self.line_hist(linetype='vert', border=border)
            a_llsub = a_ll[a_ll >= lmin]
            avgWL = np.mean(a_llsub)
            maxWL = np.max(a_llsub).astype(int)
            # restore value
            self.R = self._R
        
        d_rqa = dict([('RR', rr),
                      ('DET', DET), ('avgDL', avgDL), ('maxDL', maxDL), 
                      ('LAM', LAM), ('avgVL', avgVL), ('maxVL', maxVL),
                      ('avgWVL', avgWL), ('maxWVL', maxWL)])
        
        return d_rqa


