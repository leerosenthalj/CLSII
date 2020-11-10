import os
import csv
import pdb
from multiprocessing import Pool

import numpy as np
import pandas as pd
import scipy.optimize as op
import scipy.special as spec
from scipy import stats
from scipy.interpolate import RegularGridInterpolator
import scipy.ndimage
from scipy.linalg import cho_factor, cho_solve

import emcee
import celerite
import radvel

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import corner

class Completeness(object):
    """Object to handle a suite of injection/recovery tests

    Args:
        recoveries (DataFrame): DataFrame of injection/recovery tests from Injections class
        xcol (string): (optional) column name for independent variable. Completeness grids and
            interpolator will work in these axes
        ycol (string): (optional) column name for dependent variable. Completeness grids and
            interpolator will work in these axes

    """
    def __init__(self, recoveries, xcol='inj_au', ycol='inj_msini'):

        self.recoveries = recoveries

        self.xcol = xcol
        self.ycol = ycol

        self.grid = None
        self.interpolator = None

    def completeness_grid(self, xlim, ylim, resolution=20, xlogwin=0.5, ylogwin=0.5):

        xgrid = np.logspace(np.log10(xlim[0]),
                            np.log10(xlim[1]),
                            resolution)
        ygrid = np.logspace(np.log10(ylim[0]),
                            np.log10(ylim[1]),
                            resolution)

        xinj = self.recoveries[self.xcol]
        yinj = self.recoveries[self.ycol]

        good = self.recoveries['recovered']

        z = np.zeros((len(ygrid), len(xgrid)))
        last = 0
        for i,x in enumerate(xgrid):
            for j,y in enumerate(ygrid):
                xlow  = 10**(np.log10(x) - xlogwin/2)
                xhigh = 10**(np.log10(x) + xlogwin/2)
                ylow  = 10**(np.log10(y) - ylogwin/2)
                yhigh = 10**(np.log10(y) + ylogwin/2)

                xbox = yinj[np.where((xinj <= xhigh) & (xinj >= xlow))[0]]
                if len(xbox) == 0 or y > max(xbox) or y < min(xbox):
                    z[j, i] = np.nan
                    continue

                boxall = np.where((xinj <= xhigh) & (xinj >= xlow) &
                                  (yinj <= yhigh) & (yinj >= ylow))[0]
                boxgood = np.where((xinj[good] <= xhigh) &
                                   (xinj[good] >= xlow) & (yinj[good] <= yhigh) &
                                   (yinj[good] >= ylow))[0]

                if len(boxall) > 10:
                    z[j, i] = float(len(boxgood))/len(boxall)
                    last = float(len(boxgood))/len(boxall)
                else:
                    z[j, i] = np.nan

        self.grid = (xgrid, ygrid, z)

    def interpolate(self, x, y, refresh=False):

        if self.interpolator is None or refresh:
            assert self.grid is not None, "Must run Completeness.completeness_grid()."
            zi = self.grid[2].T
            self.interpolator = RegularGridInterpolator((self.grid[0], self.grid[1]), zi,
                                                        bounds_error=False, fill_value=0.001)

        return self.interpolator(np.array([np.atleast_1d(x), np.atleast_1d(y)]).T)


class Hierarchy(object):
    """Do hierarchical Bayesian sampling of occurrence posteriors, based on DFM et al. 2014.
    Args:
        pop (pandas DataFrame): dataframe of planet parameter chains

    """
    def __init__(self, pop, completeness, res=4, bins=np.array([[[np.log(0.02), np.log(20)],
                                                                 [np.log(2.), np.log(6000)]]]),
                                                                  chainname='occur_chains.csv'):
        self.pop          = pop
        self.completeness = completeness # Completeness grid, defined as class object above.
        self.completeness.completeness_grid([0.01, 40], [3, 7000])
        # Fill in completeness nans.
        self.completeness.grid[2][np.isnan(self.completeness.grid[2])] = 1. #0.99


        self.res = res # Resolution for logarithmic completeness integration.
        self.bins = bins # Logarithmic bins in msini/axis space.
        self.nbins = len(self.bins)
        self.lna_edges = np.unique(self.bins[:, 0])
        self.lnm_edges = np.unique(self.bins[:, 1])
        self.nabins = len(self.lna_edges) - 1
        self.nmbins = len(self.lnm_edges) - 1
        #self.na = len(self.lna_edges)
        #self.nm = len(self.lnm_edges)

        # Compute bin centers and widths.
        self.bin_widths  = np.diff(self.bins)
        self.bin_centers = np.mean(self.bins, axis=2)
        self.bin_areas   = self.bin_widths[:,0]*self.bin_widths[:,1]

        # Pre-compute integrated completeness for each bin.
        self.Qints = np.zeros(self.nbins)
        for n, binn in enumerate(self.bins):
            for i in np.arange(4): #self.res
                for j in np.arange(4):
                    lna_av = binn[0][0] + (0.25*i + 0.125)*(binn[0][1] - binn[0][0])
                    lnm_av = binn[1][0] + (0.25*j + 0.125)*(binn[1][1] - binn[1][0])
                    self.Qints[n] += (self.bin_areas[n][0]/self.res**2) * \
                                      self.completeness.interpolate(np.exp(lna_av),
                                                                    np.exp(lnm_av))

        axis  = []
        msini = []
        self.planetnames = np.unique([x[:-2] + x[-1] for x in pop.columns])
        self.starnames   = np.unique([x[:-1] for x in self.planetnames])
        self.nplanets    = len(self.planetnames)
        self.nsamples    = len(self.pop)
        self.nstars      = len(self.starnames)

        medians = pop.median() # Along chain axis, once using chains.
        for name in self.planetnames:
            axis.append(medians[[name[:-1] + 'a' + name[-1]]][0])
            msini.append(medians[[name[:-1] + 'M' + name[-1]]][0])
        self.pop_med = pd.DataFrame.from_dict({'axis':axis, 'msini':msini})

        self.chainname = chainname

    def max_like(self):
        ### Approximate max-likelihood occurrence values, with which to seed MCMC.
        mlvalues = np.empty((0, 2))
        for n, binn in enumerate(self.bins):
            # Integrate completeness across each individual bin.
            a1 = np.exp(binn[0][0])
            a2 = np.exp(binn[0][1])
            M1 = np.exp(binn[1][0])
            M2 = np.exp(binn[1][1])
            planets = self.pop_med.query('axis >= @a1 and axis < @a2 and \
                                         msini >= @M1 and msini < @M2')
            nplanets = len(planets)
            #ml  = np.log(nplanets/self.Qints[n])
            ml  = nplanets/self.Qints[n]
            uml = ml/np.sqrt(nplanets)
            if not np.isfinite(ml):
                ml = 0.01 #-7
            if not np.isfinite(uml):
                uml = 1.
            mlvalues = np.append(mlvalues, np.array([[ml, uml]]), axis=0)
        mlvalues[np.isnan(mlvalues)] = 0.01
        mlvalues[mlvalues == 0] = 0.01
        self.mlvalues = mlvalues
        self.ceiling = np.amax(mlvalues)

    def occurrence(self, lna, lnm, theta):
        # Select appropriate bins, given lna & lnm.
        ia = np.atleast_1d(np.digitize(lna, self.lna_edges) - 1)
        im = np.atleast_1d(np.digitize(lnm, self.lnm_edges) - 1)
        iao = np.copy(ia)
        imo = np.copy(im)
        ia[ia < 0] = 0
        im[im < 0] = 0
        ia[ia > self.nabins - 1] = self.nabins - 1
        im[im > self.nmbins - 1] = self.nmbins - 1

        # Logarithmic
        #occur = np.exp(theta[ia + im*self.nabins])
        # Linear
        occur = theta[ia + im*self.nabins]
        # Return filler value for samples outside of the bin limits.
        occur[iao < 0] = 0.01 #-10
        occur[imo < 0] = 0.01 #-10
        occur[iao > self.nabins - 1] = 0.01 #-10
        occur[imo > self.nmbins - 1] = 0.01 #-10
        return occur

    def lnlike(self, theta):
        # Linear
        if np.any((theta <= 0) + (theta > 2*self.ceiling)):
            return -np.inf
        # Logarithmic
        #if np.any((theta <= -11) + (theta > self.ceiling + 1)):
        #    return -np.inf
        sums = []
        for planet in self.planetnames:
            #print(planet)
            probs = []
            sample_a = np.array(self.pop[planet[:-2] + '_a' + planet[-1]])
            sample_M = np.array(self.pop[planet[:-2] + '_M' + planet[-1]])
            probs = self.completeness.interpolate(sample_a, sample_M)*self.occurrence(
                                           np.log(sample_a), np.log(sample_M), theta)
            sums.append(np.sum(probs))

        # Integrate the observed occurrence over all bins.
        nexpect = 0
        for i, binn in enumerate(self.bins):
            for j in np.arange(4):
                for k in np.arange(4):
                    lna_av = binn[0][0] + (0.25*j + 0.125)*(binn[0][1] - binn[0][0])
                    lnm_av = binn[1][0] + (0.25*k + 0.125)*(binn[1][1] - binn[1][0])
                    nexpect += (self.bin_areas[i][0]/16)*self.completeness.interpolate(
                                                                        np.exp(lna_av),
                                                                        np.exp(lnm_av))*self.occurrence(
                                                                        lna_av, lnm_av, theta)
        ll = -nexpect + np.sum(np.log(np.array(sums)/self.nsamples))
        if not np.isfinite(ll):
            return -np.inf
        return ll

    def gpprior(self, theta, mu, l0, la, lm):
        ### Prior on occurrence. Gaussian process, for smoothly changing bin heights.
        # Logarithmic
        #if not -15 < mu < 15:
        #    return -np.inf
        # Linear
        if not 0 < mu < 2*self.ceiling:
            return -np.inf
        if not -2 < l0 < 6:
            return -np.inf
        if not -2 < la < 1:
            return -np.inf
        if not -2 < lm < 0:
            return -np.inf
        # Linear
        if np.any((theta <= 0) + (theta > 2*self.ceiling)):
            return -np.inf
        # Logarithmic
        #if np.any((theta <= -11) + (theta > self.ceiling + 1)):
        #    return -np.inf

        # Compute Euclidean distance between bins, [∆i − ∆j]T Σ−1[∆i − ∆j]
        mini_inv_covar = np.array([[np.exp(la)**-1, 0], [0, np.exp(lm)**-1]])
        X              = np.matmul(self.bin_centers, mini_inv_covar)
        distance       = scipy.spatial.distance.cdist(X, X, 'sqeuclidean')

        K        = np.exp(l0)*np.exp(-0.5*distance)
        s, logdK = np.linalg.slogdet(K)

        y  = theta - mu
        F  = cho_factor(K)
        lp = -0.5*(logdK + np.dot(y, cho_solve(F, y)))

        if not np.isfinite(lp):
            return -np.inf
        return lp

    def lnpost(self, theta):
        return self.lnlike(theta)

    def gppost(self, theta_gp):
        return self.lnlike(theta_gp[:-4]) + self.gpprior(theta_gp[:-4],
                                                         theta_gp[-4],
                                                         theta_gp[-3],
                                                         theta_gp[-2],
                                                         theta_gp[-1])

    def sample(self, gp=False, parallel=False, save=True):
        nwalkers = 4*self.nbins
        ndim = self.nbins
        pos = np.array([np.abs(self.mlvalues[:, 0] + 0.01*np.random.randn(ndim)) \
                                                 for i in np.arange(nwalkers)]) + 0.01
        if gp:
            ndim += 4
            mu0 = np.mean(self.mlvalues[:, 0])
            l00 = 0
            la0 = 0
            lm0 = 0
            pos = np.append(pos, np.array([mu0, l00, la0, lm0]) + \
                                 np.array([0.05*np.random.randn(4) \
                                 for i in np.arange(nwalkers)]), axis=1)

        if parallel:
            with Pool(8) as pool:
                if gp:
                    self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.gppost, pool=pool)
                else:
                    self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnpost, pool=pool)
                self.sampler.run_mcmc(pos, 1000, progress=True)
                self.chains = self.sampler.chain[:, 100:, :].reshape((-1, ndim))
        else:
            if gp:
                self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.gppost)
            else:
                self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnpost)
            self.sampler.run_mcmc(pos, 1000, progress=True)
            self.chains = self.sampler.chain[:, 100:, :].reshape((-1, ndim))

        if save:
            chaindb = pd.DataFrame()
            for n, binn in enumerate(self.bins):
                chaindb['gamma{}'.format(n)] = self.chains[:, n]
            if gp:
                chaindb['lmu)'] = self.chains[:, -4]
                chaindb['ll0)'] = self.chains[:, -3]
                chaindb['lla)'] = self.chains[:, -2]
                chaindb['llm)'] = self.chains[:, -1]
            chaindb.to_csv(self.chainname)

    def run(self):
        self.max_like()
        self.sample()
