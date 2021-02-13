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


def save_completeness(listf, outf, stellarf, recdir):
    """stack injections on cadence, with msini & a"""

    names = pd.read_csv(listf)
    stars = pd.read_csv(stellarf)
    
    all_recoveries = pd.DataFrame()
    
    for name in names['name']:
        recoveries = pd.read_csv(recdir + '/' + name + '/recoveries.csv')
        mstar = float(stars.query('name == @name')['mass_c'])
        rstar = float(stars.query('name == @name')['radius_c'])
        teff  = float(stars.query('name == @name')['teff_c'])
        
        recoveries['inj_msini'] = radvel.utils.Msini(recoveries['inj_k'],
                                                     recoveries['inj_period'],
                                                     mstar, recoveries['inj_e'],
                                                     Msini_units='jupiter')
        recoveries['rec_msini'] = radvel.utils.Msini(recoveries['rec_k'],
                                                     recoveries['rec_period'],
                                                     mstar, recoveries['rec_e'],
                                                     Msini_units='jupiter')
        recoveries['inj_au'] = radvel.utils.semi_major_axis(recoveries['inj_period'], mstar)
        recoveries['rec_au'] = radvel.utils.semi_major_axis(recoveries['rec_period'], mstar)   
        recoveries['inj_insol'] = rvsearch.utils.insolate(teff, rstar, recoveries['inj_au'])
        recoveries['rec_insol'] = rvsearch.utils.insolate(teff, rstar, recoveries['rec_au']) 
        
        all_recoveries = all_recoveries.append(recoveries).reset_index(drop=True)
        
    all_recoveries.to_csv(outf)

# Sketch DFM occurrence likelihood, first with broken power law, then nonparametric.
def occbroken(a, theta):
    return theta[0]*(a**theta[1])*(1 - np.exp(-(a/theta[2])**theta[3]))


def nll(theta, x, y, yerr):
    return 0.5*np.sum((y-occbroken(x, theta))**2*yerr**-2 + np.log(2*np.pi*yerr**2))


def fit_broken(x, y, yerr, C_init=80, beta_init=-0.2, a0_init=0.8, gamma_init=2): 
    fit = op.minimize(nll, [C_init, beta_init, a0_init, gamma_init], args=(x, y, yerr),
                      method='Powell', options={'xtol': 1e-8, 'disp': True})
    return fit.x


def lngrid(min_a, max_a, min_M, max_M, resa, resm):
    lna1 = np.log(min_a)
    lna2 = np.log(max_a)
    lnM1 = np.log(min_M)
    lnM2 = np.log(max_M)
    
    dlna = (lna2 - lna1)/resa
    dlnM = (lnM2 - lnM1)/resm

    bins = []
    for i in np.arange(int(resa)):
        for j in np.arange(int(resm)):
            bins.append([[lna1 + i*dlna, lna1 + (i+1)*dlna], 
                         [lnM1 + j*dlnM, lnM1 + (j+1)*dlnM]])
            
    return np.array(bins)

class Completeness(object):
    """Object to handle a suite of injection/recovery tests
       Class for evaluating a completeness grid across injections and recoveries.
       Clone of BJ's code in RVSearch, except passing in msini and axis as inputs,
       rather than computing with a single stellar mass.

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
                                                        bounds_error=False, fill_value=0.001) # Maybe don't set fill
        
        return self.interpolator(np.array([np.atleast_1d(x), np.atleast_1d(y)]).T)


class Hierarchy(object):
    """Do hierarchical Bayesian sampling of occurrence posteriors, based on DFM et al. 2014.
    Args:
        pop (pandas DataFrame): dataframe of planet parameter chains

    """
    def __init__(self, pop, completeness, res=4, bins=np.array([[[np.log(0.02), np.log(20)], 
                                                                 [np.log(2.), np.log(6000)]]]),
                                                                  chainname='occur_chains.csv'):
        # TO-DO: Replace single-param planets with paths to posteriors.
        self.pop          = pop # Replace pairs of m & a with chains
        self.completeness = completeness # Completeness grid, defined as class object below.
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
            ml  = nplanets/self.Qints[n]
            uml = ml/np.sqrt(nplanets)
            if not np.isfinite(ml):
                ml = 0.01 
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
        
        occur = theta[ia + im*self.nabins]
        # Return filler value for samples outside of the bin limits.
        occur[iao < 0] = 0.01 
        occur[imo < 0] = 0.01 
        occur[iao > self.nabins - 1] = 0.01 
        occur[imo > self.nmbins - 1] = 0.01 
        return occur    

    def lnlike(self, theta): 
        if np.any((theta <= 0) + (theta > 20*self.ceiling)):
            return -np.inf
        sums = []
        for planet in self.planetnames:
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
        if not 0 < mu < 4*self.ceiling:
            return -np.inf
        if not -2 < l0 < 6:
            return -np.inf
        if not -2 < la < 1:
            return -np.inf
        if not -2 < lm < 0:
            return -np.inf
        if np.any((theta <= 0) + (theta > 4*self.ceiling)): 
            return -np.inf

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


class BrokenPowerLaw(object):
    """Do hierarchical Bayesian sampling of occurrence posteriors, based on DFM et al. 2014.
    Args:
        pop (pandas DataFrame): dataframe of planet parameter chains

    """
    def __init__(self, pop, completeness, lna_res=50, edges=np.array([[0.1, 30], [30, 6000]]), 
                 chainname='powerlaw_chains.csv'):
        # TO-DO: Replace single-param planets with paths to posteriors.
        self.pop          = pop 
        self.completeness = completeness 
        self.completeness.completeness_grid([0.01, 40], [3, 7000])
        # Fill in completeness nans.
        self.completeness.grid[2][np.isnan(self.completeness.grid[2])] = 1.  
                
        axis  = []
        msini = []
        self.planetnames = np.unique([x[:-2] + x[-1] for x in pop.columns]) 
        self.starnames   = np.unique([x[:-1] for x in self.planetnames])
        self.nplanets    = len(self.planetnames)
        self.nsamples    = len(self.pop)
        self.nstars      = len(self.starnames)
        
        medians = pop.median() 
        for name in self.planetnames:
            axis.append(medians[[name[:-1] + 'a' + name[-1]]][0])
            msini.append(medians[[name[:-1] + 'M' + name[-1]]][0])            
        self.pop_med = pd.DataFrame.from_dict({'axis':axis, 'msini':msini})
        
        self.chainname = chainname
        
        # Pre-compute integrated completeness over lna_res-many bins.
        self.lna_res = int(lna_res)
        self.edges = edges
        self.Qints = np.zeros(self.lna_res)
        self.lnawidth = (np.log(self.edges[0][1]) - np.log(self.edges[0][0]))/self.lna_res
        self.lnmwidth = (np.log(self.edges[1][1]) - np.log(self.edges[1][0]))/5.
        
        self.lna_centers = np.zeros(self.lna_res)
        self.lnm_centers = np.zeros(10)
        for i in np.arange(self.lna_res): 
            self.lna_centers[i] = np.log(self.edges[0][0]) + (i + 0.5)*self.lnawidth
        for j in np.arange(10): 
            self.lnm_centers[j] = np.log(self.edges[1][0]) + (j + 0.5)*self.lnmwidth 
        
        for i in np.arange(self.lna_res): 
            for j in np.arange(10): 
                self.Qints[i] += (self.lnawidth*self.lnmwidth) * \
                                  self.completeness.interpolate(np.exp(self.lna_centers[i]), 
                                                                np.exp(self.lnm_centers[j]))
        self.Qints /= (np.log(self.edges[1][1]) - np.log(self.edges[1][0])) # Necessary? Maybe can streamline DaDm.
        
    def max_like(self):
        # FIGURE THIS OUT? OR LEAVE AS DECENT GUESS.
        C_0 = 80
        beta_0 = -0.2
        a0_0 = 0.8
        gamma_0 = 2. 
        self.mlvalues = np.array([C_0, beta_0, a0_0, gamma_0])
        
    def occurrence(self, axis, theta):
        broken = np.atleast_1d(theta[0]*axis**theta[1]*(1 - np.exp(-(axis/theta[2])**theta[3])))
        broken[axis < 0.1] = 0.01
        broken[axis > 30.] = 0.01
        return broken 

    def lnlike(self, theta): 
        sums = []
        for planet in self.planetnames:
            probs = []
            sample_a = np.array(self.pop[planet[:-2] + '_a' + planet[-1]])
            sample_M = np.array(self.pop[planet[:-2] + '_M' + planet[-1]])
            probs = self.completeness.interpolate(sample_a, sample_M) * \
                                  self.occurrence(sample_a, theta)
            sums.append(np.sum(probs)) 
        
        nexpect = 0
        for i in np.arange(self.lna_res):
            nexpect += self.Qints[i]*self.occurrence(np.exp(self.lna_centers[i]), theta)  
             
        ll = -nexpect + np.sum(np.log(np.array(sums)/self.nsamples))
        if not np.isfinite(ll):
            return -np.inf
        return ll
    
    def lnprior(self, theta):
        if theta[0] <= 0 or theta[0] > 1600 or theta[1] > 3 or theta[1] < -6 or \
           theta[2] < 0.1 or theta[2] > 12 or theta[3] > 8 or theta[3] <= 0.1:
            return -np.inf
        return 0
    
    def lnpost(self, theta):
        return self.lnlike(theta) + self.lnprior(theta)
    
    def sample(self, parallel=False, save=True, nsamples=1000):
        nwalkers = 30
        ndim = 4
        nburn = 100
        if nsamples <= nburn:
            nburn = int(np.round(0.1*nsamples))
        pos = np.array([np.abs(self.mlvalues + 0.01*np.random.randn(ndim)) \
                                           for i in np.arange(nwalkers)])
        if parallel:
            with Pool(8) as pool:
                self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnpost, pool=pool)
                self.sampler.run_mcmc(pos, nsamples, progress=True)
                self.chains = self.sampler.chain[:, nburn:, :].reshape((-1, ndim))
        else:
            self.sampler = emcee.EnsembleSampler(nwalkers, ndim, self.lnpost)
            self.sampler.run_mcmc(pos, nsamples, progress=True)           
            self.chains = self.sampler.chain[:, nburn:, :].reshape((-1, ndim))    
            
        if save:
            chaindb = pd.DataFrame() 
            chaindb['C'] = self.chains[:, 0]
            chaindb['beta'] = self.chains[:, 1]
            chaindb['a0'] = self.chains[:, 2]
            chaindb['gamma'] = self.chains[:, 3]
            chaindb.to_csv(self.chainname)           
    
    def run(self):
        self.max_like()
        self.sample()


class Insol(object):
    """Do hierarchical Bayesian sampling of occurrence posteriors, based on DFM et al. 2014.
    Args:
        pop (pandas DataFrame): dataframe of planet parameter chains

    """
    def __init__(self, pop, completeness, res=4, bins=np.array([[[np.log(0.1), np.log(1000)], 
                                                                 [np.log(2.), np.log(6000)]]]),
                                                                  chainname='insol_chains.csv'):
        # TO-DO: Replace single-param planets with paths to posteriors.
        self.pop          = pop # Replace pairs of m & a with chains
        self.completeness = completeness # Completeness grid, defined as class object below.
        self.completeness.completeness_grid([1e-5, 2000], [10, 7000])
        # Fill in completeness nans.
        self.completeness.grid[2][np.isnan(self.completeness.grid[2])] = 1. #0.99
        
        self.res = res # Resolution for logarithmic completeness integration.
        self.bins = bins # Logarithmic bins in msini/axis space.
        self.nbins = len(self.bins)
        self.lna_edges = np.unique(self.bins[:, 0])
        self.lnm_edges = np.unique(self.bins[:, 1])
        self.nabins = len(self.lna_edges) - 1
        self.nmbins = len(self.lnm_edges) - 1
        
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
            axis.append(medians[[name[:-1] + 'S' + name[-1]]][0])
            msini.append(medians[[name[:-1] + 'M' + name[-1]]][0])            
        self.pop_med = pd.DataFrame.from_dict({'insol':axis, 'msini':msini})
        
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
            planets = self.pop_med.query('insol >= @a1 and insol < @a2 and \
                                         msini >= @M1 and msini < @M2')
            nplanets = len(planets)
            print(a1, a2, M1, M2, nplanets)
            ml  = nplanets/self.Qints[n]
            uml = ml/np.sqrt(nplanets)
            if not np.isfinite(ml):
                ml = 0.01 
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
        
        occur = theta[ia + im*self.nabins]
        # Return filler value for samples outside of the bin limits.
        occur[iao < 0] = 0.01 
        occur[imo < 0] = 0.01 
        occur[iao > self.nabins - 1] = 0.01 
        occur[imo > self.nmbins - 1] = 0.01 
        return occur    

    def lnlike(self, theta): 
        if np.any((theta <= 0) + (theta > 20*self.ceiling)):
            return -np.inf
        sums = []
        for planet in self.planetnames:
            probs = []
            sample_a = np.array(self.pop[planet[:-2] + '_S' + planet[-1]])
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
        if not 0 < mu < 4*self.ceiling:
            return -np.inf
        if not -2 < l0 < 6:
            return -np.inf
        if not -2 < la < 1:
            return -np.inf
        if not -2 < lm < 0:
            return -np.inf
        if np.any((theta <= 0) + (theta > 4*self.ceiling)): 
            return -np.inf

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
            with Pool(4) as pool:
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
