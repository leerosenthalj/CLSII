
import numpy as np

import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import seaborn as sns
import corner

from radvel.utils import semi_major_axis

from cls2.occur import *
from cls2.io import log_edges2centers

def format_fn(tick_val, tick_pos):
    return str(10**tick_val)

def mass_dist(planets, outname='plots/mass_histogram_all.png'):
    fig, ax = plt.subplots()
    ax.hist(np.log10(planets.query('mass < 15')['mass']), 
            bins=10, density=False, histtype='step', range=(np.log10(0.008), np.log10(15)),
            lw=3, color='black', alpha=1, label='All Planets')

    matplotlib.rcParams.update({'font.size': 11})
    ax.set(xlabel=r'$M\mathrm{sin}i/M_J$', 
            ylabel='Number')
    matplotlib.rcParams.update({'font.size': 11})
    #ax.set_title(r'$M$sin$i$ > 0.1 $M_J$, a > 0.1 AU')
            
    ax.xaxis.set_major_formatter(FuncFormatter(format_fn))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))   
    matplotlib.rcParams.update({'font.size': 11})
    ax.legend(loc=2)

    fig.savefig(outname, dpi=500, bbox_inches='tight')

def mcmc_trend(sampler):
    fig, ax = plt.subplots()
    index = np.arange(len(sampler.chains))
    ax.scatter(index, sampler.chains[:, 0])
    ax.set_xlabel('Step')
    ax.set_ylabel(r'$C$')
    #ax.set_xlim([0, 20000])

    fig.show()


def power_law_fit(chains, outname='plots/broken_powerlaw_solo.pdf'):
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_xticks((0.1, 0.3, 1, 3, 10, 30))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_yscale('log')
    ax.set_yticks((1, 3, 10))
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax.set_xlabel(r'$a$ (AU)')
    ax.set_ylabel(r'$N_P$ / 100 stars / $\Delta$ln($a$) (30 - 6000 $\mathrm{M_{\oplus}}$)')

    ax.set_xlim([0.1, 30])
    ax.set_ylim([1, 15])

    axes = np.logspace(-1, np.log10(30), num=50)
    for i in np.arange(20):
        ax.plot(axes, (100/719)*occbroken(axes, chains.iloc[np.random.randint(0, len(chains))].values), alpha=0.05, color='green')
    ax.plot(axes, (100/719)*occbroken(axes, np.median(chains, axis=0)), color='black', lw=4)
    fig.savefig(outname, bbox_inches='tight')


def binned_hist(planets, allbins, hierarchical_one, outname='plots/hist_11x1_1014_fancy_mode.pdf', chainfile='occur_chains.csv'):
    matplotlib.rcParams.update({'font.size': 12})
    chains_db = pd.read_csv(chainfile)

    binwidth = allbins[0][0][1] - allbins[0][0][0]

    # Do simple counting.
    simple_counts = []
    for n in np.arange(hierarchical_one.nbins):
        a1 = np.exp(allbins[n][0][0])
        a2 = np.exp(allbins[n][0][1])
        npl = len(planets.query('mass >= 0.1 and axis >= @a1 and axis < @a2'))
        # npl = len(planets.query('mass >= 0.1 and insol >= @a1 and insol < @a2'))
        simple_counts.append(npl)    
    simple_counts = np.array(simple_counts)*(100/719)/binwidth

    a_chains = np.empty((0, len(chains_db)))
    for n in np.arange(hierarchical_one.nbins):
        a_chains = np.append(a_chains, np.array([chains_db['gamma{}'.format(n)]]), axis=0)
    a_chains *= hierarchical_one.bin_areas[0][0]*(100/719)/binwidth

    # Record modes & medians.
    a_medians = np.median(a_chains, axis=1)
    a_sqvars  = np.std(a_chains, axis=1)
    a_modes = []
    a_159 = []
    a_841 = []
    a_682 = []

    for n in np.arange(hierarchical_one.nbins):
        chains = np.array([chains_db['gamma{}'.format(n)]])*hierarchical_one.bin_areas[0][0]*(100/719)
        hist, bin_edges = np.histogram(chains, bins=40, range=(np.percentile(chains, 2), np.percentile(chains, 95)))
        a_modes.append(bin_edges[np.argmax(hist)])
        a_159.append(np.percentile(chains, 15.9))
        a_841.append(np.percentile(chains, 84.1))
        a_682.append(np.percentile(chains, 68.2))
    a_modes = np.array(a_modes)/binwidth
    a_159 = np.array(a_159)/binwidth
    a_841 = np.array(a_841)/binwidth
    a_682 = np.array(a_682)/binwidth

    #pdb.set_trace()

    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_xticks((0.1, 0.3, 1, 3, 10, 30))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax.set_xlabel(r'Semi-major axis (au)')
    #ax.set_ylabel(r'Planets per 100 stars (30 - 6000 $\mathrm{M_{\oplus}}$)')
    ax.set_ylabel(r'$N_P$ / 100 stars / $\Delta$ln($a$) (30 - 6000 $\mathrm{M_{\oplus}}$)')

    ax.set_xlim([np.exp(hierarchical_one.lna_edges[0]), np.exp(hierarchical_one.lna_edges[-1])])
    #ax.set_ylim([0, 10])
    ax.set_ylim([0, 15])
    lnaw = hierarchical_one.lna_edges[1] - hierarchical_one.lna_edges[0]

    # Plot just-counting, no-completeness histogram.
    ax.step(np.insert(np.exp(hierarchical_one.lna_edges[:-1] + lnaw), 0, np.exp(hierarchical_one.lna_edges[0])), 
            np.insert(simple_counts, 0, simple_counts[0]), lw=2, c='blue', linestyle=':', label='Count')

    # Plot mode & 68.2% CI.
    ax.scatter(np.exp(hierarchical_one.lna_edges[:-1] + 0.5*lnaw), a_modes,
            color='black', s=30, label='_nolegend_')#label='Occurrence mode & CI')
    ax.vlines(np.exp(hierarchical_one.lna_edges[:-2] + 0.5*lnaw), a_159[:-1],
            a_841[:-1], alpha=0.5, color='black', lw=3, label='68.2%')
    # Show CI from 0 to 68.2 for the last bin.
    ax.vlines(np.exp(hierarchical_one.lna_edges[-2] + 0.5*lnaw), 0,
            a_682[-1], alpha=0.5, color='black', lw=3, label='_nolegend_')

    # Plot occurrence histogram.
    ax.step(np.insert(np.exp(hierarchical_one.lna_edges[:-1] + lnaw), 0, np.exp(hierarchical_one.lna_edges[0])), 
            np.insert(a_modes, 0, a_modes[0]), color='black', lw=2, label='Occurrence')

    for i in np.arange(1000):
        ax.step(np.insert(np.exp(hierarchical_one.lna_edges[:-1] + lnaw), 0, np.exp(hierarchical_one.lna_edges[0])), 
                np.insert(a_chains[:, np.random.randint(0, 10000)], 0, a_medians[0]), 
                color='black', lw=1, alpha=0.01, label='_nolegend_')
    
    ax.legend(loc=2) 

    fig.savefig(outname, dpi=1000, bbox_inches='tight')

def last_bin(hierarchical_one, chains_file, outname='plots/last_bin_and_prior.png'):
    fig, ax = plt.subplots()
    chains_db = pd.read_csv(chains_file)

    factor = hierarchical_one.bin_areas[0][0]*(100/719)
    prior_grid = np.linspace(0, 4*hierarchical_one.ceiling*factor, num=50)

    ax.plot(prior_grid, (hierarchical_one.Qints[-1]) * \
                np.exp(-hierarchical_one.Qints[-1]*prior_grid), lw=3, label='Prior')
    ax.hist(factor*chains_db['gamma10'][1000:], bins=20, density=True, fill=False, 
            histtype='step', color='black', lw=3, label='Posterior')

    ax.set_xlim([0, 20])
    ax.set_ylim([0, 0.3])
    ax.set_xlabel('Last bin value')
    ax.set_ylabel('N')

    ax.legend()
    fig.savefig(outname, dpi=1000, bbox_inches='tight')


def fit_overlay(planets, hierarchical_one, broken, allbins, chains_file='occur_chains.csv', 
                outname='plots/hist_11x2_1026_fancy_mode.pdf', show_others=False):
    matplotlib.rcParams.update({'font.size': 12})
    chains_db = pd.read_csv(chains_file)

    binwidth = allbins[0][0][1] - allbins[0][0][0]
    nbins = len(hierarchical_one.lna_edges) - 1

    # Do simple counting.
    simple_counts = []
    for n in np.arange(nbins):
        a1 = np.exp(allbins[n][0][0])
        a2 = np.exp(allbins[n][0][1])
        npl = len(planets.query('mass >= 0.1 and axis >= @a1 and axis < @a2'))
        simple_counts.append(npl)    
    simple_counts = np.array(simple_counts)*(100/719)/binwidth

    a_chains = np.empty((0, len(chains_db)))
    for n in np.arange(nbins):
        a_chains = np.append(a_chains, np.array([chains_db['gamma{}'.format(n)]]), axis=0)
    a_chains *= hierarchical_one.bin_areas[0][0]*(100/719)/binwidth

    # Record modes & medians.
    a_medians = np.median(a_chains, axis=1)
    a_sqvars  = np.std(a_chains, axis=1)
    a_modes = []
    a_159 = []
    a_841 = []
    a_682 = []

    for n in np.arange(nbins):
        chains = np.array([chains_db['gamma{}'.format(n)]])*(100/719)*hierarchical_one.bin_areas[0][0]
        hist, bin_edges = np.histogram(chains, bins=40, range=(np.percentile(chains, 2), np.percentile(chains, 95)))
        a_modes.append(bin_edges[np.argmax(hist)])
        a_159.append(np.percentile(chains, 15.9))
        a_841.append(np.percentile(chains, 84.1))
        a_682.append(np.percentile(chains, 68.2))
    a_modes = np.array(a_modes)/binwidth
    a_159 = np.array(a_159)/binwidth
    a_841 = np.array(a_841)/binwidth
    a_682 = np.array(a_682)/binwidth

    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_xticks((0.1, 0.3, 1, 3, 10, 30))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax.set_xlabel(r'Semi-major axis (au)')
    #ax.set_ylabel(r'Planets per 100 stars (30 - 6000 $\mathrm{M_{\oplus}}$)')
    ax.set_ylabel(r'$N_P$ / 100 stars / $\Delta$ln($a$) (30 - 6000 $\mathrm{M_{\oplus}}$)')

    #ax.set_xlim([np.exp(hierarchical_one.lna_edges[0]), np.exp(hierarchical_one.lna_edges[-1])])
    ax.set_xlim([0.1, np.exp(hierarchical_one.lna_edges[-1])])
    #ax.set_ylim([0, 10])
    ax.set_ylim([0, 15])
    lnaw = hierarchical_one.lna_edges[1] - hierarchical_one.lna_edges[0]

    # Plot just-counting, no-completeness histogram.
    #ax.step(np.insert(np.exp(hierarchical_one.lna_edges[:-1] + lnaw), 0, np.exp(hierarchical_one.lna_edges[0])), 
    #        np.insert(simple_counts, 0, simple_counts[0]), lw=2, c='blue', linestyle=':', label='Count')

    # Plot mode & 68.2% CI.
    ax.scatter(np.exp(hierarchical_one.lna_edges[:-1] + 0.5*lnaw), a_modes,
            color='black', s=30, label='_nolegend_')
    
    ax.vlines(np.exp(hierarchical_one.lna_edges[:-2] + 0.5*lnaw), a_159[:-1],
            a_841[:-1], alpha=0.5, color='black', lw=3, label='_nolegend_')#label='68.2%')
    # Show CI from 0 to 68.2 for the last bin.
    ax.vlines(np.exp(hierarchical_one.lna_edges[-2] + 0.5*lnaw), 0,
            a_682[-1], alpha=0.5, color='black', lw=3, label='_nolegend_')

    # Plot occurrence histogram.
    ax.step(np.insert(np.exp(hierarchical_one.lna_edges[:-1] + lnaw), 0, np.exp(hierarchical_one.lna_edges[0])), 
            np.insert(a_modes, 0, a_modes[0]), color='black', lw=2)

    # Broken power law comparison.
    #axes = np.logspace(-1, np.log10(30), num=50)
    axes = np.logspace(np.log10(np.exp(hierarchical_one.lna_edges[0])), np.log10(30), num=1000)
    for i in np.arange(300):
        # ax.step(np.insert(np.exp(hierarchical_one.lna_edges[:-1] + lnaw), 0, np.exp(hierarchical_one.lna_edges[0])), 
        #     np.insert(a_chains[:, np.random.randint(0, 10000)], 0, a_medians[0]), 
        #     color='black', lw=1, alpha=0.01, label='_nolegend_')
        ax.plot(axes, (100/719)*occbroken(axes, broken.chains[np.random.randint(0, 10000), :]), 
            alpha=0.04, color='green', label='_nolegend_')
    
    ax.plot(axes, (100/719)*occbroken(axes, np.median(broken.chains, axis=0)), 
            color='black', lw=2, ls='--', label='Broken power law')

    if show_others:
        a = -0.31
        b = 0.26
        mass_edges = [30, 6000]
        bin_areas = hierarchical_one.bin_areas[0][0] / 2
        mm = 10**np.mean([np.log10(mass_edges[0]), np.log10(mass_edges[1])])
        predict = 20*(axes**(3/2.))**b *  mm**a * bin_areas # eye-balled constant in front

        # Read in Fernandes data
        fd = pd.read_csv('legacy_tables/fernandes_period.txt')
        fd['axis'] = semi_major_axis(fd['period'].values, 1.0)

        plt.plot(axes, predict, lw=2, linestyle='dashed', label='Cumming et al. (2008)', zorder=11)
        plt.errorbar(fd['axis'], fd['rate']*100* bin_areas, yerr=fd['rate_err']*100, elinewidth=0.5,
                    fmt='o-', linestyle='dashed', lw=2, ms=8, zorder=10, mfc='none', mew=2,
                    label="Fernandes et al. (2019)")

        # Read in Wittenmyer 2020 data
        wn = pd.read_csv('legacy_tables/whttenmyer_20.csv')
        wt = wn.iloc[:-1]
        wt['axis'] = log_edges2centers(semi_major_axis(wn['per_bin_start'].values, 1.0))
        plt.errorbar(wt['axis'], wt['rate']* bin_areas, yerr=[wt['err_low'], wt['err_high']], elinewidth=0.5,
                fmt='o-', linestyle='dashed', lw=2, ms=8, zorder=10, mfc='none', mew=2, color='purple',
                label="Wittenmyer et al. (2020)")
        
        plt.ylim(0, 16)
        
        plt.legend()



    ax.legend(loc=2) 

    fig.savefig(outname, dpi=1000, bbox_inches='tight')


def survey_summary(planets, outname='plots/all_contours.pdf'):
    recoveries_planets = pd.read_csv('recoveries_planets_earth.csv')
    completey_planets  = Completeness(recoveries_planets)
    completey_planets.completeness_grid([0.01, 40], [2, 9000])

    recoveries_all = pd.read_csv('recoveries_all_earth.csv')
    completey_all  = Completeness(recoveries_all)
    completey_all.completeness_grid([0.01, 40], [2, 9000])

    planets_old = planets
#     planets_old = planets.query('status == "K"').reset_index(drop=True)
#     planets_new = planets.query('status == "C" or status == "J"').reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks((0.1, 1, 10))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())

    CS = ax.contourf(completey_all.grid[0], completey_all.grid[1],
                     completey_all.grid[2], 10, cmap=plt.cm.gray)

    matplotlib.rcParams.update({'font.size': 18})
    # ax.set_title('All Star Contours')
    ax.scatter(planets_old.axis, 317.8*planets_old.mass, c='b', s=50, alpha=0.75, 
            label='{} planets'.format(len(planets_old)))
#     ax.scatter(planets_new.axis, 317.8*planets_new.mass, s=400, c='g', alpha=0.75, 
#             label='{} New'.format(len(planets_new)))

    ax.set_xlim([2.5*10**-2, 30])
    ax.set_ylim([2.3, 9000])

    ymin, ymax = ax.get_ylim()

    matplotlib.rcParams.update({'font.size': 16})
    ax.set_xlabel('Semi-major axis (au)')
    ax.set_ylabel(r'$M$sin$i$ ($M_{\oplus}$)')
    ax.legend(loc=4)

    plt.colorbar(mappable=CS, pad=0, label='Completeness')

    fig.savefig(outname, dpi=500, bbox_inches='tight')


def insol_hist(planets, allbins, hierarchical_one,
               outname='plots/hist_insol.pdf',
               chainfile='insol_chains.csv'):
    matplotlib.rcParams.update({'font.size': 12})
    chains_db = pd.read_csv(chainfile)

    binwidth = allbins[0][0][1] - allbins[0][0][0]

    # Do simple counting.
    simple_counts = []
    for n in np.arange(hierarchical_one.nbins):
        a1 = np.exp(allbins[n][0][0])
        a2 = np.exp(allbins[n][0][1])
        npl = len(planets.query('mass >= 0.1 and insol >= @a1 and insol < @a2'))
        simple_counts.append(npl)    
    simple_counts = np.array(simple_counts)*(100/719)/binwidth

    a_chains = np.empty((0, len(chains_db)))
    for n in np.arange(hierarchical_one.nbins):
        a_chains = np.append(a_chains, np.array([chains_db['gamma{}'.format(n)]]), axis=0)
    a_chains *= hierarchical_one.bin_areas[0][0]*(100/719)/binwidth

    # Record modes & medians.
    a_medians = np.median(a_chains, axis=1)
    a_sqvars  = np.std(a_chains, axis=1)
    a_modes = []
    a_159 = []
    a_841 = []
    a_682 = []

    for n in np.arange(hierarchical_one.nbins):
        chains = np.array([chains_db['gamma{}'.format(n)]])*hierarchical_one.bin_areas[0][0]*(100/719)
        hist, bin_edges = np.histogram(chains, bins=40, range=(np.percentile(chains, 2), np.percentile(chains, 95)))
        a_modes.append(bin_edges[np.argmax(hist)])
        a_159.append(np.percentile(chains, 15.9))
        a_841.append(np.percentile(chains, 84.1))
        a_682.append(np.percentile(chains, 68.2))
    a_modes = np.array(a_modes)/binwidth
    a_159 = np.array(a_159)/binwidth
    a_841 = np.array(a_841)/binwidth
    a_682 = np.array(a_682)/binwidth

    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_xticks((0.0001, 0.001, 0.01, 0.1,  1, 10,  100, 1000))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.FormatStrFormatter('%s'))
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())

    ax.set_xlabel(r'Stellar light intensity realtive to Earth')
    #ax.set_ylabel(r'Planets per 100 stars (30 - 6000 $\mathrm{M_{\oplus}}$)')
    ax.set_ylabel(r'$N_P$ / 100 stars / $\Delta$ln($S$) (30 - 6000 $\mathrm{M_{\oplus}}$)')

    ax.set_xlim([np.exp(hierarchical_one.lna_edges[-1]), np.exp(hierarchical_one.lna_edges[0])])
    ax.set_ylim([0, 7])
    lnaw = hierarchical_one.lna_edges[1] - hierarchical_one.lna_edges[0]

    # Plot just-counting, no-completeness histogram.
    ax.step(np.insert(np.exp(hierarchical_one.lna_edges[:-1] + lnaw), 0, np.exp(hierarchical_one.lna_edges[0])), 
            np.insert(simple_counts, 0, simple_counts[0]), lw=2, c='blue', linestyle=':', label='Count')

    # Plot mode & 68.2% CI.
    ax.scatter(np.exp(hierarchical_one.lna_edges[:-1] + 0.5*lnaw), a_modes,
            color='black', s=30, label='_nolegend_')#label='Occurrence mode & CI')
    ax.vlines(np.exp(hierarchical_one.lna_edges[:-1] + 0.5*lnaw), a_159,
            a_841, alpha=0.5, color='black', lw=3, label='68.2%')
#     # Show CI from 0 to 68.2 for the last bin.
#     ax.vlines(np.exp(hierarchical_one.lna_edges[-2] + 0.5*lnaw), 0,
#             a_682[-1], alpha=0.5, color='black', lw=3, label='_nolegend_')

    # Plot occurrence histogram.
    ax.step(np.insert(np.exp(hierarchical_one.lna_edges[:-1] + lnaw), 0, np.exp(hierarchical_one.lna_edges[0])), 
            np.insert(a_modes, 0, a_modes[0]), color='black', lw=2, label='Occurrence')

    for i in np.arange(1000):
        ax.step(np.insert(np.exp(hierarchical_one.lna_edges[:-1] + lnaw), 0, np.exp(hierarchical_one.lna_edges[0])), 
                np.insert(a_chains[:, np.random.randint(0, 10000)], 0, a_medians[0]), 
                color='black', lw=1, alpha=0.01, label='_nolegend_')
    
    ax.legend(loc=2) 

    fig.savefig(outname, dpi=1000, bbox_inches='tight')


def subjovians(hierarchical_sub, hierarchical_sup, subjupbins, supjupbins,
               outname='plots/hist_super_sub_Jupiters.pdf'):
    chains_sub = pd.read_csv('occur_chains_11x1_sub.csv')
    chains_sup = pd.read_csv('occur_chains_11x1_sup.csv')
    a_modes_sub = []
    a_159_sub = []
    a_841_sub = []
    for n in np.arange(hierarchical_sub.nbins):
        chains = np.array([chains_sub['gamma{}'.format(n)]])*hierarchical_sub.bin_areas[0][0]*(100/719)
        hist, bin_edges = np.histogram(chains, bins=40, range=(np.percentile(chains, 2), np.percentile(chains, 95)))
        a_modes_sub.append(bin_edges[np.argmax(hist)])
        a_159_sub.append(np.percentile(chains, 15.9))
        a_841_sub.append(np.percentile(chains, 84.1))
    a_modes_sub = np.array(a_modes_sub)
    a_159_sub = np.array(a_159_sub)
    a_841_sub = np.array(a_841_sub)
    a_modes_sup = []
    a_159_sup = []
    a_841_sup = []
    for n in np.arange(hierarchical_sup.nbins):
        chains = np.array([chains_sup['gamma{}'.format(n)]])*hierarchical_sup.bin_areas[0][0]*(100/719)
        hist, bin_edges = np.histogram(chains, bins=40, range=(np.percentile(chains, 2), np.percentile(chains, 95)))
        a_modes_sup.append(bin_edges[np.argmax(hist)])
        a_159_sup.append(np.percentile(chains, 15.9))
        a_841_sup.append(np.percentile(chains, 84.1))
    a_modes_sup = np.array(a_modes_sup)
    a_159_sup = np.array(a_159_sup)
    a_841_sup = np.array(a_841_sup)
    fig, ax = plt.subplots()
    ax.set_xscale('log')
    ax.set_xticks((0.1, 0.3, 1, 3, 10, 30))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r'Semi-major axis (au)')
    ax.set_ylabel('$N_P$ / 100 stars / $\Delta$ln($a$)\n(30 - 6000 $\mathrm{M_{\oplus}}$)')
    ax.set_xlim([np.exp(hierarchical_sup.lna_edges[0]), np.exp(hierarchical_sup.lna_edges[-1])])
    ax.set_ylim([0, 8.25])
    lnaw_sub = subjupbins[0][0][1] - subjupbins[0][0][0]
    lnaw_sup = supjupbins[0][0][1] - supjupbins[0][0][0]
    ax.scatter(np.exp(hierarchical_sub.lna_edges[:-1] + 0.5*lnaw_sub), a_modes_sub,
            color='blue', s=30, label='_nolegend_')
    ax.vlines(np.exp(hierarchical_sub.lna_edges[:-1] + 0.5*lnaw_sub), a_159_sub,
            a_841_sub, alpha=0.5, color='blue', lw=3, label='_nolegend_')
    ax.step(np.insert(np.exp(hierarchical_sub.lna_edges[:-1] + lnaw_sub), 0, np.exp(hierarchical_sub.lna_edges[0])), 
            np.insert(a_modes_sub, 0, a_modes_sub[0]), color='blue', lw=2, linestyle=':', label=r'30 - 300 M$_{\oplus}$')
    ax.scatter(np.exp(hierarchical_sup.lna_edges[:-1] + 0.5*lnaw_sup), a_modes_sup,
            color='green', s=30, label='_nolegend_')
    ax.vlines(np.exp(hierarchical_sup.lna_edges[:-1] + 0.5*lnaw_sup), a_159_sup,
            a_841_sup, alpha=0.5, color='green', lw=3, label='_nolegend_')
    ax.step(np.insert(np.exp(hierarchical_sup.lna_edges[:-1] + lnaw_sup), 0, np.exp(hierarchical_sup.lna_edges[0])), 
            np.insert(a_modes_sup, 0, a_modes_sup[0]), color='green', lw=2, linestyle='--', label=r'300 - 6000 M$_{\oplus}$')
    ax.legend(loc=2) 
    # fig.savefig('plots/hist_super_sub_Jupiters.png', dpi=1000, bbox_inches='tight')
    fig.savefig(outname, bbox_inches='tight')