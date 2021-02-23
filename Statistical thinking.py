import numpy as np
import pandas as pd
from pandas import Series
from pandas import DataFrame
import pandas_datareader as pdr
from matplotlib import pyplot as plt
import math
import datetime
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.holiday import Holiday, USMemorialDay, AbstractHolidayCalendar, nearest_workday, MO
import seaborn as sns
from scipy.stats import norm, linregress
import statsmodels.formula.api as smf
#import empiricaldist
#import this
from sqlalchemy import create_engine


def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, len(x)+1) / n
    return x, y

###############################################
def bootstrap_replicate_1d(data, func):
    """Generate bootstrap replicate of 1D data.
    func can np.mean, np.sum ... """
    bs_sample = np.random.choice(data, len(data))
    return func(bs_sample)


################################################
def draw_bs_reps(data, func, size=1):
    """Draw bootstrap replicates (generate several bootstrap replicates)"""

    # Initialize array of replicates: bs_replicates
    bs_replicates = np.empty(size=size)

    # Generate replicates
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)

    return bs_replicates

###############################################
"""95% confidence interval"""
#conf_int = np.percentile(#data, [2.5,97.5])

##############################################

def draw_bs_pairs_linreg(x, y, size=1):
    """Perform pairs bootstrap for linear regression."""

    # Set up array of indices to sample from: inds
    inds = np.arange(0, len(x))

    # Initialize replicates: bs_slope_reps, bs_intercept_reps
    bs_slope_reps = np.empty(size)
    bs_intercept_reps = np.empty(size)

    # Generate replicates
    for i in range(size):
        bs_inds = np.random.choice(inds, size=len(inds))
        bs_x, bs_y = x[bs_inds], y[bs_inds]
        bs_slope_reps[i], bs_intercept_reps[i] = np.polyfit(bs_x, bs_y, 1)

    return bs_slope_reps, bs_intercept_reps

################################################################

def permutation_sample(data1, data2):
    """Generate a permutation sample from two data sets."""

    # Concatenate the data sets: data
    data = np.concatenate((data1, data2))

    # Permute the concatenated array: permuted_data
    permuted_data = np.random.permutation(data)

    # Split the permuted array into two: perm_sample_1, perm_sample_2
    perm_sample_1 = permuted_data[:len(data1)]
    perm_sample_2 = permuted_data[len(data1):]

    return perm_sample_1, perm_sample_2

############################################

def draw_perm_reps(data_1, data_2, func, size=1):
    """Generate multiple permutation replicates."""

    # Initialize array of replicates: perm_replicates
    perm_replicates = np.empty(size)

    for i in range(size):
        # Generate permutation sample
        perm_sample_1, perm_sample_2 = permutation_sample(data_1, data_2)

        # Compute the test statistic #the func in here is 'diff_of_means'
        perm_replicates[i] = func(perm_sample_1, perm_sample_2)

    return perm_replicates

#######################################################

def diff_of_means(data_1, data_2):
    """Difference in means of two arrays."""

    # The difference of means of data_1, data_2: diff
    diff = np.mean(data_1)-np.mean(data_2)

    return diff

#####################################################

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat=np.corrcoef(x,y)

    # Return entry [0,1]
    return corr_mat[0,1]


