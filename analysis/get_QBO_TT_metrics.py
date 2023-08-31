# Function to return QBO period using Transition Time method
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline 

def get_QBO_periods_amplitudes(u_zonal, N_smooth=5, points_per_month=1):
    """ Function that returns all QBO periods and amplitudes (mean and the covariance matrices)
    using Transition Time (TT) method described in Schenzinger et al 2017
    Inputs: u_zonal (np array) zonal mean zonal wind at given height level, recommended 10hPa (MiMA index 13) 
            N_smooth (int) number of timesteps over which smoothing occurs. Should be 5 months. If monthly data
                    is provided (e.g. observed radiosonde data from Freie Uni. Berlin) N=5. 
                    If daily data is provided (e.g. output from MiMA) N_smooth=5*30=600
    Outputs: period (np array) all periods occuring in the time series. Note this will be returned same units
                    as data monthly or daily
             amplitude (np array) all amplitudes in m/s occuring in the time series
             """
    # First we smooth with a 5 month binomial smoothing
    u_smoothed = np.convolve(u_zonal, np.ones(N_smooth), mode='same')/N_smooth

    new_QBO_cycle = []
    t = range(len(u_zonal))
    # Go through time series and save each transition
    for it in t[1:]:
        if (u_smoothed[it] >=0 and u_smoothed[it-1] <= 0):
            new_QBO_cycle.append(it)


    print("QBO transition times:", new_QBO_cycle)
    new_QBO_cycle = np.array(new_QBO_cycle)
    periods = new_QBO_cycle[1:] - new_QBO_cycle[:-1]
    periods = periods/30.
    print(f"Periods: {periods}")
    amplitudes = np.zeros(len(periods))
    for ic in range(len(periods)):
        start_ind = new_QBO_cycle[ic]
        end_ind = new_QBO_cycle[ic+1]
        amplitudes[ic] = 0.5 * (np.max(u_smoothed[start_ind:end_ind] ) - np.min(u_smoothed[start_ind:end_ind] ) )


    return periods, amplitudes

