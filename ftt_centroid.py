# -*- coding: utf-8 -*-
"""
Created on Wed Apr 01 11:16:10 2015

@author: TMM
"""

import matplotlib.pyplot as plt
import numpy as np
import plotRoutine
from scipy.stats import norm
from numpy import ma


"""
Instrument and source properties
"""
#magnitude = np.arange(10.0, 16.0)  # TODO
magnitude = 15.0
diameter = 3.5  # [m]
wavelength = 635  # [nm]
bandpass = 880 - 390  # [nm]
extinction = 0.5  # Extinction in magnitudes per airmass
transmission = 0.72
qe = 0.9
fiber_rejection = 0.16
integration_time = 0.01  # [seconds]


#n_photons = 10000

"""
Detector and fiber properties
"""
field = 10.0  # [arcsecs]
pixel_size = np.array([5.0, 2.5, 2.0, 1.0, 0.5])  # [arcsecs] 
fwhm = 0.76  # fwhm of beam [arcsecs]
fiber_radius = 0.45  # 1/e radius [arcsecs]
sigma = fwhm / 2.3458  # [arcsecs]
mux = 0.05  # [arcsecs]
muy = 0.05  # [arcsecs]
noise_level = 0


"""
Fiber mask, binning parameters
"""
n_bins = 500  # Grid to be binned into detector
bin_size = field / n_bins
mask = plotRoutine.fiber_mask(fiber_radius, n_bins, bin_size)
xlims = [-(n_bins/2.0 * bin_size), (n_bins/2.0 * bin_size)]
ylims = [-(n_bins/2.0 * bin_size), (n_bins/2.0 * bin_size)]


# Pixel edges on detector
xedges = np.linspace(xlims[0], xlims[1], n_bins+1)
yedges = np.linspace(xlims[0], xlims[1], n_bins+1)



"""
Realizations
"""
err = np.zeros((len(pixel_size), 2))
for m in range(len(pixel_size)):
    n_photons = plotRoutine.source_photons(bandpass, wavelength, magnitude, diameter, extinction) * qe * fiber_rejection * transmission * integration_time    
    
    n_pix = field / pixel_size[m]  # Detector pixels
    bins_per_pix = n_bins / n_pix
    xdet_edges = xedges[::bins_per_pix]  # [arcsecs]
    ydet_edges = yedges[::bins_per_pix]  # [arcsecs]
    err2 = np.zeros(1000)
    
    for n in range(1000):
        # Generate random photons
        # Dominate error in EMCCDs is multiplicative error in gain stage
        # Gain factor adds sqrt(2) to the shot noise
        # Accomodate by multiplying n_photons by sqrt(2)
        gain_noise_photons = n_photons + norm.rvs(loc=0.0, scale=np.sqrt(2), size=1)
        x = norm.rvs(loc=mux, scale=sigma, size=gain_noise_photons)
        y = norm.rvs(loc=muy, scale=sigma, size=gain_noise_photons)
        # Histogram of photons on focal plane, apply fiber mask, bin for detector
        H, xedges, yedges = np.histogram2d(x, y, bins=n_bins, range=[xlims, ylims])
        H = H * mask
        detector = plotRoutine.detector_bin(H, n_pix) + plotRoutine.add_gaussian_noise(n_pix, noise_level)
    
        """
        Centroid and error
        """
        # Pop last value from edge arrays, not necessary
        xc, yc = plotRoutine.centroid(detector, xdet_edges[:-1], ydet_edges[:-1])
        # Centroid assumes x,y center of pixels, add half pixel to offset
        xc += pixel_size[m] / 2.0
        yc += pixel_size[m] / 2.0
        err2[n] = np.sqrt((xc-mux)**2 + (yc-muy)**2)
        
    err[m,:] = [np.std(err2), np.mean(err2)]

plt.plot(pixel_size, err[:,1])
