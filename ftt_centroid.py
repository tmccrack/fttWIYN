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
magnitude = 16.0
diameter = 3.5  # [m]
wavelength = 635  # [nm]
bandpass = 880 - 390  # [nm]
extinction = 0.5  # Extinction in magnitudes per airmass
transmission = 0.523
qe = 0.9
fiber_rejection = 0.16
integration_time = 0.01  # [seconds]


#n_photons = 10000

"""
Detector and fiber properties
"""
field = 5.0  # [arcsecs]
#
#pixel_size = np.array([1.0, 0.75, 0.5, 0.25, 0.2, 0.175, 0.15, 0.125, 0.1, 0.075, 0.05, 0.025])
pixel_size = np.array([0.025])
#
n_pix = np.round(field / pixel_size)  # Pixels in ~5 arcsecond field
n_bins = n_pix * 30.0  # Bins for putting photons on prior to detector binning
bin_size = pixel_size / 30.0
bins_per_pix = n_bins / n_pix  # Photon bins per detector pixel

fwhm = 0.76  # fwhm of beam [arcsecs]
fiber_radius = 0.45  # 1/e radius [arcsecs]
sigma = fwhm / 2.3458  # [arcsecs]
mux = 0.0  # [arcsecs]
muy = 0.0  # [arcsecs]
em_gain = 1000.
rn = 0.5  # read noise [electrons/pixel]
dark = 0.001 * integration_time  # dark current [electrons/pixel]
noise_level = rn+dark

realizations = 1

"""
Realizations
"""
err = np.zeros((len(pixel_size), 2))
for m in range(len(pixel_size)):        
    """
    Fiber mask, binning parameters
    """
      # Grid to be binned into detector
    mask = plotRoutine.fiber_mask(fiber_radius, n_bins[m], bin_size[m])
    lim = n_bins[m]/2.0 * bin_size[m]  # edge of bin grid in arcseconds
    
    # Pixel edges on detector
    xedges = np.linspace(-lim, lim, n_bins[m]+1)
    yedges = np.linspace(-lim, lim, n_bins[m]+1)    
    
    
    # Do not consider fiber rejection, taken up in mask
    n_photons = plotRoutine.source_photons(bandpass, wavelength, magnitude, diameter, extinction) * qe * transmission * integration_time    
    
    xdet_edges = xedges[::bins_per_pix[m]]  # [arcsecs]
    ydet_edges = yedges[::bins_per_pix[m]]  # [arcsecs]
    err2 = np.zeros(realizations,2)
    print(m)    
    
    for n in range(realizations):
        # Generate random photons
        # Dominate error in EMCCDs is multiplicative error in gain stage
        # Gain factor adds sqrt(2) to the shot noise
        # Accomodate by multiplying n_photons by sqrt(2)
        x = norm.rvs(loc=mux, scale=sigma, size=np.int(n_photons*em_gain))
        y = norm.rvs(loc=muy, scale=sigma, size=np.int(n_photons*em_gain))
        # Histogram of photons on focal plane, apply fiber mask, bin for detector
        H, xedges, yedges = np.histogram2d(x, y, bins=n_bins[m], range=[[-lim, lim], [-lim, lim]])
        H = H * mask
        detector = plotRoutine.detector_bin(H, n_pix[m]) + plotRoutine.add_gaussian_noise(n_pix[m], noise_level)
        
        """
        Centroid and error
        """
        # Pop last value from edge arrays, not necessary
        xc, yc = plotRoutine.centroid(detector, xdet_edges[:-1], ydet_edges[:-1])
        # Centroid assumes x,y center of pixels, add half pixel to offset
        xc += pixel_size[m] / 2.0
        yc += pixel_size[m] / 2.0
        err2[n,:] = (xc-mux), (yc-muy)
        print('\t' + str(n))
        
    err[m,0:1] = np.mean(err2,axis=0), 
    err[m,2] = np.std(np.sqrt((np.mean(err2[0,:])-mux)**2 + (np.mean(err2[1,:])-muy)**2))
    #figure()
    #plt.imshow(detector, interpolation='none')
    #plt.colorbar()

figure()
plt.plot(pixel_size, err[:,1])
