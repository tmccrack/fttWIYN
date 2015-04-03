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

field = 10.0  # [arcsecs]
pixel_size = 0.1  # [arcsecs] 

"""
Specifying the spot size and fiber size. Fiber sized to 1/e (37% of peak) or 1/e2 (13.5% of peak).
These values are beam radius though, need diameter.
TODO: calculate 1/e diameter
"""
fwhm = 0.76  # fwhm of beam [arcsecs]
fiber_radius = 0.45  # 1/e radius [arcsecs]
sigma = fwhm / 2.3458  # [arcsecs]
mu = 0.0  # [arcsecs]

"""
Source and integration time to get number of photons.
Set up distributions to for binning.
"""
magnitude = 0.0  # TODO
n_photons = 10000.0

# Generate random photons
x = norm.rvs(loc=0.0, scale=sigma, size=n_photons)
y = norm.rvs(loc=0.0, scale=sigma, size=n_photons)

"""
Fine binning
"""
n_bins = 500
bin_size = field / n_bins
xlims = [-(n_bins/2.0 * bin_size), (n_bins/2.0 * bin_size)]
ylims = [-(n_bins/2.0 * bin_size), (n_bins/2.0 * bin_size)]

# Mask focal plane
mask = plotRoutine.fiber_mask(fiber_radius, n_bins, bin_size)
H, xedges, yedges = np.histogram2d(x, y, bins=n_bins, range=[xlims, ylims])
H = H * mask

n_pix = field / pixel_size
detector = plotRoutine.detector_bin(H, n_pix) + plotRoutine.add_noise(n_pix)

#plt.imshow(data)
#plt.figure()
plt.imshow(H * mask, interpolation='none')
figure()
plt.imshow(detector, interpolation='none')
plt.colorbar()



'''
mask = plotRoutine.fiber_mask(fiber_radius, n_bins, bin_size)
data, x1, y2 = plotRoutine.binner(x, y, n_bins, bin_size)
'''
