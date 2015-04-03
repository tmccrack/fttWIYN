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
pixel_size = 10.0  # [arcsecs] 
grid_size = 1000.  # points on the grid

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

# Create histogram and kill center value used for mathcing arrays

"""
Bin for detector
"""
n_bins = 500
bin_size = field / n_bins
xlims = [-(n_bins/2.0 * bin_size), (n_bins/2.0 * bin_size)]
ylims = [-(n_bins/2.0 * bin_size), (n_bins/2.0 * bin_size)]

ym, xm = np.ogrid[-fiber_radius:fiber_radius, -fiber_radius:fiber_radius]
mask = plotRoutine.fiber_mask(fiber_radius, n_bins, bin_size)
H, xedges, yedges = np.histogram2d(x, y, bins=n_bins, range=[xlims, ylims])

'''
# Mask for fiber
x_masked = ma.masked_inside(x, -fiber_radius, fiber_radius).compressed()
y_masked = ma.masked_inside(y, -fiber_radius, fiber_radius).compressed()

# Adjust length for histogram
x = np.zeros(n_photons)
x[0:len(x_masked)] = x_masked
y=np.zeros(n_photons)
y[0:len(y_masked)] = y_masked



H[n_bins/2, n_bins/2] = 0 
'''

#plt.imshow(data)
#plt.figure()
plt.imshow(H * mask, interpolation='none')
figure()
plt.imshow(mask)
plt.colorbar()



'''
mask = plotRoutine.fiber_mask(fiber_radius, n_bins, bin_size)
data, x1, y2 = plotRoutine.binner(x, y, n_bins, bin_size)
'''
