# -*- coding: utf-8 -*-
"""
Created on Wed Apr 01 11:16:10 2015

@author: TMM
"""

import matplotlib.pyplot as plt
import numpy as np
import plotRoutine

field = 15.0  # [arcsecs]
pixel_size = 10.0  # [arcsecs] 
grid_size = 1000.  # points on the grid
#field_scale = np.linspace(-field/2.0, field/2.0, grid)
#field_grid = np.ones((grid, grid))

"""
Specifying the spot size and fiber size. Fiber sized to 1/e (37% of peak) or 1/e2 (13.5% of peak).
These values are beam radius though, need diameter.
TODO: calculate 1/e diameter
"""
fwhm = 2.0  # fwhm of beam [arcsecs]
fiber_size = 2 * 0.8493218 * fwhm  # 1/e^2 DIAMETER [arcsecs]
#beam_diameter = 0.37  # 1/e DIAMETER TODO
sigma = fwhm / 2.3458  # [arcsecs]
mu = 0.0  # [arcsecs]

"""
Source and integration time to get number of photons.
Set up distributions to for binning.
"""
magnitude = 0.0  # TODO
n_photons = 10000.0

x = np.random.normal(0.0, sigma, n_photons)
y = np.random.normal(0.0, sigma, n_photons)

"""
Fine binning, take out fiber
"""
n_bins = 5000
assert np.mod(n_bins, 2) == 0  # Make fine binning even
bin_size = field / n_bins  # [arcsecs]
mask = plotRoutine.fiber_mask(fiber_size, n_bins, bin_size)
data, x1, y2 = plotRoutine.binner(x, y, n_bins, bin_size)

#plt.imshow(data)
plt.imshow(mask * data, interpolation='none')


#plotRoutine.plotter(x, y, pixel_size, grid_size, fiber_size)
