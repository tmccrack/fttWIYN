# -*- coding: utf-8 -*-
"""
Created on Wed Apr 01 11:16:10 2015

@author: TMM
"""

import matplotlib.pyplot as plt
import numpy as np
import plotRoutine

field = 30.0  # [arcsecs]
pixel_size = 15.0  # [arcsecs] 
grid_size = 1000.  # points on the grid
#field_scale = np.linspace(-field/2.0, field/2.0, grid)
#field_grid = np.ones((grid, grid))

beam_diameter = 0.135  # 1/e^2
beam_diameter = 0.37  # 1/e
fiber_size = 1.2  # [arcsecs]

sigma = 1.0  # [arcsecs]
mu = 0.0  # [arcsecs]
magnitude = 0.0
n_photons = 10000.0

x = np.random.normal(0.0, 1.0, n_photons)
y = np.random.normal(0.0, 1.0, n_photons)
plotRoutine.plotter(x, y, pixel_size, grid_size, fiber_size)
