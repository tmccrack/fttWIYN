# -*- coding: utf-8 -*-
"""
Created on Thu Apr 02 10:45:49 2015

@author: TMM
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

def coupling(fiber_radius, fwhm):
    sigma = fwhm / 2.34582  # [arcsecs]
    eff = norm.cdf(fiber_radius, loc=0.0, scale=sigma) - norm.cdf(-fiber_radius, loc=0.0, scale=sigma)
    return eff

fiber_radius = 0.45
fwhm = np.arange(0.75, 2.01, 0.01)   # [arcsecs]
eff = coupling(fiber_radius, fwhm)
'''
Plots
'''
plt.close()
figure()
plt.plot(fwhm, eff,'k')
plt.axis('tight')
plt.title('coupling efficiency')
plt.xlabel('FWHM [arcsecs]')
plt.ylabel('efficiency')
#plt.savefig('fiberCouplingEfficiency.png')

figure()
plt.plot(fwhm, eff / eff[0], 'k')
plt.axis('tight')
plt.title('relative efficiency')
plt.xlabel('FWHM [arcsecs]')
plt.ylabel('eff/max(eff)')
#plt.savefig('fiberCouplingEfficiencyRel.png')