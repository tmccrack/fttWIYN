# -*- coding: utf-8 -*-
"""
Created on Wed Apr 08 20:38:20 2015

@author: Katie
"""

import numpy as np

class Moffat:
    """
    Moffat distribution
    f(x,y;a,b) =    (b-1) * (pi*a**2)**-1 * (1 + ((x**2+y**2)/a**2))**-b
    Define with fwhm and b (beta)
    alpha = hwhm*(2^(1.0/beta)-1)^(-0.5)
    beta typically ranges from 2 to 5. IRAF sets beta at 2.5,
    """
    
    def __init__(self, fwhm, beta):
        self.alpha = fwhm / 2.0 / np.sqrt((2**(1.0/beta)-1))
        self.beta= beta
        
    def value(self, x, y):
        """
        Return value of Moffat distribution and coordinates (x,y)
        """
        return (self.beta-1) * (np.pi*self.alpha**2)**(-1) * (1+(x**2+y**2)/self.alpha**2)**(-self.beta)
        