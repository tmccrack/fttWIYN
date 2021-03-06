"""
Adapted from http://www.astrobetter.com/visualization-fun-with-python-2d-histogram-with-1d-histograms-on-axes/
Thanks Jess K!
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, MaxNLocator
plt.ion()


def centroid(data, x, y):
    """
    Determine centroid of 2D data array with values located at x, y
    """
    xc = np.dot(x, np.sum(data, axis=1))
    yc = np.dot(y, np.sum(data, axis=0))
    total = np.sum(np.sum(data))      
    return xc/total, yc/total

def binner(x, y, n_bins, bin_size):
    """
    Bin random x and y data into array (n_bins, n_bins)
    """
    xlims = [-(n_bins/2.0 * bin_size), (n_bins/2.0 * bin_size)] 
    ylims = [-(n_bins/2.0 * bin_size), (n_bins/2.0 * bin_size)]
    data, xedges, yedges = np.histogram2d(x, y, n_bins, range=[xlims, ylims])
    return data, xedges, yedges


def detector_bin(data, n_pix):
    """
    Take data array and bin/sum into array of (n_pix, n_pix)
    """
    assert (np.mod(data.shape[0], n_pix) == 0) & (np.mod(data.shape[1], n_pix) == 0), "Non-integer number of bins per pixel."
    shape = n_pix, data.shape[0]//n_pix, n_pix, data.shape[1]//n_pix
    return data.reshape(shape).sum(-1).sum(1)


def add_poisson_noise(dimension, noise):
    return np.random.poisson(lam=noise, size=dimensoin**2).reshape(dimension, dimension)

def add_gaussian_noise(dimension, noise):
    """
    Return array (size, size) with random noise at the specified level.
    """
    return np.random.normal(loc=noise, scale=1.0, size=dimension**2).reshape(dimension, dimension)
    
    
def fiber_mask(fiber_radius, n_bins, bin_size, refl):
    """
    Returns centered circular fiber mask
    """
    assert n_bins%2 == 0, "Number of bins in fiber mask must be even."
    r = fiber_radius / bin_size;
    # Check if bin size is even or odd
    x, y = np.ogrid[0:n_bins/2, 0:n_bins/2]
    mask = (x*x + y*y >= r*r).astype(int) + (x*x + y*y < r*r).astype(int)*refl # 1/4 of mask, must concatenate
    mask = np.column_stack((np.fliplr(mask), mask))  # 1/2 of mask
    mask = np.vstack((np.flipud(mask), mask))  # all of mask
    return mask
    
 
def source_photons(bandpass, wavelength, magnitude, diameter, extinction):
    """
    Number of photons collected per second by a telescope given diameter from source of given magnitude
    Adopted from Massey et al. 1988
    Specify bandpass and mean wavelength of observation
    Does not assume transmission or detector efficiency
    """
    return np.floor(4.5e10 / wavelength * 10**(-(magnitude + extinction) / 2.5) * diameter**2 * bandpass) 
 
# Define a function to make the ellipses
def ellipse(ra,rb,ang,x0,y0,Nb=100):
    xpos, ypos = x0, y0
    radm, radn = ra, rb
    an = ang
    co, si = np.cos(an), np.sin(an)
    the = np.linspace(0,2*np.pi,Nb)
    X = radm * np.cos(the) * co - si * radn * np.sin(the) + xpos
    Y = radm * np.cos(the) * si + co * radn * np.sin(the) + ypos
    return X, Y


def plotter(x, y, n_pix, pixel_size, grid_size, fiber_size):
    # Set up default x and y limits
    xlims = [-(np_pix/2.0), (n_pix/2.0)] * pixel_size
    ylims = [-(np_pix/2.0), (n_pix/2.0)] * pixel_size
     
    # Define the locations for the axes
    left, width = 0.12, 0.55
    bottom, height = 0.12, 0.55
    bottom_h = left_h = left+width+0.02
     
    # Set up the geometry of the three plots
    rect_temperature = [left, bottom, width, height] # dimensions of temp plot
    rect_histx = [left, bottom_h, width, 0.25] # dimensions of x-histogram
    rect_histy = [left_h, bottom, 0.25, height] # dimensions of y-histogram
     
    # Set up the size of the figure
    fig = plt.figure(1, figsize=(9.5,9))
     
    # Find the min/max of the data
    xmin = np.min(xlims)
    xmax = np.max(xlims)
    ymin = np.min(ylims)
    ymax = np.max(ylims)
     
    # Make the 'main' temperature plot
    # Define the number of bins
    nxbins = 50
    nybins = 50
    nbins = 100
     
    xbins = np.linspace(start = xmin, stop = xmax, num = nxbins)
    ybins = np.linspace(start = ymin, stop = ymax, num = nybins)
    xcenter = (xbins[0:-1]+xbins[1:])/2.0
    ycenter = (ybins[0:-1]+ybins[1:])/2.0
    aspectratio = 1.0*(xmax - 0)/(1.0*ymax - 0)
     
    H, xedges,yedges = np.histogram2d(x, y, pixel_size)
    X = xcenter
    Y = ycenter
    Z = H
    
    # Make the three plots
    axTemperature = plt.axes(rect_temperature) # temperature plot
    axHistx = plt.axes(rect_histx) # x histogram
    axHisty = plt.axes(rect_histy) # y histogram
     
    # Remove the inner axes numbers of the histograms
    nullfmt = NullFormatter()
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)
     
    # Plot the temperature data
    cax = (axTemperature.imshow(H, extent=[xmin,xmax,ymin,ymax],
           interpolation='nearest', origin='lower',aspect=aspectratio))
     
    # Plot the temperature plot contours
    contourcolor = 'white'
    xcenter = np.mean(x)
    ycenter = np.mean(y)
    ra = np.std(x)
    rb = np.std(y)
    ang = 0
     
    X,Y=ellipse(ra,rb,ang,xcenter,ycenter)
    axTemperature.plot(X,Y,"k:",ms=1,linewidth=2.0)
    axTemperature.annotate('$1\\sigma$', xy=(X[15], Y[15]), xycoords='data',xytext=(10, 10),
                           textcoords='offset points', horizontalalignment='right',
                           verticalalignment='bottom',fontsize=25)
     
    X,Y=ellipse(2*ra,2*rb,ang,xcenter,ycenter)
    axTemperature.plot(X,Y,"k:",color = contourcolor,ms=1,linewidth=2.0)
    axTemperature.annotate('$2\\sigma$', xy=(X[15], Y[15]), xycoords='data',xytext=(10, 10),
                           textcoords='offset points',horizontalalignment='right',
                           verticalalignment='bottom',fontsize=25, color = contourcolor)
     
    X,Y=ellipse(3*ra,3*rb,ang,xcenter,ycenter)
    axTemperature.plot(X,Y,"k:",color = contourcolor, ms=1,linewidth=2.0)
    axTemperature.annotate('$3\\sigma$', xy=(X[15], Y[15]), xycoords='data',xytext=(10, 10),
                           textcoords='offset points',horizontalalignment='right',
                           verticalalignment='bottom',fontsize=25, color = contourcolor)
     
    #Plot the axes labels
    axTemperature.set_xlabel(xlabel,fontsize=25)
    axTemperature.set_ylabel(ylabel,fontsize=25)
     
    #Make the tickmarks pretty
    ticklabels = axTemperature.get_xticklabels()
    for label in ticklabels:
        label.set_fontsize(18)
        label.set_family('serif')
     
    ticklabels = axTemperature.get_yticklabels()
    for label in ticklabels:
        label.set_fontsize(18)
        label.set_family('serif')
     
    #Set up the plot limits
    axTemperature.set_xlim(xlims)
    axTemperature.set_ylim(ylims)
     
    #Set up the histogram bins
    xbins = np.arange(xmin, xmax, (xmax-xmin)/nbins)
    ybins = np.arange(ymin, ymax, (ymax-ymin)/nbins)
     
    #Plot the histograms
    axHistx.hist(x, bins=xbins, color = 'blue')
    axHisty.hist(y, bins=ybins, orientation='horizontal', color = 'red')
     
    #Set up the histogram limits
    axHistx.set_xlim( np.min(x), np.max(x) )
    axHisty.set_ylim( np.min(y), np.max(y) )
     
    #Make the tickmarks pretty
    ticklabels = axHistx.get_yticklabels()
    for label in ticklabels:
        label.set_fontsize(12)
        label.set_family('serif')
     
    #Make the tickmarks pretty
    ticklabels = axHisty.get_xticklabels()
    for label in ticklabels:
        label.set_fontsize(12)
        label.set_family('serif')
     
    #Cool trick that changes the number of tickmarks for the histogram axes
    axHisty.xaxis.set_major_locator(MaxNLocator(4))
    axHistx.yaxis.set_major_locator(MaxNLocator(4))
     
    #Show the plot
    plt.draw()