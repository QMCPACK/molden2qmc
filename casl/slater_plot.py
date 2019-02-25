#!/usr/bin/env python3

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from numpy.linalg import norm
import numpy as np

def plot_implicit(fn, bbox=(-1.5, 1.5)):
    ''' create a plot of an implicit function
    fn  ...implicit function (plot where fn==0)
    bbox ..the x,y,and z limits of plotted interval'''
    xmin, xmax, ymin, ymax, zmin, zmax = bbox*3
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    A = np.linspace(xmin, xmax, 100) # resolution of the contour
    B = np.linspace(xmin, xmax, 15) # number of slices
    A1,A2 = np.meshgrid(A,A) # grid on which the contour is plotted

    for z in B: # plot contours in the XY plane
        X,Y = A1,A2
        Z = fn(X,Y,z)
        cset = ax.contour(X, Y, Z+z, [z], zdir='z')
        # [z] defines the only level to plot for this contour for this value of z

    for y in B: # plot contours in the XZ plane
        X,Z = A1,A2
        Y = fn(X,y,Z)
        cset = ax.contour(X, Y+y, Z, [y], zdir='y')

    for x in B: # plot contours in the YZ plane
        Y,Z = A1,A2
        X = fn(x,Y,Z)
        cset = ax.contour(X+x, Y, Z, [x], zdir='x')

    # must set plot limits because the contour will likely extend
    # way beyond the displayed level.  Otherwise matplotlib extends the plot limits
    # to encompass all values in the contour.
    ax.set_xlim3d(xmin, xmax)
    ax.set_ylim3d(ymin, ymax)
    ax.set_zlim3d(zmin, zmax)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.show()

# http://www.theochem.ru.nl/~pwormer/Knowino/knowino.org/wiki/Slater_orbital.html
def slater_1s(x, y, z):
    Z = 3
    r = np.sqrt(x**2+y**2+z**2)
    return np.sqrt(Z**3/np.pi) * np.exp(-Z*r)

def slater_2s(x, y, z):
    Z = 3
    r = np.sqrt(x**2+y**2+z**2)
    return np.sqrt(Z**5/(3*np.pi)) * r * np.exp(-Z*r)

def slater_2px(x, y, z):
    Z = 2
    r = np.sqrt(x**2+y**2+z**2)
    return np.sqrt(Z**5/np.pi) * x * np.exp(-Z*r)

def slater_2py(x, y, z):
    Z = 2
    r = np.sqrt(x**2+y**2+z**2)
    return np.sqrt(Z**5/np.pi) * y * np.exp(-Z*r)

def slater_2pz(x, y, z):
    Z = 2
    r = np.sqrt(x**2+y**2+z**2)
    return np.sqrt(Z**5/np.pi) * z * np.exp(-Z*r)

def Be_1s2s(x1, y1, z1, x2, y2, z2):
    return slater_1s(x1, y1, z1) * slater_2s(x2, y2, z2) - slater_1s(x2, y2, z2) * slater_2s(x1, y1, z1)

def Be_1s2px(x1, y1, z1, x2, y2, z2):
    return slater_1s(x1, y1, z1) * slater_2px(x2, y2, z2) - slater_1s(x2, y2, z2) * slater_2px(x1, y1, z1)

def Be_1s2py(x1, y1, z1, x2, y2, z2):
    return slater_1s(x1, y1, z1) * slater_2py(x2, y2, z2) - slater_1s(x2, y2, z2) * slater_2py(x1, y1, z1)

def Be_1s2pz(x1, y1, z1, x2, y2, z2):
    return slater_1s(x1, y1, z1) * slater_2pz(x2, y2, z2) - slater_1s(x2, y2, z2) * slater_2pz(x1, y1, z1)

def Be(r12_minus, r34_minus, r12_plus):
    A = 3.0
    B = np.pi/4.0
    r1 = (r12_plus + r12_minus)/2.0
    r2 = (r12_plus - r12_minus)/2.0
    x1, y1, z1 = r1, 0, 0
    x2, y2, z2 = r2*np.cos(B), r2*np.sin(B), 0
    r3 = A
    r4 = A - r34_minus
    x3, y3, z3 = r3, 0, 0
    x4, y4, z4 = r4*np.cos(B), r4*np.sin(B), 0
    return Be_1s2s(x1, y1, z1, x2, y2, z2) * Be_1s2s(x3, y3, z3, x4, y4, z4)

def Be_4det(r12_minus, r34_minus, r12_plus):
    A = 3.0
    B = np.pi/4.0
    C = -0.15
    r1 = (r12_plus + r12_minus)/2.0
    r2 = (r12_plus - r12_minus)/2.0
    x1, y1, z1 = r1, 0, 0
    x2, y2, z2 = r2*np.cos(B), r2*np.sin(B), 0
    r3 = A
    r4 = A - r34_minus
    x3, y3, z3 = r3, 0, 0
    x4, y4, z4 = r4*np.cos(B), r4*np.sin(B), 0
    return Be_1s2s(x1, y1, z1, x2, y2, z2) * Be_1s2s(x3, y3, z3, x4, y4, z4) + \
           C * Be_1s2pz(x1, y1, z1, x2, y2, z2) * Be_1s2pz(x3, y3, z3, x4, y4, z4) + \
           C * Be_1s2py(x1, y1, z1, x2, y2, z2) * Be_1s2py(x3, y3, z3, x4, y4, z4) + \
           C * Be_1s2px(x1, y1, z1, x2, y2, z2) * Be_1s2px(x3, y3, z3, x4, y4, z4)

#plot_implicit(Be)
plot_implicit(Be_4det)
