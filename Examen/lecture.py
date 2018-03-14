# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 14:16:31 2017

@author: nirin
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as plt
import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from collections import defaultdict
from pylab import *
import csv


from numpy import genfromtxt

maxNt = int(input('Choose max Nt: '))
maxN = int(input('Choose max N: '))
numberT = int(input('Choose number Nt: '))
numberN = int(input('Choose number N: '))

end_file = str(maxNt) + ',' + str(maxN) + ',' + str(numberT) + ',' + str(numberN) + '.csv'

X = genfromtxt('Ns' + end_file, delimiter=',')

Y = genfromtxt('Nts' + end_file, delimiter = ',')

Z = genfromtxt('e' + end_file, delimiter = ',')

plt.contourf(X, Y, Z)












