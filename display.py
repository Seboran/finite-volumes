# -*- coding: utf-8 -*-

from fonctions import *
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import matplotlib as mpl
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pylab as plt
import matplotlib.collections as coll
import os
#==============================================================================
# Animation des solutions
#==============================================================================
  

""" Ces fonctions permettent d'afficher les résultats obtenus dans main.py
Ces fonctions sont un peu techniques et par conséquent ont été plus abondemment commentées """

def animate(title, mesh, t, datas, legends = ["P", "C"]):
    ''' This functions creates an animation and adds all the required legends
    Due to matplotlib limitations there are some trickled down technics to
    display a legend inside the code '''
    # Updates the screen
    def update(k):
    
        ax.clear()
        
        # Set title and max z
        ax.set_title(title + ' t = ' + str(round(t[k], 2)) + 's')
        ax.set_zlim(0, np.max(datas[0][0]))

        # Prints all data in a single figure
        for data in datas:
            surf(mesh, data[k], ax)
        
        # Legend display
        lines = []
        color_number = 0
        colors = ['b', 'orange']
        for legend in legends:
            lines.append(mpl.lines.Line2D([0],[0], color = colors[color_number],linestyle="none", marker = 'o'))
            color_number += 1
        ax.legend(lines, legends, numpoints = 1)

    
        
    ''' Figure initialisation '''

    fig = plt.figure()
    ax = fig.add_subplot(111, projection = '3d')
    for data in datas:
        surf(mesh, data[0], ax)
    
    # Set title and max z
    ax.set_title(title + ' t = ' + str(round(t[0], 2)) + 's')
    ax.set_zlim(0, np.max(datas[0][0]))

    # Prints all data in a single figure
    lines = []
    for legend in legends:
        lines.append(mpl.lines.Line2D([0],[0], linestyle="none", marker = 'o'))
    
    ax.legend(lines, legends, numpoints = 1)
    ''' End of initialisation '''
    
    ani = animation.FuncAnimation(fig, update, frames = len(t),
                            interval = t[-1] / len(t), blit = False, repeat = True)

    
    return ani

def export_animation(ani, title):
    '''Saves the animation into a mp4 and a gif'''
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Nirina'), bitrate=1200)
    
    ani.save('animations/animation' + title + '.mp4', dpi=80, writer= writer)
    os.system("ffmpeg -i animations/animation" + title + ".mp4 gifs\\animation" + title + ".gif")
    

