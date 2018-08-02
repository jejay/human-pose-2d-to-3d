import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.colors as colors
from matplotlib.animation import ArtistAnimation
import matplotlib.patheffects as pe
from Quaternions import Quaternions

def animation_plot(animations, interval=20):
    
    for ai in range(len(animations)):
        animations[ai] = animations[ai].reshape((animations[ai].shape[0], -1, 3))
    
    scale = 51.25*((len(animations))/2)
    
    fig = plt.figure(figsize=(12,8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-scale*30, scale*30)
    ax.set_zlim3d( 0, scale*60)
    ax.set_ylim3d(-scale*30, scale*30)
    ax.set_xticks([], [])
    ax.set_yticks([], [])
    ax.set_zticks([], [])
    ax.set_aspect('equal')
    
    acolors = list(sorted(colors.cnames.keys()))[::-1]
    points = []
    
    for ai, anim in enumerate(animations):
        points.append([plt.plot([0], [0], [0],
                               "*", 
                               color=acolors[ai],
                               markersize=10
                               )[0] for _ in range(anim.shape[1])])
    def animate(i):
        changed = []
        for ai in range(len(animations)):
            
            offset = 1000*(ai-((len(animations))/2))
        
            for j in range(animations[ai].shape[1]):
                points[ai][j].set_data([animations[ai][i,j,0]+offset],[-animations[ai][i,j,1]])
                points[ai][j].set_3d_properties([animations[ai][i,j,2]])
            
            changed += points[ai]
            
        return changed
        
    plt.tight_layout()
        
    ani = animation.FuncAnimation(fig, 
        animate, np.arange(len(animations[0])), interval=interval)
    

    plt.show()

    return ani
        