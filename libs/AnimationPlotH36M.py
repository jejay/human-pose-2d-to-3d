import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.colors as colors
from matplotlib.animation import ArtistAnimation
import matplotlib.patheffects as pe
from Quaternions import Quaternions

DontGarbageCollect = []

def animation_plot(animations, ignore_root=False, interval=33.33):        

    for ai in range(len(animations)):
        anim = animations[ai]
        
        joints, vel, rot = anim[:,0:-5], anim[:,-4:-1], anim[:,-1]
        joints = joints.reshape((len(joints), -1, 3))
        
        rotation = Quaternions.id(1)
        offsets = []
        translation = np.array([[0,0,0]])
        
        if not ignore_root:
            for i in range(len(joints)):
                joints[i,:,:] = rotation * joints[i] + translation
                rotation = Quaternions.from_angle_axis(rot[i], np.array([0,0,1])) * rotation
                offsets.append(rotation * np.array([0,0,1]))
                translation = translation + rotation * vel[i]
    
        animations[ai] = joints
    
    
    scale = 30*((len(animations))/2)
    
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
    lines = []
    
    parents = np.array([10,0,1,2,3,10,5,6,7,8,-1,10,11,12,11,14,15,16,16,11,19,20,21,21])
    
    for ai, anim in enumerate(animations):
        lines.append([plt.plot([0,0], [0,0], [0,0],
                               color=acolors[ai],
                               lw=2,
                               path_effects=[pe.Stroke(linewidth=3, foreground='black'),
                                             pe.Normal()],
                               )[0] for _ in range(anim.shape[1])])
    
    def animate(i):
        changed = []
        
        for ai in range(len(animations)):
        
            offset = 1000*(ai-((len(animations))/2))
        
            for j in range(len(parents)):
                
                if parents[j] != -1:
                    lines[ai][j].set_data(
                        [ animations[ai][i,j,0]+offset, animations[ai][i,parents[j],0]+offset],
                        [-animations[ai][i,j,1],       -animations[ai][i,parents[j],1]])
                    lines[ai][j].set_3d_properties(
                        [ animations[ai][i,j,2],        animations[ai][i,parents[j],2]])
            
            changed += lines[ai]
            
        return changed
        
    plt.tight_layout()
        
    ani = animation.FuncAnimation(fig, 
        animate, np.arange(len(animations[0])), interval=interval)
    

    plt.show()
    
    DontGarbageCollect += ani

    return ani
        