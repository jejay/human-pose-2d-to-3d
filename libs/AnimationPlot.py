import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.colors as colors
from matplotlib.animation import ArtistAnimation
import matplotlib.patheffects as pe
from Quaternions import Quaternions

DontGarbageCollect = []

def animation_plot(animations, filename=None, ignore_root=False, interval=33.33):
    
    footsteps = []
    
    for ai in range(len(animations)):
        anim = np.swapaxes(animations[ai][0].copy(), 0, 1)
        
        joints, root_x, root_z, root_r = anim[:,:-7], anim[:,-7], anim[:,-6], anim[:,-5]
        joints = joints.reshape((len(joints), -1, 3))
        
        rotation = Quaternions.id(1)
        offsets = []
        translation = np.array([[0,0,0]])
        
        if not ignore_root:
            for i in range(len(joints)):
                joints[i,:,:] = rotation * joints[i]
                joints[i,:,0] = joints[i,:,0] + translation[0,0]
                joints[i,:,2] = joints[i,:,2] + translation[0,2]
                rotation = Quaternions.from_angle_axis(-root_r[i], np.array([0,1,0])) * rotation
                offsets.append(rotation * np.array([0,0,1]))
                translation = translation + rotation * np.array([root_x[i], 0, root_z[i]])
        
        animations[ai] = joints
        footsteps.append(anim[:,-4:])
    
        """
        offsets = np.array(offsets)[:,0]
        print(offsets)
        
        import matplotlib.pyplot as plt
        
        plt.plot(joints[::10,0,0], joints[::10,0,2], 'o')
        plt.plot(joints[:,0,0], joints[:,0,2])
        for j in range(0, len(joints), 10):
            if footsteps[ai][j,0] > 0.5: plt.plot(joints[j,3,0], joints[j,3,2], '.', color='black')
            if footsteps[ai][j,2] > 0.5: plt.plot(joints[j,7,0], joints[j,7,2], '.', color='black') 
            #plt.plot(
            #    np.array([joints[j,0,0], joints[j,0,0] + 3 * offsets[j,0]]),
            #    np.array([joints[j,0,2], joints[j,0,2] + 3 * offsets[j,2]]), color='red')
        plt.xlim([-30, 30])
        plt.ylim([-10, 50])
        plt.grid(False)
        plt.show()
        """
        
    #raise Exception()
    
    footsteps = np.array(footsteps)
    
    scale = 1.25*((len(animations))/2)
    
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
    
    parents = np.array([-1,0,1,2,3,4,1,6,7,8,1,10,11,12,12,14,15,16,12,18,19,20])
    
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
        
            offset = 25*(ai-((len(animations))/2))
        
            for j in range(len(parents)):
                
                if parents[j] != -1:
                    lines[ai][j].set_data(
                        [ animations[ai][i,j,0]+offset, animations[ai][i,parents[j],0]+offset],
                        [-animations[ai][i,j,2],       -animations[ai][i,parents[j],2]])
                    lines[ai][j].set_3d_properties(
                        [ animations[ai][i,j,1],        animations[ai][i,parents[j],1]])
            
            changed += lines[ai]
            
        return changed
        
    plt.tight_layout()
        
    ani = animation.FuncAnimation(fig, 
        animate, np.arange(len(animations[0])), interval=interval)
    

    plt.show()
    
    DontGarbageCollect =+ ani

    return ani
        