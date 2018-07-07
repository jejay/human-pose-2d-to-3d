import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.colors as colors
from matplotlib.animation import ArtistAnimation
import matplotlib.patheffects as pe

import sys
sys.path.append('../motion/')

from Quaternions import Quaternions

manifolded = np.load("manifolded.npz")
ani = manifolded["Xorgi"][0,0:72,100].reshape(-1,3)

animations = [ani]
ignore_root=False
interval=33.33
    
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

plt.plot([ani[0, 0]], [ani[0, 2]], [ani[0, 1]], "c+", label="0")
plt.plot([ani[1, 0]], [ani[1, 2]], [ani[1, 1]], "c*", label="1")
plt.plot([ani[2, 0]], [ani[2, 2]], [ani[2, 1]], "co", label="2")
plt.plot([ani[3, 0]], [ani[3, 2]], [ani[3, 1]], "cs", label="3")
plt.plot([ani[4, 0]], [ani[4, 2]], [ani[4, 1]], "c^", label="4")
plt.plot([ani[5, 0]], [ani[5, 2]], [ani[5, 1]], "c.", label="5")
plt.plot([ani[6, 0]], [ani[6, 2]], [ani[6, 1]], "r+", label="6")
plt.plot([ani[7, 0]], [ani[7, 2]], [ani[7, 1]], "r*", label="7")
plt.plot([ani[8, 0]], [ani[8, 2]], [ani[8, 1]], "ro", label="8")
plt.plot([ani[9, 0]], [ani[9, 2]], [ani[9, 1]], "rs", label="9")
plt.plot([ani[10, 0]], [ani[10, 2]], [ani[10, 1]], "r^", label="10")
plt.plot([ani[11, 0]], [ani[11, 2]], [ani[11, 1]], "r.", label="11")
plt.plot([ani[12, 0]], [ani[12, 2]], [ani[12, 1]], "g+", label="12")
plt.plot([ani[13, 0]], [ani[13, 2]], [ani[13, 1]], "g*", label="13")
plt.plot([ani[14, 0]], [ani[14, 2]], [ani[14, 1]], "go", label="14")
plt.plot([ani[15, 0]], [ani[15, 2]], [ani[15, 1]], "gs", label="15")
plt.plot([ani[16, 0]], [ani[16, 2]], [ani[16, 1]], "g^", label="16")
plt.plot([ani[17, 0]], [ani[17, 2]], [ani[17, 1]], "g.", label="17")
plt.plot([ani[18, 0]], [ani[18, 2]], [ani[18, 1]], "b+", label="18")
plt.plot([ani[19, 0]], [ani[19, 2]], [ani[19, 1]], "b*", label="19")
plt.plot([ani[20, 0]], [ani[20, 2]], [ani[20, 1]], "bo", label="20")
plt.plot([ani[21, 0]], [ani[21, 2]], [ani[21, 1]], "bs", label="21")
plt.plot([ani[22, 0]], [ani[22, 2]], [ani[22, 1]], "b^", label="22")
plt.plot([ani[23, 0]], [ani[23, 2]], [ani[23, 1]], "b.", label="23")


parents = np.array([-1,-1,1,2,3,4,1,6,7,8,1,10,11,12,12,14,15,16,12,18,19,20])
for j in range(len(parents)):
    if parents[j] != -1:
        plt.plot([ ani[j,0], ani[parents[j],0]],
            [ani[j,2],       ani[parents[j],2]],
            [ ani[j,1],        ani[parents[j],1]], 'black', alpha=0.2)

plt.legend()

#parents = np.array([-1,0,1,2,3,4,1,6,7,8,1,10,11,12,12,14,15,16,12,18,19,20])

#for ai, anim in enumerate(animations):
#    lines.append([plt.plot([0,0], [0,0], [0,0],
#                           color=acolors[ai],
#                           lw=2,
#                           path_effects=[pe.Stroke(linewidth=3, foreground='black'),
#                                         pe.Normal()],
#                           )[0] for _ in range(anim.shape[1])])

    
plt.tight_layout()

    
plt.show()
        