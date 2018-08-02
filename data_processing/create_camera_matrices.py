import numpy as np
rotmats = []
for i in range(240*64*2):
    theta = np.random.uniform(low=-np.pi,high=np.pi)
    alpha = np.random.normal(loc=-np.pi/64, scale=np.pi/32)
    rotmat_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                       [0, 1, 0],
                       [-np.sin(theta), 0, np.cos(theta)]])
    rotmat_x = np.array([[1, 0, 0],
                       [0, np.cos(alpha), -np.sin(alpha)],
                       [0, np.sin(alpha), np.cos(alpha)]])
    
    rotmat = rotmat_x @ rotmat_y
    rotmats.append(rotmat.T)

np.savez("../data/cammats.npz", mats=np.array(rotmats).astype(np.float32))