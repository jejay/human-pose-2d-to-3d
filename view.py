import sys
sys.path.append("libs")
import numpy as np
from AnimationPlotPoints import animation_plot

enc = np.load("encoded_frames.npz")["encoded_frames"]
preprocess = np.load("preprocess.npz")
Xmean = preprocess["Xmean"]
Xstd = preprocess["Xstd"]

data = np.load("data/predictions/9.Discussion.Discussion 1.54138969.h5-sh.npz")

#enc_in = np.load("data/predictions/11.Walking.Walking.60457274.h5-sh.npz")["enc_in"]
dec_out = data["dec_out"]
poses3d = data["poses3d"]

enc_ani = enc[0].swapaxes(1,2) * Xstd + Xmean

a = animation_plot([dec_out, poses3d])
"""
import matplotlib.pyplot as plt

n50 = np.arange(0,50)/50
n60 = np.arange(0,60)/60

plt.xlim([0, 0.1])

plt.plot(n60, np.repeat([1], 60), "+", markersize=10)
plt.plot(n50, np.repeat([1], 50), "+")
"""