import numpy as np
import tensorflow as tf

def ornsteinUhlenbeckNoise(n, window, mu=0, sigma=1, theta=0.15, dt=2e-2):
    x_prev = tf.zeros([n])
    xx = []
    for i in range(window):
        x_prev = x_prev + theta * (mu - x_prev) * dt + sigma * tf.sqrt(dt) * tf.random_normal(shape=[n])
        xx.append(x_prev)
    return tf.concat([xx], axis=0)