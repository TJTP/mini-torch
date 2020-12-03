import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 

def load_data(path):
    return np.loadtxt(path, dtype=float, delimiter=",")

def draw_scatter(features, labels, color=None):
    fig = plt.figure()
    ax3 = Axes3D(fig)

    x1 = features[:, :1]
    x2 = features[:, 1:2]

    ax3.scatter(x1, x2, labels, c=color)
    plt.show()

def draw_2d(features, labels):
    fig = plt.figure()

    plt.scatter(features, labels)
    plt.show()