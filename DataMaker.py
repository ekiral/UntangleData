import numpy as np

import torch
import torch.tensor as tt
import torch.nn as nn
import torch.nn.functional as F



# Random Point Generator for the points on the
# surface of a torus with radii R and r.
def rpg(R, r, N):
    #two random angles are chosen
    theta1 = 2*np.pi*np.random.rand(N)
    theta2 = 2*np.pi*np.random.rand(N)
    sr = r*np.ones(N)
    #define a rotator matrix for each theta in the array theta2
    Roty = np.array(list(map(lambda x: rot(x, axis = 'y'), theta2)))
    #now create a set of points of the form (r, 0, 0)^t
    prerot = np.zeros((N,3))
    prerot[:,0] = sr
    #and rotate them according to the corresponding theta2 angles
    addpoints = np.array(list(map(lambda x, y: x.dot(y), Roty, prerot)))
    #Now add these points to (R,0,0)^t, 
    #then rotate them according to angles in theta1 around  axis
    Rotz = np.array(list(map(rot, theta1)))
    prerot2 = np.zeros((N,3))
    prerot2[:,0] = R*np.ones(N)
    prerot2 += addpoints
    points = np.array(list(map(lambda x,y: x.dot(y), Rotz, prerot2))) 
    return points


# Random Point Generator for points that have the body of the torus
# where first points are picked from a 3D gaussian with Mean (R,0,0)
# Standard Deviation r, and later rotated on the z axis with a random angle.

def rpg2(R,r,N):
    points = r*np.random.multivariate_normal(mean = [R,0,0], cov = np.identity(3), size = N)
    thetas = 2*np.pi*np.random.rand(N)
    Rots = np.array(list(map(rot, thetas)))
    for i in np.arange(N):
        points[i] = points[i].dot(Rots[i])
    return points

def rot(theta, axis = 'z'):
    c, s = np.cos(theta), np.sin(theta)
    if axis == 'z':
        R = np.array(((c,-s,0),(s,c,0),(0,0,1)))
    if axis == 'y':
        R = np.array(((c, 0, -s), (0,1,0),(s,0,c)))
    if axis == 'x':
        R = np.array(((1,0,0),(0,c,-s),(0,s,c)))
    return R


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# This function visualizes 3D data points colored red or blue.
# Input is a list xs of points in 3D Euclidean space and another
# list ys of the same length where each entry is [0,1] or [0,1].
# The ys provides the coloring of the xs via their list index.
def dataVisualizer(xs, ys, size = 0.1):
    BluePoints = xs[list(map(lambda a: list(a)==[1,0], ys))]
    RedPoints = xs[list(map(lambda a: list(a)==[0,1], ys))]
    bxs, bys, bzs = BluePoints.transpose(0,1)
    rxs, rys, rzs = RedPoints.transpose(0,1)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(rxs, rys, rzs, s = size, c = 'red', marker='o', alpha = 0.1)
    ax.scatter(bxs, bys, bzs, s = size, c = 'blue', marker = 'o', alpha = 0.1)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()



# This two functions create a set of points and labels,
# with N1 blue points labeled [1,0] and N2 red points labeled [0,1]

# This is the data that produces 
def dataCreatorLinkedTori(R, r, N1, N2):
    BluePoints = rpg2(R,r,N1)
    RedPoints = rpg2(R,r,N2)
    Rot = rot(np.pi/2, axis = 'y')
    RedPoints = np.array(list(map(lambda x: Rot.dot(x) + np.array([0,R,0]),RedPoints)))
    X = torch.cat((tt(BluePoints, dtype = torch.float), tt(RedPoints, dtype=torch.float)),0)
    Y = torch.cat((tt([1,0], dtype=torch.float).repeat(N1,1), tt([0,1], dtype=torch.float).repeat(N2,1)),0)
    return X,Y

def dataCreatorSheath(R,r1, r2, N1, N2):
    BluePoints = rpg(R,r1,N1)
    RedPoints = rpg(R,r2, N2)
    X = torch.cat((tt(BluePoints, dtype = torch.float), tt(RedPoints, dtype=torch.float)), 0 )
    Y = torch.cat((tt([1,0], dtype=torch.float).repeat(N1,1), tt([0,1], dtype=torch.float).repeat(N2,1)),0)
    return X,Y
    
