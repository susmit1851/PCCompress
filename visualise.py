from utils import *

def visualisePCD(pcd_points):
    '''
    Note the arg pcd_points should be an array of 3D points, eg. of shape (num_points, 3)
    '''
    X = [p[0] for p in pcd_points]
    Y = [p[1] for p in pcd_points]
    Z = [p[2] for p in pcd_points]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X, Y, Z, c='r')
    plt.show()