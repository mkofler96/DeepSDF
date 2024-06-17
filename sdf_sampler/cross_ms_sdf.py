import numpy as np

class CrossMsSDF():
    def __init__(self, radius):
        self.r = radius
        
    def SDF(self, xyz):
        output = np.linalg.norm(xyz, axis=1, ord=np.inf) - 0.9

        #substract corners
        corners = np.array(np.meshgrid([-1, 1], [-1, 1], [-1, 1])).T.reshape(-1,3)
        for corner in corners:
            sphere_like = np.linalg.norm(xyz-corner, axis=1, ord=3) - self.r
            output = np.maximum(output, -sphere_like)

        return output.reshape(-1,1)