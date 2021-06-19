import numpy as np
from utils.interpolate import smooth
import itertools

class Perlin:
    # Shape and res are both shapes
    def __init__(self, shape, res, omega):
        self.shape, self.res = shape, res

        # Caches
        self.delta = (res[0] / shape[0], res[1] / shape[1])
        self.d = (shape[0] // res[0], shape[1] // res[1])
        self.grid = np.mgrid[0:res[0]:self.delta[0], 0:res[1]:self.delta[1]].transpose(1, 2, 0) % 1

        self.cell_size = res
        # Build gradient
        self.dxn = 2*np.pi*np.random.rand(res[0]+1, res[1]+1) # Everything after this must be looped
        self.mag = np.ones((res[0]+1, res[1]+1))
        self.omega = omega#np.random.normal(omega, omega, (res[0]+1, res[1]+1))
        self.gradients = np.dstack((np.cos(self.dxn), np.sin(self.dxn))) # * self.mag

        # Tile gradients by resolution
        #self.gradients = self.gradients.repeat(self.d[0], 0).repeat(self.d[1], 1)

        # Corner slices, only need to set once because they are views
        self.g00 = self.gradients[:-self.d[0], :-self.d[1]]
        self.g10 = self.gradients[self.d[0]:, :-self.d[1]]
        self.g01 = self.gradients[:-self.d[0], self.d[1]:]
        self.g11 = self.gradients[self.d[0]:, self.d[1]:]
        self.out = np.zeros((shape[0], shape[1], 3))
        self.k = 1

        x, y = np.arange(shape[0]), np.arange(shape[1])
        self.coords = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])


    def update(self, step):
        self.rotate(self.omega)
        self.recompute()

    def rotate(self, theta):
        self.dxn += theta
        self.gradients = np.dstack((np.cos(self.dxn), np.sin(self.dxn)))
        self.gradients = self.gradients.repeat(self.d[0], 0).repeat(self.d[1], 1)

    def recompute(self):
        # Empty grid
        self.grid = np.mgrid[0:self.res[0]:self.delta[0], 0:self.res[1]:self.delta[1]].transpose(1, 2, 0) % 1

        n00 = np.sum(np.dstack((self.grid[:, :, 0], self.grid[:, :, 1])) * self.gradients[:-self.d[0], :-self.d[1]], 2)
        n10 = np.sum(np.dstack((self.grid[:, :, 0] - 1, self.grid[:, :, 1])) * self.gradients[self.d[0]:, :-self.d[1]], 2)
        n01 = np.sum(np.dstack((self.grid[:, :, 0], self.grid[:, :, 1] - 1)) * self.gradients[:-self.d[0], self.d[1]:], 2)
        n11 = np.sum(np.dstack((self.grid[:, :, 0] - 1, self.grid[:, :, 1] - 1)) * self.gradients[self.d[0]:, self.d[1]:], 2)

        t = smooth(self.grid)
        n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
        n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
        self.grid = np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1) + 1
        self.out = self.grid

    def set_gradients(self, grad):
        self.gradients = grad


class Compound:
    # Layers is a list of Perlin fields
    def __init__(self, layers, weights):
        self.layers = layers
        self.weights = weights
        self.shape = layers[0].out.shape
        self.out = np.zeros(layers[0].out.shape)

    def update(self, step):
        self.out = np.zeros(self.layers[0].out.shape)
        for i in range(len(self.layers)):
            self.layers[i].update(step)
            self.out += self.layers[i].out * self.weights[i]

        self.out = (self.out // 48) * 64






def generate_perlin_noise_2d(shape, res, tileable=(False, False)):
    """Generate a 2D numpy array of perlin noise.
    Args:
        shape: The shape of the generated array (tuple of two ints).
            This must be a multple of res.
        res: The number of periods of noise to generate along each
            axis (tuple of two ints). Note shape must be a multiple of
            res.
        tileable: If the noise should be tileable along each axis
            (tuple of two bools). Defaults to (False, False).
        interpolant: The interpolation function, defaults to
            t*t*t*(t*(t*6 - 15) + 10).
    Returns:
        A numpy array of shape shape with the generated noise.
    Raises:
        ValueError: If shape is not a multiple of res.
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]]\
             .transpose(1, 2, 0) % 1
    #Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]:
        gradients[-1,:] = gradients[0,:]
    if tileable[1]:
        gradients[:,-1] = gradients[:,0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[    :-d[0],    :-d[1]]
    g10 = gradients[d[0]:     ,    :-d[1]]
    g01 = gradients[    :-d[0],d[1]:     ]
    g11 = gradients[d[0]:     ,d[1]:     ]
    #Ramps
    n00 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    #Interpolation
    t = smooth(grid)
    n0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    n1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*n0 + t[:,:,1]*n1)

generate_perlin_noise_2d((10, 10), (10, 10))
