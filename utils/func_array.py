import numpy as np

def load_coordinates(x_size, y_size, increment):
    x = np.transpose(np.tile(np.arange(x_size) / increment, (y_size, 1)))
    y = np.tile(np.arange(y_size) / increment, (x_size, 1))
    return x, y

def centered_unit(l):
    x_coords, y_coords = load_coordinates(l, l, l / 2)
    x_coords -= np.ones((l, l))
    y_coords -= np.ones((l, l))
    return x_coords, y_coords

# func_x and func_y are both 2-variable functions that take in the x and y arrays
def generate(func, x, y):
    assert x == y
    cx, cy = centered_unit(x)
    return func(cx, cy)
