import numpy as np

class Diffuse:
    def __init__(self, initial_array, xdrift, ydrift, weight_func=lambda x: x):
        self.out = initial_array
        self.xdrift, self.ydrift = xdrift, ydrift
        self.shape = initial_array.shape
        self.wfunc = weight_func
        self.total = np.sum(initial_array)

    # Sample the Bernoulli variable of probability p
    def flip(self, p):
        return np.random.random() < p


    def clip(self, x, y):
        return min(max(0, x), self.shape[0] - 1), min(max(0, y), self.shape[1] - 1)

    # Return a random n-size partition of variables which add up to 1 by virtue of Poisson process equivalence
    def fission(self, n):
        arr = np.log(np.random.rand(n))
        return arr / np.sum(arr)

    # Randomly sample from the entire grid
    def grid_sample(self):
        return np.random.randint(0, self.shape[0] - 1), np.random.randint(0, self.shape[1] - 1)

    # Sample from the axis-aligned square of length 2d+1 centered on x, y
    def box_sample(self, x, y, d):
        x, y = self.clip(x, y)
        dx, dy = np.random.randint(-d, d + 1), np.random.randint(-d, d + 1)
        return self.clip(x + dx, y + dy)


    def stoc(self, n):
        extra = self.flip(n % 1)
        if extra:
            return np.floor(n) + 1
        else:
            return np.floor(n)

    # Return a random neighbor with inverse radius weighting
    # max distance squared, NOT distance
    def invsq_neighbor(self, x, y, max_dist):
        cx, cy = self.box_sample(x + self.stoc(self.xdrift[x, y]), y + self.stoc(self.ydrift[x, y]), max_dist)
        return self.clip(int(cx), int(cy))


    def sample_filter(self):
        x, y = 0, 0
        finished = 0
        while not finished:
            x, y = self.grid_sample()
            finished = self.flip(self.out[x, y])
        return x, y

    # Swap the contents of two cells
    def transfer(self, x1, y1, x2, y2, amt1, amt2):
        cell_1, cell_2 = self.out[x1, y1], self.out[x2, y2]
        self.out[x1, y1] = cell_1 + amt1 * (cell_2 - cell_1)
        self.out[x2, y2] = cell_2 + amt2 * (cell_1 - cell_2)

    # Perform a single
    def step(self, d, diff=1, prob_weight=True):
        x, y, finished = 0, 0, False
        while not finished:
            x, y = self.sample_filter()
            finished = self.flip(self.wfunc(self.out[x, y]))

        nx, ny = self.invsq_neighbor(x, y, d)
        self.transfer(x, y, nx, ny, diff, diff)

    def feed(self, input):
        self.total += np.sum(input)
        self.out += input
