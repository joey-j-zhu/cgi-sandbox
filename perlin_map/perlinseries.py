import perlin_map.perlinfield as pf
import numpy as np

class PerlinSeries:
    # Image resolution must be a multiple of 3
    def __init__(self, image, depth):
        px = image.shape[0] * image.shape[1]
        self.avg = (np.sum(image) / px)
        self.image = image - np.ones(image.shape) * self.avg
        self.error = self.image
        self.shape = image.shape
        self.out = np.zeros(image.shape)
        self.fields = []
        self.size = depth
        for i in range(1, depth + 1):
            self.fields.append(pf.PerlinField(image, (2 ** i, 2 ** i)))

    def copy(self):
        new = PerlinSeries(self.image, self.size)
        for i in range(self.size):
            shp = self.fields[i].mag.shape
            new.fields[i].mag = self.fields[i].mag + np.zeros(shp)
            new.fields[i].dxn = self.fields[i].dxn + np.zeros(shp)
        return new

    # Stream: yield intermittent frames which are multiples of the given number. 0 to turn off
    def epoch(self, iterations, learning_rate, stream=0):
        residue = self.image
        for i in range(len(self.fields)):
            self.fields[i].set_image(residue)
            for j in range(iterations):
                self.fields[i].descent(learning_rate)
                if stream != 0 and j % stream == 0:
                    self.render()
                    yield residue
            residue = residue - self.fields[i].out


    def render(self, octaves=-1):
        #print(self.avg)
        self.out = np.ones(self.shape) * self.avg
        if octaves == -1:
            octaves = len(self.fields)
        for i in range(octaves):
            self.fields[i].render()
            self.out = self.out + self.fields[i].out #SEGFAULT

    def update_error(self):
        self.error = self.image - self.out

    def total_error(self):
        return 0.5 * np.sum(self.error * self.error, axis=None)

    # Generate an RGB array for OpenCV render from a scalar array
    # For a good render, make sure the array is within [-1, 1]
    def quick_rgb(self, array):
        val = np.clip((array + np.ones(array.shape)) * 128, 0, 255)
        out = np.array(np.dstack((val, val, val)), dtype=np.uint8)
        return out

    # Save this Perlin series to a file
    def save(self, path):
        arrays = [self.image, np.array([self.avg])]
        for i in range(self.size):
            arrays.append(self.fields[i].dxn)
            arrays.append(self.fields[i].mag)
        sav = np.array(arrays, dtype=object)
        np.savez(path, sav)

    def set_avg(self, avg):
        self.avg = avg

    def spin(self, revolutions):
        for field in self.fields:
            field.rotate(revolutions * 2 * np.pi)

    #
    def align_angles(self, other):
        for i in range(self.size):
            for x in range(self.fields[i].dxn.shape[0]):
                for y in range(self.fields[i].dxn.shape[1]):
                    if self.fields[i].dxn[x, y] - other.fields[i].dxn[x, y] < 0:
                        # If we keep making full rotations, one of them will have to be less than a half circle
                        while abs(self.fields[i].dxn[x, y] - other.fields[i].dxn[x, y]) > np.pi:
                            self.fields[i].dxn[x, y] += 2 * np.pi
                        # Rotate CCW to bring up angle
                    else:
                        while abs(self.fields[i].dxn[x, y] - other.fields[i].dxn[x, y]) > np.pi:
                            self.fields[i].dxn[x, y] -= 2 * np.pi

    def jitter(self):
        for i in range(self.size):
            offset = np.random.randint(-1, 1, self.fields[i].dxn.shape)
            self.fields[i].rotate(offset * 2 * np.pi)


# Read a Perlin series and load it
def load(path):
    arrays = np.load(path, allow_pickle=True)['arr_0']
    size = (len(arrays) - 2) // 2
    # Must pass an empty array in so construction works
    series = PerlinSeries(arrays[0], size)
    for i in range(size):
        series.fields[i].load(arrays[2 * i + 3], arrays[2 * i + 2])
    #print(arrays)
    series.set_avg(arrays[1][0])
    series.render()
    return series

def noise(shape, res):
    image = np.zeros(shape)
    p = pf.PerlinField(image, res)
    p.render()
    return p