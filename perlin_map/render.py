from perlin_map.perlinseries import *
from utils.interpolate import smooth

def render_interp(a, b, t, octaves, func=smooth):
    temp = a.copy()
    for i in range(a.size):
        # Full rotation
        temp.fields[i].load(a.fields[i].mag + (b.fields[i].mag - a.fields[i].mag) * func(t),
                            a.fields[i].dxn + (b.fields[i].dxn - a.fields[i].dxn) * func(t))
        # Larger rotations are masked
        #temp.fields[i].load(np.ones(a.fields[i].mag.shape) - np.abs(b.fields[i].mag - a.fields[i].mag) * np.sin(t * np.pi / 2) / (2 * np.pi),
                            #a.fields[i].dxn + (b.fields[i].dxn - a.fields[i].dxn) * func(t))
    temp.render(octaves)
    avg = np.ones(a.shape) * (a.avg + (b.avg - a.avg) * func(t))
    return temp.out + avg


# Yields a triplet of r, g, b arrays
# files is a list of file names, without the color or the suffix so rgb wont have to be repeated
def slideshow(files, transition, idle, rev_offset, octaves=-1):
    red_maps, green_maps, blue_maps = [], [], []

    for file in files:
        red_maps.append(load("perlinmaps/" + file + "_red.npz"))
        green_maps.append(load("perlinmaps/" + file + "_green.npz"))
        blue_maps.append(load("perlinmaps/" + file + "_blue.npz"))

    for i in range(len(files)):
        red_maps[i].align_angles(red_maps[(i + 1) % len(files)])
        green_maps[i].align_angles(green_maps[(i + 1) % len(files)])
        blue_maps[i].align_angles(blue_maps[(i + 1) % len(files)])

        red_maps[i].jitter()
        green_maps[i].jitter()
        blue_maps[i].jitter()

        for j in range(idle):
            yield red_maps[i].out, green_maps[i].out, blue_maps[i].out
        for j in range(transition):
            yield render_interp(red_maps[i], red_maps[(i + 1) % len(files)], j / transition, octaves),\
                  render_interp(green_maps[i], green_maps[(i + 1) % len(files)], j / transition, octaves), \
                  render_interp(blue_maps[i], blue_maps[(i + 1) % len(files)], j / transition, octaves)