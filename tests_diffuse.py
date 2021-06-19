from tests import *
from stochastic_diffuse.diffuse import *
from utils.interpolate import *
from utils.cursor_grid import *

# CONTROL PANEL
# Data Inputs
red, green, blue = png_to_arrays("test_images/windowsxp")
blue_contr = np.clip(blue * 2 - np.ones(blue.shape), 0, 1)
strength = 5
inv_sq = lambda x, y: 1 / (x * x + y * y)
dist = lambda x, y: np.sqrt(x * x + y * y)
field_x = lambda x, y: strength * (x + y)
field_y = lambda x, y: strength * (y - x)



# Parameters
max_rad, exchange = 1, 0.96
buffer_frames, rate = 10, 0.1
wfunc = lambda x: x

# Render settings
frames, steps_per_frame = 100, 1000
t = np.arange(frames)


# TESTING ZONE
dx, dy = fa.generate(field_x, D, D), fa.generate(field_y, D, D)

# Stochastic diffuse array
b_diff = Diffuse(blue, dx, dy, wfunc)
# Video time-blurring filter
b_smooth = SmoothStream(b_diff.out + np.zeros(b_diff.out.shape), buffer_frames, rate)

# Cursor map
rad, omega = 0.75, 0.04
cursor_x, cursor_y = lambda t: -rad * np.cos(t * omega), lambda t: rad * np.sin(t * omega)
stream_cx, stream_cy = cursor_x(t), cursor_y(t)

cgrid = CursorGrid(buf_size=10, gx0=-1, gy0=-1, gx1=1, gy1=1, grid_x=D, grid_y=D, start_x=rad, start_y=0, diffuse_rate=0.5, move_rate=0.06, heat_range=12, heat_rate=8)

for i in range(frames):
    # Take multiple diffusion steps per frame
    for j in range(steps_per_frame):
        b_diff.step(max_rad, exchange)

    # Update the cursor
    cgrid.heat(cgrid.smooth_x, cgrid.smooth_y)
    #print(stream_cx[i])
    cgrid.add_frame(stream_cx[i], stream_cy[i])
    b_diff.feed(cgrid.out)

    # Update the smoothing filter
    b_smooth.add_frame(b_diff.out)
    b_smooth.step()

    # Convert to a frame for openCV to render
    raster = cv_rgb(b_smooth.out, b_smooth.out, b_smooth.out)
    video.write(raster)
    print("Frame " + str(i))

video.release()
print("LETS FUCKING GOOOOOO")




