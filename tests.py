# TODO: Fix magnitude descent
# TODO: Mask error onto successive magnitude arrays
# TODO: Clean up magic numbers
# TODO: Create orientation array so vectors are rotating efficiently
# TODO: Map functions from space coords to gradient grid coords

# TODO: Perturbation w/ position and velocity
# TODO: Goal: 512x512 total square error < 1000
    # Gradients affected by an inv.square force
    # Also proportional with cross product length


import numpy as np
import cv2
from PIL import Image as im
import stochastic_diffuse as diffuse
import utils.func_array as fa

# ALL TESTS ARE OF 512x512 IMAGES
D = 512
UPSCALE = 1
frameSize = (D, D)

# OpenCV avi video setup
width = D * UPSCALE
height = D * UPSCALE
FPS = 60
seconds = 60
VSCALE = 256

# Posterizing for "level curve" render
ROUND = 1

fourcc = cv2.VideoWriter_fourcc(*'MP42')
video = cv2.VideoWriter('render.avi', fourcc, float(FPS), (width, height))


# Returns a tuple of a red, green, and blue array from an image, clipped/padded to test dimensions
def png_to_arrays(path):
    image = im.open(path)
    image = image.resize((D, D))
    arrays = (np.array(image) / VSCALE).transpose([2, 0, 1])
    return arrays[2], arrays[1], arrays[0]

# Convert input channels of range [0, 1] into an array to directly feed into OpenCV
def cv_rgb(red, green, blue):
    red = np.clip(red * VSCALE, 0, VSCALE - 1)
    green = np.clip(green * VSCALE, 0, VSCALE - 1)
    blue = np.clip(blue * VSCALE, 0, VSCALE - 1)
    raster = np.array((np.dstack((red, green, blue)) // ROUND) * ROUND, dtype=np.uint8)
    out = raster.repeat(UPSCALE, axis=0).repeat(UPSCALE, axis=1)
    return out