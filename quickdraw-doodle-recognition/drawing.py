import ujson
import numpy as np
import cv2
# from PIL import Image, ImageDraw
# from rdp import rdp
from simplification.cutil import simplify_coords


def parse_drawing(drawing, normalize=True):
    strokes = ujson.loads(drawing)
    # Ignore the timestamps. Format as a list of numpy arrays
    # with two columns for the x and y coordinates.
    strokes = [np.array([s[0], s[1]]).T for s in strokes]
    if normalize:
        strokes = normalize_strokes(strokes)
    return strokes


def normalize_strokes(strokes):
    # Uniformly scale and center the drawing in [0,1]^2
    coords = np.vstack(strokes)
    min_coords = coords.min(0)
    max_coords = coords.max(0)
    delta = max_coords - min_coords
    max_delta = delta.max()
    shift = (max_delta - delta) / 2
    new_strokes = [((s - min_coords) + shift) / max_delta for s in strokes]
    return new_strokes


def simplify_strokes(strokes, epsilon):
    # Ramer-Douglas-Peucker algorithm.
    new_strokes = [simplify_coords(s, epsilon) for s in strokes]
    return new_strokes


def rotate_strokes(strokes, degrees):
    theta = np.radians(degrees)
    cos, sin = np.cos(theta), np.sin(theta)
    r = np.array(((cos, -sin), (sin, cos)))
    new_strokes = [np.dot(s, r.T) for s in strokes]
    return new_strokes


def mirror_strokes(strokes):
    r = np.array(((-1, 0), (0, 1)))
    new_strokes = [np.dot(s, r.T) for s in strokes]
    return new_strokes


# def render_image(strokes, image_size, asarray=True):
#     image = Image.new('L', (image_size, image_size), color=0)
#     draw = ImageDraw.Draw(image)
#     for stroke in strokes:
#         xy = ((image_size - 1) * stroke.flatten()).tolist()
#         draw.line(xy, fill=255, width=1)
#     if asarray:
#         image = np.asarray(image, dtype=np.float32) / 255
#         image = np.expand_dims(image, axis=-1)
#         return image

# Faster image rendering using OpenCV
def render_image(strokes, image_size):
    image = np.zeros((image_size, image_size, 1), dtype=np.float32)
    coords = [((image_size - 1) * s).astype(np.int32) for s in strokes]
    cv2.polylines(image, coords, False, 1)
    return image
