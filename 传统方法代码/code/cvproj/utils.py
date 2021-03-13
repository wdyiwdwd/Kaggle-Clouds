import re

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os
from .config import *
import datetime


def log(s):
    print(datetime.datetime.now().strftime('%H:%M:%S.%f ') + s)


def load(name, directory):
    image = cv.imread(name if directory is None else path_join(directory, name))
    if image is None:
        log('failed to load image "%s"' % name)
    return image


def save(name, image, directory):
    name = path_join(directory, name)
    if print_file_operations:
        log('saving ' + name)
    cv.imwrite(name, image)


def bgr2rgb(image):
    return image[:, :, [2, 1, 0]]


def show(image):
    plt.figure()
    plt.imshow(bgr2rgb(image))
    plt.show()


def show_binary(image_binary, vmin=None, vmax=None):
    plt.figure()
    plt.imshow(image_binary, cmap='gray', vmin=vmin, vmax=vmax)
    plt.show()


def threshold(image):
    image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    image_binary = np.choose(image_hsv[:, :, 2] > 127, [np.uint8(0), np.uint8(255)])
    return image_binary


def grayscale(image):
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image_gray


def make_mask(rlc):
    mask = np.zeros([1400, 2100], dtype=np.bool)
    if rlc is None:
        return mask
    for start, length in rlc:
        x, y_start = divmod(start, 1400)
        x_span, y_end = divmod(y_start + length - 1, 1400)
        y_end += 1

        if x_span == 0:
            mask[y_start:y_end, x] = True
        else:
            mask[y_start:, x] = True
            if x_span > 1:
                mask[:, x + 1:x + x_span] = True
            if x + x_span == 2100:
                log('[WARN ] mask flows over (1400,2100).\n       start=%d,length=%d,x=%d,y_start=%d,x_span=%d,y_end=%d' % (start, length, x, y_start, x_span, y_end))
                continue
            mask[:y_end, x + x_span] = True

    return mask


def mask_2channel(mask, image):
    return np.choose(mask, [0, image])


def mask_3channel(mask, image):
    return np.choose(mask[:, :, np.newaxis], [0, image])


def show_masked_binary(mask, image_binary):
    plt.imshow(
        np.select(
            [np.logical_and(mask, image_binary), np.logical_and(mask, np.logical_not(image_binary)),
             image_binary.astype(bool)],
            [.8, .2, 1],
            default=0
        ),
        cmap='gray'
    )


def show_masked(mask, image):
    show(mask_3channel(image, mask))


def path_join(dirname, path):
    return os.path.join(dirname, path)


def mkdir_if_not_exist(path):
    if not os.path.exists(path):
        if print_file_operations:
            log('making dir: ' + path)
        os.mkdir(path)


def ls_pattern(directory, pattern):
    regex = re.compile(pattern)
    return filter(lambda x: x[1] is not None, ((name, re.fullmatch(regex, name)) for name in os.listdir(directory)))


def save_np(obj, filename, dirname):
    with open(path_join(dirname, filename), 'wb') as f:
        np.save(f, obj)


def load_np(filename, dirname):
    try:
        with open(path_join(dirname, filename), 'rb') as f:
            return np.load(f)
    except IOError:
        return None
