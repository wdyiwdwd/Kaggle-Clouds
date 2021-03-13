import csv
from .utils import log
from .config import *


def pairwise(string):
    if len(string) == 0:
        return None
    it = iter(string.split(' '))
    result = []
    try:
        while True:
            result.append((int(next(it)), int(next(it))))
    except StopIteration:
        return result


def load_labeled_images():
    labeled_images = []
    labeled_images_index = dict()

    with open(config.train_filename, 'r') as f:
        log('reading train data from file ' + config.train_filename)
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            # print(row[0])
            names = row[0].split('_')
            name = names[0]
            label_name = names[1]

            if name not in labeled_images_index:
                labeled_images_index[name] = len(labeled_images)
                labeled_images.append((name, [None, None, None, None]))

            labeled_images[labeled_images_index[name]][1][labels[label_name] - 1] = pairwise(row[1])

    return labeled_images

