from .utils import *
from .config import label_names
from itertools import product

sizes = [
    # (5, 25),
    (10, 50),
    (25, 100),
    (50, 200),
    (100, 400),
    # (200, 700)
]  # step, size
size_weights = {
    50: 1,
    100: 0.9,
    200: 0.7,
    400: 0.4
}


def slices(step, size):  # (image, step, size):
    # delta = size - 1
    # for i in range(0, 1400 - delta, step):
    #     for j in range(0, 2100 - delta, step):
    #         patch = image[i:i + delta, j:j + delta, :]
    #         yield i, j  # , patch
    return product(range(0, 1400 - size + 1, step), range(0, 2100 - size + 1, step))


def label_patch(row, col, size, masks):
    area = size * size
    scores = [0.0 for _ in masks]

    for k, mask in enumerate(masks):
        scores[k] = mask[row:row + size, col:col + size].sum() / area
        if scores[k] >= 0.8:
            return k + 1

    if sum(scores) <= 0.2:
        return 0

    return None


# def slice_and_save(labeled_images):
#     mkdir_if_not_exist(patch_dir)
#
#     for step, size in sizes:
#         subdir_prefix = patch_dir + sep + str(size) + '-'
#         for label_name in label_names:
#             mkdir_if_not_exist(subdir_prefix + label_name)
#
#     for image_filename, rlcs in labeled_images.items():
#         masks = [make_mask(rlc) for rlc in rlcs]
#
#         for step, size in sizes:
#             subdir_prefix = patch_dir + str(size) + '-'
#
#             for row, col in slices(step, size):
#                 label = label_patch(row, col, size, masks)
#
#                 if label is not None:
#                     continue
#
#                 save('%s%s%s%s.%d_%d.png' % (subdir_prefix, label_names[label], sep, image_filename, row, col), patch)
