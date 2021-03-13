import random

from . import features, slicer, train_loader, model
from .utils import *

batch_size = 128


# def preprocess():
#     labeled_images = train_loader.load_labeled_images()
#     slicer.slice_and_save(labeled_images)


# def sample_slice(size):
#     for label_name, label in labels.items():
#         dirname = '%s%d-%s' % (slicer.patch_dir, size, label_name)
#         sample_names = os.listdir(dirname)
#         for i in range(0, len(sample_names), batch_size):
#             yield dirname, label, sample_names[i:i + 128]
#
#
# def train_only(model_type):
#     models = []
#     model.set_type(model_type)
#
#     for _, size in slicer.sizes:
#         model_for_size = model.make_model()
#
#         for dirname, label, samples in sample_slice(size):
#             X = np.ndarray((len(samples), features.cnt))
#             y = np.ones(len(samples)) * label
#             for i, image_name in enumerate(samples):
#                 image = load(image_name, dirname)
#                 X[i, :] = features.extract(image)
#             model_for_size.partial_fit(X, y, [0, 1, 2, 3, 4])
#
#         print('finished training the model for size %d' % size)
#         models.append(model_for_size)
#     return models


def get_image_mask(image_name, rlcs):
    return load(image_name, config.data_dir), [make_mask(rlc) for rlc in rlcs]


def sample(it, ratio):
    while True:
        try:
            tmp = next(it)
        except StopIteration:
            return
        if random.random() < ratio:
            yield tmp


def scan_checkpoints():
    checkpoints = []
    for name, result in ls_pattern(config.patch_dir, 'x%d-(\\d+)\\.pickle' % (slicer.sizes[-1][1])):
        checkpoints.append(int(result.group(1)))
    return checkpoints


def preprocess(feature_init):
    labeled_images = train_loader.load_labeled_images()
    if feature_init is None:
        features.init_with(labeled_images)
    else:
        features.init_with(None, feature_init)

    feature_matrices = {size: ([], []) for step, size in slicer.sizes}

    log('scanning checkpoints...')
    checkpoints = scan_checkpoints()
    start = max(checkpoints) if checkpoints else 0
    log('detected checkpoints before %d.' % start)
    if start < len(labeled_images):
        log('collecting feature matrices from %d to %d...' % (start, len(labeled_images)))
        i = start
        for i in range(start, len(labeled_images)):
            image, masks = get_image_mask(*labeled_images[i])

            log('processing image %d' % i)
            global_features = features.extract_global(image)
            for step, size in slicer.sizes:
                window_cnt = ((1400 - size) // step + 1) * ((2100 - size) // step + 1)

                X = np.ndarray(shape=(window_cnt, features.feature_vector_size), dtype=np.float32)
                y = np.ndarray(shape=(window_cnt, 1), dtype=np.uint8)
                row_index = 0

                for row, col in slicer.slices(step, size):
                    label = slicer.label_patch(row, col, size, masks)

                    if label is not None:
                        features_in_patch = features.extract_local(row, col, size, global_features)

                        if features_in_patch is not None:
                            X[row_index, :] = features_in_patch
                            y[row_index, 0] = label
                            row_index += 1

                feature_matrices[size][0].append(X[:row_index, :])
                feature_matrices[size][1].append(y[:row_index, :])

            if (i + 1) % 500 == 0:
                collect(i + 1, feature_matrices)
                feature_matrices = {size: ([], []) for step, size in slicer.sizes}
                checkpoints.append(i + 1)

        i += 1
        collect(i, feature_matrices)
        checkpoints.append(i)
        log('finished collecting feature matrices.')
    return checkpoints


def stack(feature_matrices):
    return {
        size: (
            np.vstack(Xs),
            np.vstack(ys).reshape(-1)
        )
        for size, (Xs, ys) in feature_matrices.items()
    }


def collect(index, feature_matrices):
    collected_feature_matrices = stack(feature_matrices)
    for step, size in slicer.sizes:
        X, y = collected_feature_matrices[size]
        log('collected number of vectors used in model for size %d: %d' % (size, y.shape[0]))

        if X.shape[0] == 0 and Y.shape[0] == 0:
            log('[WARN ] no data to collect, skipping index=%d, size=%d' % (index, size))

        mkdir_if_not_exist(config.patch_dir)
        save_np(X, 'x%d-%s.pickle' % (size, str(index)), config.patch_dir)
        save_np(y, 'y%d-%s.pickle' % (size, str(index)), config.patch_dir)

    return collected_feature_matrices


def sample_train_data(checkpoints):
    feature_matrices = {size: ([], []) for _, size in slicer.sizes}
    for checkpoint in checkpoints:
        log('sampling train data for checkpoint %d...' % checkpoint)
        for _, size in slicer.sizes:
            y = load_np('y%d-%s.pickle' % (size, str(checkpoint)), config.patch_dir)
            X = load_np('x%d-%s.pickle' % (size, str(checkpoint)), config.patch_dir)

            # sampling
            if size >= 200:
                feature_matrices[size][1].append(y[:, np.newaxis])
                feature_matrices[size][0].append(X[:, :])
                continue
            elif size >= 100:
                ratio = [0.3, 0.3, 0.3, 0.3, 0.3]
            elif size >= 50:
                ratio = [0.1, 0.1, 0.1, 0.1, 0.1]
            else:
                ratio = [0]

            chosen = np.random.uniform(size=y.shape[0]) < np.choose(y, ratio)

            feature_matrices[size][1].append(y[chosen, np.newaxis])
            feature_matrices[size][0].append(X[chosen, :])

    return stack(feature_matrices)


def train_after_process(feature_matrices, model_type):
    model.set_type(model_type)

    preprocessed_feature_matrices = {
        size: (X, y.reshape(-1))
        for size, (X, y)
        in feature_matrices.items()
    }

    weights_for_size = [
        preprocessed_feature_matrices[size][1].shape[0] - np.bincount(preprocessed_feature_matrices[size][1])
        for _, size in slicer.sizes
    ]
    models = model.make_models(weights_for_size)
    model.set_train(models)

    # if feature_matrices is None:
    #     def load_xy(size_to_load):
    #         return (
    #             load_np('%sx%d-final.pickle' % (config.patch_dir, size_to_load)),
    #             load_np('%sy%d-final.pickle' % (config.patch_dir, size_to_load)).reshape(-1)
    #         )
    # else:

    for model_for_size, (step, size) in zip(models, slicer.sizes):
        log('training for size %d...' % size)
        X, y = preprocessed_feature_matrices[size]

        if model.need_loop():
            for i in range(3000):
                if i % 10 == 0:
                    log('training epoch %4d (after last epoch, loss=%f)...' % (i, model_for_size.get_last_loss()))
                if model_for_size.fit(X, y):
                    break
            log('finished after training %4d epochs at a loss of %f.' % (i + 1, model_for_size.get_last_loss()))
        else:
            model_for_size.fit(X, y)

    return models


def train_only(model_type):
    checkpoints = scan_checkpoints()
    sampled = sample_train_data(checkpoints)
    return train_after_process(sampled, model_type)


def train(model_type, feature_init):
    checkpoints = preprocess(feature_init)
    sampled = sample_train_data(checkpoints)
    return train_after_process(sampled, model_type)
