from . import slicer, features, model, test_loader
from .utils import *


def predict_sized_results(image, image_code, models, save_sized_results, save_sized_masks):
    results = []
    scores = []
    if save_sized_masks or save_sized_results:
        mkdir_if_not_exist('output/')
        mkdir_if_not_exist('output/' + image_code)

    global_features = features.extract_global(image)

    for (step, size), model_for_size in zip(slicer.sizes, models):
        result_for_size = np.zeros(shape=((1400 - size) // step + 1, (2100 - size) // step + 1), dtype=np.uint8)
        score_for_size = np.zeros(shape=((1400 - size) // step + 1, (2100 - size) // step + 1, 5), dtype=np.float32)

        for row, col in slicer.slices(step, size):
            features_in_patch = features.extract_local(row, col, size, global_features)
            if features_in_patch is None:
                label_for_patch = np.ones(label_cnt) / label_cnt
            else:
                label_for_patch = model_for_size.predict_one_score(features_in_patch)
            score_for_size[row // step, col // step, :] = label_for_patch
            result_for_size[row // step, col // step] = label_for_patch.argmax()

        if save_sized_results:
            save('%s/%d.png' % (image_code, size), result_for_size, config.output_dir)

        if save_sized_masks:
            for i, label_name in enumerate(label_names):
                save('%s/%d-%s.png' % (image_code, size, label_name), np.choose(result_for_size == i, [0, 255]),
                     config.output_dir)

        results.append(result_for_size)
        scores.append(score_for_size)

    return results, scores


def predict_pixelwise_results(results_scores_for_size, div4):
    results_for_size, scores_for_size = results_scores_for_size
    if div4:
        y_cnt = 350
        x_cnt = 525
    else:
        y_cnt = 1400
        x_cnt = 2100
    pixel_label_scores = np.zeros(shape=(y_cnt, x_cnt, 5), dtype=np.float32)
    final_results = np.zeros(shape=(y_cnt, x_cnt), dtype=np.uint8)
    for (step, size), score_for_size, result_for_size in zip(slicer.sizes, scores_for_size, results_for_size):
        weight = slicer.size_weights[size]
        y_result, x_result, _ = score_for_size.shape
        if not div4:
            for i in range(y_result):
                for j in range(x_result):
                    # pixel_label_scores[i*step:i*step+size, j*step:j*step+size, :] += score_for_size[i, j, :]
                    pixel_label_scores[i * step:i * step + size, j * step:j * step + size,
                    result_for_size[i, j]
                    ] += weight # * score_for_size[i, j, result_for_size[i, j]]
        else:
            scaled_step = step // 4
            scaled_size = size // 4
            for i in range(y_result):
                for j in range(x_result):
                    pixel_label_scores[
                    i * scaled_step:(i * scaled_step + scaled_size),
                    j * scaled_step:(j * scaled_step + scaled_size),
                    result_for_size[i, j]
                    ] += weight # * score_for_size[i, j, result_for_size[i, j]]
                    # :] += score_for_size[i, j, :]

    # pixel_label_scores[i, j, 0] /= 2 # more weights on non-none labels
    for i in range(y_cnt):
        for j in range(x_cnt):
            final_results[i, j] = pixel_label_scores[i, j, :].argmax()

    cnt = np.bincount(final_results.reshape(-1))
    discard_indicator = np.logical_and(0 < cnt, cnt < y_cnt * x_cnt / 64)
    for i, discard in enumerate(discard_indicator):
        if i != 0 and discard:
            final_results[final_results == i] = 0

    return final_results


def test(image_filename, feature_init, models, save_sized_result, save_sized_masks, div4):
    model.set_eval(models)
    assert feature_init is not None, "bag-of-word vocabulary should be specified when testing"
    features.init_with(None, feature_init)

    image = load(image_filename, None)
    image_code = image_filename.split('/')[-1].split('.')[0]

    pixelwise_results = predict_pixelwise_results(
        predict_sized_results(image, image_code, models, save_sized_result, save_sized_masks), div4
    )
    if save_sized_masks:
        for i, label_name in enumerate(label_names):
            save('%s/final-%s.png' % (image_code, label_name), np.choose(pixelwise_results == i, [0, 255]),
                 config.output_dir)

    return pixelwise_results


def evaluate(feature_init, models, div4):
    model.set_eval(models)
    assert feature_init is not None, "bag-of-word vocabulary should be specified when testing"
    features.init_with(None, feature_init)

    test_filenames = test_loader.load_test_images()
    mkdir_if_not_exist(config.test_patch_dir)

    results = []
    checkpoints = []
    for name, result in ls_pattern(config.test_patch_dir, 'output-before(\\d+)\\.csv'):
        checkpoints.append(int(result.group(1)))
    start = max(checkpoints) if checkpoints else 0
    log('starting to predict for test cases...')
    if start != len(test_filenames):
        i = start
        for i in range(start, len(test_filenames)):
            image_filename = test_filenames[i]
            log('performing test on image %d' % i)

            image = load(image_filename, config.test_dir)
            image_code = image_filename.split('.')[0]
            pixelwise_results = predict_pixelwise_results(
                predict_sized_results(image, image_code, models, False, False), div4
            )
            results.append((image_filename, pixelwise_results))

            if (i + 1) % 200 == 0:
                test_loader.write_test_csv(results, '%soutput-before%d.csv' % (config.test_patch_dir, i + 1))
                checkpoints.append(i + 1)
                results = []

        i += 1
        test_loader.write_test_csv(results, '%soutput-before%d.csv' % (config.test_patch_dir, i))
        checkpoints.append(i)

    with open(config.test_filename, 'w') as f:
        f.write('Image_Label,EncodedPixels\n')
        for checkpoint in checkpoints:
            with open('%soutput-before%d.csv' % (config.test_patch_dir, checkpoint)) as part_file:
                f.writelines(part_file.readlines())
