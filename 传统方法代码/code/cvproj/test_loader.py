from .utils import *


def load_test_images():
    if os.path.exists(config.test_file_list):
        with open(config.test_file_list, 'r') as f:
            return [filename.strip() for filename in f.readlines()]

    test_images = [filename for filename, match in ls_pattern(config.test_dir, '[0-9a-f]+\\.jpg')]
    with open(config.test_file_list, 'w') as f:
        f.writelines(filename + '\n' for filename in test_images)
    return test_images


def encode_mask(mask):
    row_order = mask.reshape((-1, 1), order='F')
    last_state = False
    count = 0
    rlc = []
    for i in range(row_order.shape[0]):
        if row_order[i]:
            count += 1
            if not last_state:
                last_state = True
                rlc.append(i)
        else:
            if last_state:
                rlc.append(count)
                last_state = False
                count = 0
    if last_state:
        rlc.append(count)
    return ' '.join(map(str, rlc))


def write_test_csv(test_image_results, output_filename):
    test_csv = []
    for filename, result in test_image_results:
        for label, label_id in labels.items():
            if label_id == 0:
                continue
            test_csv.append(filename + '_' + label + ',' + encode_mask(result == label_id) + '\n')
    with open(output_filename, 'w') as f:
        log('write test csv named ' + output_filename)
        f.writelines(test_csv)
