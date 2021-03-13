labels = {'None': 0, 'Sugar': 1, 'Flower': 2, 'Fish': 3, 'Gravel': 4}
label_names = ['None', 'Sugar', 'Flower', 'Fish', 'Gravel']
label_cnt = len(label_names)
print_file_operations = True


class Config(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


config = Config(
    data_dir='dataset/',
    test_dir='testset/',
    patch_dir='dataset-patches/',
    vocab_dir='vocab/',
    train_filename='train_part.csv',
    test_file_list='test_file_list.csv',
    test_patch_dir='testset-patches/',
    test_filename='test.csv',
    output_dir='output/',
)
