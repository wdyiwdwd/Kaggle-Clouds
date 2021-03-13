#!/usr/bin/env python3
# -*- encoding: utf-8 -*-

import argparse

import cvproj


def prepare_cli():
    parser = argparse.ArgumentParser(description='cv project')
    parser.add_argument('--train', action='store_const', dest='mode', const='train', default=None,
                        help='set program to train phase')
    parser.add_argument('--train-only', action='store_const', dest='mode', const='train-only', default=None,
                        help='set program to train phase(without picture slice generation)')
    parser.add_argument('--test', action='store_const', dest='mode', const='test', default=None,
                        help='set program to test phase')
    parser.add_argument('--eval', action='store_const', dest='mode', const='eval', default=None,
                        help='set program to evaluate trained model')
    parser.add_argument('--image', action='store', dest='image_filename', metavar='IMAGE',
                        help='set image file to test')
    parser.add_argument('--model-type', action='store', dest='model_type', metavar='MODEL',
                        help='set model type to use')
    parser.add_argument('--model-dir', action='store', dest='model_dirname', metavar='MODEL', default='model/',
                        help='set model file to use')
    parser.add_argument('--save-result', action='store_true', dest='save_result',
                        help='save result for test')
    parser.add_argument('--save-masks', action='store_true', dest='save_masks',
                        help='save masks for test')
    parser.add_argument('--save-sized-results', action='store_true', dest='save_sized_results',
                        help='save result for every window size for test')
    parser.add_argument('--save-sized-masks', action='store_true', dest='save_sized_masks',
                        help='save masks for every window size for test')
    parser.add_argument('--div4', action='store_true', dest='div4',
                        help='output coord divided by 4')
    parser.add_argument('--features-init', action='store', dest='features_init', metavar='COUNT', type=int, default=None,
                        help='reuse bag-of-word trained with COUNT words')
    parser.add_argument('--set-data-dir', action='store', dest='data_dir', metavar='DIR', default=None)
    parser.add_argument('--set-test-dir', action='store', dest='test_dir', metavar='DIR', default=None)
    parser.add_argument('--set-patch-dir', action='store', dest='patch_dir', metavar='DIR', default=None)
    parser.add_argument('--set-vocab-dir', action='store', dest='vocab_dir', metavar='DIR', default=None)
    parser.add_argument('--set-train-file', action='store', dest='train_file', metavar='FILE', default=None)
    parser.add_argument('--set-test-file', action='store', dest='test_file', metavar='FILE', default=None)
    parser.add_argument('--set-test-file-list', action='store', dest='test_file_list', metavar='FILE', default=None)
    parser.add_argument('--set-test-patch-dir', action='store', dest='test_patch_dir', metavar='DIR', default=None)
    return parser


def main():
    parser = prepare_cli()
    args = parser.parse_args()

    if args.data_dir:
        cvproj.config.data_dir = args.data_dir
        cvproj.log('data-dir set to ' + cvproj.config.data_dir)
    if args.test_dir:
        cvproj.config.test_dir = args.test_dir
        cvproj.log('test-dir set to ' + cvproj.config.test_dir)
    if args.patch_dir:
        cvproj.config.patch_dir = args.patch_dir
        cvproj.log('patch-dir set to ' + cvproj.config.patch_dir)
    if args.vocab_dir:
        cvproj.config.vocab_dir = args.vocab_dir
        cvproj.log('vocab-dir set to ' + cvproj.config.vocab_dir)
    if args.train_file:
        cvproj.config.train_filename = args.train_file
        cvproj.log('train-filename set to ' + cvproj.config.train_filename)
    if args.test_file:
        cvproj.config.test_filename = args.test_file
        cvproj.log('test-filename set to ' + cvproj.config.test_filename)
    if args.test_file_list:
        cvproj.config.test_file_list = args.test_file_list
        cvproj.log('test-file-list filename set to ' + cvproj.config.test_file_list)
    if args.test_patch_dir:
        cvproj.config.test_patch_dir = args.test_patch_dir
        cvproj.log('test-patch-dir set to ' + cvproj.config.test_patch_dir)

    if args.mode == 'train':
        models = cvproj.train(args.model_type, args.features_init)
        if 'model_dirname' in args:
            cvproj.model.save(models, args.model_dirname)
    elif args.mode == 'train-only':
        models = cvproj.train_only(args.model_type)
        if 'model_dirname' in args:
            cvproj.model.save(models, args.model_dirname)
    elif args.mode == 'test':
        if 'model_dirname' not in args:
            exit(1)
        models = cvproj.model.load(args.model_type, args.model_dirname)
        masks = cvproj.test(args.image_filename, args.features_init, models,
                            args.save_sized_results, args.save_sized_masks, args.div4)
    elif args.mode == 'eval':
        if 'model_dirname' not in args:
            exit(1)
        models = cvproj.model.load(args.model_type, args.model_dirname)
        cvproj.evaluate(args.features_init, models, args.div4)


if __name__ == '__main__':
    main()
