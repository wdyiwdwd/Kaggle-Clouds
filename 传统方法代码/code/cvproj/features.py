from .utils import *

voc_cluster_cnt = 30
feature_vector_size = 5 + voc_cluster_cnt

# def extract(image): # 暂时不用了
#     features = np.zeros(feature_vector_size)
#
# 提取特征前的处理
#     image_binary = threshold(image)
#     height, width, _ = image.shape
#     area = height * width
#
# 提取特征
#     features[0] = (image_binary != 0).sum() / area
#
#     structElem = cv.getStructuringElement(cv.MORPH_RECT, (3,3))
#     image_eroded = cv.erode(image_binary, structElem)
#     features[1] = (image_eroded != 0).sum() / area
#
#     image_closed = cv.dilate(image_eroded, structElem)
#     features[2] = (image_closed != 0).sum() / area
#
# 提取特征结束
#     return features


extractor = cv.xfeatures2d.SIFT_create()
# bow = cv.BOWImgDescriptorExtractor(extractor, cv.FlannBasedMatcher_create())
flann_index = cv.flann_Index()
bow_image_cnt = None


def voc_file(image_cnt):
    return '%d_%d.pickle' % (image_cnt, voc_cluster_cnt)


def init_with(labeled_images, image_cnt=None):
    global bow_image_cnt
    assert labeled_images is not None or image_cnt is not None, "either to load or to generate, but neither is chosen."
    log('starting to init BoW...')
    bow_image_cnt = image_cnt if image_cnt is not None else len(labeled_images)
    voc_filename = voc_file(bow_image_cnt)
    log('loading vocabulary...')
    voc = load_np(voc_filename, config.vocab_dir)
    if voc is None:
        log('failed to load vocabulary, generating new vocab...')
        bow_kmeans_trainer = cv.BOWKMeansTrainer(voc_cluster_cnt)
        for image_name, _ in labeled_images:
            image = load(image_name, config.train_dir)
            image_gray = grayscale(image)
            _, descriptors = extractor.detectAndCompute(image_gray, None)
            bow_kmeans_trainer.add(descriptors.astype(np.float32))
        voc = bow_kmeans_trainer.cluster()
        mkdir_if_not_exist(config.vocab_dir)
        save_np(voc, voc_filename, config.vocab_dir)
    log('vocabulary shaped ' + repr(voc.shape) + ' is prepared.')
    flann_index.build(voc, {
        'algorithm': 1,  # FLANN_INDEX_KDTREE,
        'trees': 5
    })
    # bow.setVocabulary(voc)
    log('finished init-ing BoW.')


eye = np.eye(voc_cluster_cnt, dtype=np.float32)


def bow_compute_one_hot(descriptors):
    global flann_index
    indices, _ = flann_index.knnSearch(descriptors, 1)
    success = False
    cnt = 0
    while not success:
        try:
            result = eye[indices[:, 0], :]   # BUG: WON'T FIX: FlannIndex数据错误会引起IndexError
            # https://github.com/opencv/opencv/pull/14092 引起bug的似乎是这个问题
            # 因为SIFT要<=3.4.2.16，这个bug是3.4.6解决的，所以只能先这样了
            success = True
        except IndexError:
            if cnt > 10:
                log('sorry but failed to reload BoW. we have carefully set checkpoints so that '
                    'just rerun it and it will resume from last checkpoint...')
                exit(1)
            cnt += 1
            log('flann_index has corrupted. trying to re-init-ing BoW. retry: %d. corrupted data:' % cnt)
            print(str(indices))
            flann_index.release()
            flann_index = cv.flann_Index()  # WON'T FIX: 索引似乎是静态的，这样重新初始化也解决不了
            init_with(None, bow_image_cnt)
    return result


def extract_global(image):
    # 为图片进行局部特征提取所需的全局处理，返回处理结果，作为局部抽取函数的参数传入
    # 可能需要保存成二维位置，方便下面局部提取时切片
    image_binary = threshold(image)
    structElem = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    image_eroded = cv.erode(image_binary, structElem)
    image_dilated = cv.dilate(image_binary, structElem)
    image_closed = cv.dilate(image_eroded, structElem)
    image_opened = cv.erode(image_dilated, structElem)

    image_gray = grayscale(image)
    key_points, descriptors = extractor.detectAndCompute(image_gray, None)
    kp = np.array([kp.pt for kp in key_points])
    one_hots = bow_compute_one_hot(descriptors.astype(np.float32))

    return image_binary != 0, image_eroded != 0, image_dilated != 0, image_closed != 0, image_opened != 0, \
           kp, one_hots


def extract_local(row, col, size, global_features):
    # 为局部提取特征向量
    area = size * size
    # 解包全局提取结果
    image_binary, image_eroded, image_dilated, image_closed, image_opened, \
        key_points, descriptors_one_hot = global_features

    image_binary = image_binary[row: row + size, col: col + size]
    image_eroded = image_eroded[row: row + size, col: col + size]
    image_dilated = image_dilated[row: row + size, col: col + size]
    image_closed = image_closed[row: row + size, col: col + size]
    image_opened = image_opened[row: row + size, col: col + size]
    descriptors_in_range = descriptors_one_hot[
                           np.logical_and(
                               np.logical_and(key_points[:, 0] >= col, key_points[:, 0] < col + size),
                               np.logical_and(key_points[:, 1] >= row, key_points[:, 1] < row + size)
                           ), :
                           ]
    vec = descriptors_in_range.sum(axis=0)

    features = np.ndarray(shape=(feature_vector_size,), dtype=np.float32)

    features[0] = image_binary.sum().astype(np.float32) / area
    features[1] = image_eroded.sum().astype(np.float32) / area
    features[2] = image_dilated.sum().astype(np.float32) / area
    features[3] = image_closed.sum().astype(np.float32) / area
    features[4] = image_opened.sum().astype(np.float32) / area

    norm = np.linalg.norm(vec)
    if norm == 0:
        return None
    vec /= norm
    features[5:5 + voc_cluster_cnt] = vec

    return features
