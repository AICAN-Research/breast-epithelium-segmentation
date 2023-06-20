"""
Script to evaluate segmentation results on TMA-level at 10x created at 20x. Builds on eval_tma_cylinders.py
"""
import pandas as pd
import os
import tensorflow as tf
import numpy as np
import h5py
import multiprocessing as mp


# No smoothing when evaluating, to make differenciable during training
def class_dice_(y_true, y_pred, class_val):
    output1 = y_pred[..., class_val]
    gt1 = y_true[..., class_val]

    intersection1 = tf.reduce_sum(output1 * gt1)
    union1 = tf.reduce_sum(output1 * output1) + tf.reduce_sum(
        gt1 * gt1)  # @TODO: why do we need output*output in reduce sum?
    if union1 == 0:
        dice = 1.  # used to be 0 before 28.05.23
        dice_u = True
    else:
        dice = (2. * intersection1) / union1
        dice_u = False

    return dice, dice_u


def class_dice_class_present(y_true, y_pred, class_val):
    count = False

    output1 = y_pred[..., class_val]
    gt1 = y_true[..., class_val]

    intersection1 = tf.reduce_sum(output1 * gt1)
    union1 = tf.reduce_sum(output1 * output1) + tf.reduce_sum(
        gt1 * gt1)  # @TODO: why do we need output*output in reduce sum?
    if union1 == 0:
        dice = 1.  # used to be 0 before 28.05.23
        dice_u = True
    else:
        dice = (2. * intersection1) / union1
        dice_u = False

    if tf.reduce_sum(gt1) or tf.reduce_sum(output1):  # used to be tf.reduce_sum(gt1) before 28.05.23
        count = True

    return dice, count, dice_u


#  @TODO: look at what they do in scikit-learn for edge cases. They set precision/recall to zero if denominator is zero
#  @TODO: by default, but one can set to 1
def precision(y_true, y_pred, object_):
    """
    based on https://github.com/andreped/H2G-Net/blob/main/src/utils/metrics.py
    and network.get_dice_loss()
    :param y_true: true values
    :param y_pred: predicted values
    :param nb_classes: number of classes
    :param use_background: True or False
    :param dims:
    :return: precision: tp / (tp + fp)
    """
    precision_ = 0

    output1 = y_pred[..., object_]
    target1 = y_true[..., object_]

    true_positives = tf.reduce_sum(target1 * output1)
    predicted_positives = tf.reduce_sum(output1)
    if predicted_positives == 0:
        precision_ = 1  # used to be 0 before 28.05.23
    else:
        precision_ += true_positives / predicted_positives

    return precision_


def precision_class_present(y_true, y_pred, object_):
    """
    Only calculate precision when there are positives in y_true
    :param y_true: true values
    :param y_pred: predicted values
    :param nb_classes: number of classes
    :param use_background: True or False
    :param dims:
    :return: precision: tp / (tp + fp), True/False depending on whether there are positives in image
    """
    precision_ = 0
    count = False

    output1 = y_pred[..., object_]
    target1 = y_true[..., object_]

    true_positives = tf.reduce_sum(target1 * output1)
    predicted_positives = tf.reduce_sum(output1)
    if predicted_positives == 0:
        precision_ = 1  # used to be 0 before 28.05.23
    else:
        precision_ += true_positives / predicted_positives
    if tf.reduce_sum(target1) or tf.reduce_sum(output1):  # used to be tf.reduce_sum(target1) before 28.05.23
        count = True

    return precision_, count


def recall(y_true, y_pred, object_):
    """
    based on https://github.com/andreped/H2G-Net/blob/main/src/utils/metrics.py
    and network.get_dice_loss()
    :param y_true: true values
    :param y_pred: predicted values
    :param nb_classes: number of classes
    :param use_background: True or False
    :param dims:
    :return: recall: tp / (tp + fn)
    """
    recall_ = 0

    output1 = y_pred[..., object_]
    target1 = y_true[..., object_]

    true_positives = tf.reduce_sum(
        target1 * output1)  # TODO: consider reduce_sum vs K.sum, is there a difference in speed
    possible_positives = tf.reduce_sum(target1)
    if possible_positives == 0:
        recall_ = 1  # used to be 0 before 28.05.23
    else:
        recall_ += true_positives / possible_positives

    return recall_


def recall_class_present(y_true, y_pred, object_):
    """
    Only calculate recall when there are positives in y_true
    :param y_true: true values
    :param y_pred: predicted values
    :param nb_classes: number of classes
    :param use_background: True or False
    :param dims:
    :return: recall: tp / (tp + fn)
    """
    recall_ = 0
    count = False

    output1 = y_pred[..., object_]
    target1 = y_true[..., object_]

    true_positives = tf.reduce_sum(
        target1 * output1)  # TODO: consider reduce_sum vs K.sum, is there a difference in speed
    possible_positives = tf.reduce_sum(target1)
    if possible_positives == 0:
        recall_ = 1  # used to be 0 before 28.05.23
    else:
        recall_ += true_positives / possible_positives
    if tf.reduce_sum(target1) or tf.reduce_sum(output1):  # used to be tf.reduce_sum(target1) before 28.05.23
        count = True

    return recall_, count


def eval_wrapper(some_inputs_):
    return eval_patch(*some_inputs_)


def eval_patch(path, model):
    import fast

    class PadderPO(fast.PythonProcessObject):
        def __init__(self, width=1024, height=1024):
            super().__init__()
            self.createInputPort(0)
            self.createOutputPort(0)

            self.height = height
            self.width = width

        def execute(self):
            # Get image and invert it with numpy
            image = self.getInputData()
            np_image = np.asarray(image)
            tmp = np.zeros((self.height, self.width, 3), dtype="uint8")
            shapes = np_image.shape
            tmp[:shapes[0], :shapes[1]] = np_image

            # Create new fast image and add as output
            new_output_image = fast.Image.createFromArray(tmp)
            new_output_image.setSpacing(image.getSpacing())
            self.addOutputData(0, new_output_image)

    with h5py.File(path, "r") as f:
        image = np.asarray(f["input"])
        gt = np.asarray(f["output"])

    image = image.astype("uint8")

    data_fast = fast.Image.createFromArray(image)
    generator = fast.PatchGenerator.create(2048, 2048, overlapPercent=0.3).connect(0, data_fast)
    padder = PadderPO.create(width=2048, height=2048).connect(generator)
    network = fast.NeuralNetwork.create(modelFilename=model, inferenceEngine="OpenVINO", scaleFactor=0.00392156862) \
        .connect(padder)
    converter = fast.TensorToSegmentation.create(threshold=0.5).connect(0, network, 5)
    resizer = fast.ImageResizer.create(width=2048, height=2048, useInterpolation=False, preserveAspectRatio=True) \
        .connect(converter)
    stitcher = fast.PatchStitcher.create().connect(resizer)

    for _ in fast.DataStream(stitcher):
        pass

    pred = stitcher.runAndGetOutputData()

    pred = np.asarray(pred)

    del data_fast, generator, padder, network, converter, resizer, stitcher

    gt_shape = gt.shape
    pred = pred[:gt_shape[0], :gt_shape[1]]
    gt = np.argmax(gt, axis=-1).astype("uint8")
    pred = pred[..., 0].astype("uint8")

    # one-hot gt and pred
    gt_back = (gt == 0).astype("float32")
    gt_inv = (gt == 1).astype("float32")
    gt_healthy = (gt == 2).astype("float32")
    gt_inSitu = (gt == 3).astype("float32")
    pred_back = (pred == 0).astype("float32")
    pred_inv = (pred == 1).astype("float32")
    pred_healthy = (pred == 2).astype("float32")
    pred_inSitu = (pred == 3).astype("float32")

    gt = np.stack(
        [gt_back, gt_inv,
         gt_healthy, gt_inSitu], axis=-1)
    pred = np.stack(
        [pred_back, pred_inv,
         pred_healthy, pred_inSitu], axis=-1)

    dice_scores = []
    precisions_ = []
    recalls_ = []
    unions = []

    dice_scores_exist = []
    precisions_exists = []
    recalls_exists = []
    unions_exist = []
    counts_d = []
    counts_p = []
    counts_r = []

    class_names = ["invasive", "benign", "insitu"]
    for i, x in enumerate(class_names):
        c_dice, union_d = class_dice_(gt, pred, class_val=i + 1)
        c_precision = precision(gt, pred, object_=i + 1)
        c_recall = recall(gt, pred, object_=i + 1)
        dice_scores.append(c_dice)  # list of three dices, one for each class
        precisions_.append(c_precision)  # list of three precisions, one for each class
        recalls_.append(c_recall)  # list of three recalls, one for each class
        unions.append(union_d)  # list of three booleans, if true then union is zero and dice set to zero

        c_dice_exist, count_d, union_d_exist = class_dice_class_present(gt, pred, class_val=i + 1)
        c_precision_exist, count_p = precision_class_present(gt, pred, object_=i + 1)
        c_recall_exist, count_r = recall_class_present(gt, pred, object_=i + 1)
        dice_scores_exist.append(c_dice_exist)
        precisions_exists.append(c_precision_exist)
        recalls_exists.append(c_recall_exist)
        unions_exist.append(union_d_exist)
        counts_d.append(count_d)
        counts_p.append(count_p)
        counts_r.append(count_r)

    return np.asarray(dice_scores), np.asarray(precisions_), np.asarray(recalls_), unions, \
           np.asarray(dice_scores_exist), np.asarray(precisions_exists), np.asarray(recalls_exists), unions_exist, \
           counts_d, counts_p, counts_r


def eval_on_dataset():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    plot_flag = False

    path = './datasets_tma_cores/040623_113425_level_1_ds_4/ds_test/'
    model_name = './output/converted_models/model_030623_224255_agunet_bs_8_as_1_lr_0.0005_d_None_bl_1_br_0.3_h_0.05_s_0.3_st_1.0_fl_1.0_rt_1.0_mp_0_ntb_160_nvb_40.onnx'
    dataframe_path = './output/eval/dataframes/'
    name = 'model_' + '030623_224255' + 'ds_' + '040623_113425'

    cylinders_paths = os.listdir(path)
    paths_ = np.array([path + x for x in cylinders_paths]).astype("U400")

    cnt = 0

    dice_scores_total = [[], [], []]
    precisions_total = [[], [], []]
    recalls_total = [[], [], []]
    dice_scores_exist_total = [[], [], []]
    precisions_exists_total = [[], [], []]
    recalls_exists_total = [[], [], []]
    count_stop = 0
    for path in paths_:
        if count_stop == 20:
            continue
        count_stop+=1
        inputs_ = [[path, model_name]]
        p = mp.Pool(1)
        output = p.map(eval_wrapper, inputs_)
        output = output[0]

        dice_scores, precisions_, recalls_, unions, dice_scores_exist, precisions_exists, recalls_exists, \
        unions_exist, counts_d, counts_p, counts_r = output[0], output[1], output[2], output[3], output[4], output[5], \
                                                     output[6], output[7], output[8], output[9], output[10]

        p.terminate()
        p.join()
        del p, inputs_
        cnt += 1

        class_names = ["invasive", "benign", "insitu"]
        for i, x in enumerate(class_names):
            dice_scores_total[i].append(dice_scores[i])
            precisions_total[i].append(precisions_[i])
            recalls_total[i].append(recalls_[i])

            if counts_d[i]:
                dice_scores_exist_total[i].append(dice_scores_exist[i])
            if counts_p[i]:
                precisions_exists_total[i].append(precisions_exists[i])
            if counts_r[i]:
                recalls_exists_total[i].append(recalls_exists[i])

        cnt += 1

    print("cnt: ", cnt)
    mu_1 = np.mean(dice_scores_total[0])
    std_1 = np.std(dice_scores_total[0])
    mu_2 = np.mean(dice_scores_total[1])
    std_2 = np.std(dice_scores_total[1])
    mu_3 = np.mean(dice_scores_total[2])
    std_3 = np.std(dice_scores_total[2])
    p_1 = np.mean(precisions_total[0])
    p_2 = np.mean(precisions_total[1])
    p_3 = np.mean(precisions_total[2])
    r_1 = np.mean(recalls_total[0])
    r_2 = np.mean(recalls_total[1])
    r_3 = np.mean(recalls_total[2])

    mu_1_exist = np.mean(dice_scores_exist_total[0])
    mu_2_exist = np.mean(dice_scores_exist_total[1])
    mu_3_exist = np.mean(dice_scores_exist_total[2])
    p_1_exist = np.mean(precisions_exists_total[0])
    p_2_exist = np.mean(precisions_exists_total[1])
    p_3_exist = np.mean(precisions_exists_total[2])
    r_1_exist = np.mean(recalls_exists_total[0])
    r_2_exist = np.mean(recalls_exists_total[1])
    r_3_exist = np.mean(recalls_exists_total[2])

    print(mu_1)
    print("std_1: ", std_1)
    print("mean precisions invasive: ", p_1)
    print("mean recalls invasive: ", r_1)
    print()
    print(mu_2)
    print("std_2: ", std_2)
    print("mean precisions benign: ", p_2)
    print("mean recalls benign: ", r_2)
    print()
    print(mu_3)
    print("std_3: ", std_3)
    print("mean precisions inSitu: ", p_3)
    print("mean recalls inSitu: ", r_3)

    print("EXISTS: ")
    print(mu_1_exist)
    print("mean precisions invasive exist: ", p_1_exist)
    print("mean recalls invasive exist: ", r_1_exist)
    print()
    print(mu_2_exist)
    print("mean precisions benign exist: ", p_2_exist)
    print("mean recalls benign exist: ", r_2_exist)
    print()
    print(mu_3_exist)
    print("mean precisions inSitu exist: ", p_3_exist)
    print("mean recalls inSitu exist: ", r_3_exist)

    print("COUNT:")
    print(len(dice_scores_exist_total[0]), len(precisions_exists_total[0]), len(recalls_exists_total[0]))
    print(len(dice_scores_exist_total[1]), len(precisions_exists_total[1]), len(recalls_exists_total[1]))
    print(len(dice_scores_exist_total[2]), len(precisions_exists_total[2]), len(recalls_exists_total[2]))

    print()
    print(model_name)
    print(path)

    results = pd.DataFrame(np.array([dice_scores_total[0], dice_scores_total[1], dice_scores_total[2],
                                     precisions_total[0], precisions_total[1], precisions_total[2], recalls_total[0],
                                     recalls_total[1], recalls_total[2], dice_scores_exist_total[0],
                                     dice_scores_exist_total[1], dice_scores_exist_total[2], precisions_exists_total[0],
                                     precisions_exists_total[1], precisions_exists_total[2], recalls_exists_total[0],
                                     recalls_exists_total[1], recalls_exists_total[2]]),
                           index=['dice_total invasive', 'dice_total benign', 'dice_total inSitu',
                                  'precision_total invasive', 'precision_total benign', 'precision_total inSitu',
                                  'recall_total invasive', 'recall_total benign', 'recall_total inSitu',
                                  'dice_exist invasive', 'dice_exist benign', 'dice_exist inSitu',
                                  'precision_exist invasive', 'precision_exist benign', 'precision_exist inSitu',
                                  'recall_exist invasive', 'recall_exist benign', 'recall_exist inSitu'])
    eval_results = pd.DataFrame(np.array([mu_1, std_1, mu_2, std_2, mu_3, std_3, p_1, p_2, p_3, r_1, r_2, r_3,
                                          mu_1_exist, mu_2_exist, mu_3_exist, p_1_exist, p_2_exist, p_3_exist,
                                          r_1_exist, r_2_exist, r_3_exist]),
                                index=['mu_1', 'std_1', 'mu_2', 'std_2', 'mu_3', 'std_3', 'p_1', 'p_2', 'p_3', 'r_1',
                                       'r_2', 'r_3', 'mu_1_exist', 'mu_2_exist', 'mu_3_exist', 'p_1_exist', 'p_2_exist',
                                       'p_3_exist', 'r_1_exist', 'r_2_exist', 'r_3_exist'])

    count = pd.DataFrame(np.array([len(dice_scores_exist_total[0]), len(precisions_exists_total[0]),
                                   len(recalls_exists_total[0]), len(dice_scores_exist_total[1]),
                                   len(precisions_exists_total[1]), len(recalls_exists_total[1]),
                                   len(dice_scores_exist_total[2]), len(precisions_exists_total[2]),
                                   len(recalls_exists_total[2])]), index=['d_e_1', 'p_e_1', 'r_e_1', 'd_e_2', 'p_e_2',
                                                                          'r_e_2', 'd_e_3', 'p_e_3', 'r_e_3'])

    os.makedirs(dataframe_path, exist_ok=True)
    results.to_csv(dataframe_path + name + '/' + 'results' + ".csv")
    eval_results.to_csv(dataframe_path + name + '/' + 'eval_results' + ".csv")
    count.to_csv(dataframe_path + name + '/' + 'count' + ".csv")


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    eval_on_dataset()

