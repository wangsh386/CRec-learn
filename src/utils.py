import logging
import os

import numpy as np
from sklearn.metrics import average_precision_score

logger = logging.getLogger(__name__)


def set_logger(log_dir, displaying=True, saving=True, debug=False):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger()  # get root logger

    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s-%(name)s-%(levelname)s: %(message)s", datefmt="%Y/%m/%d %H:%M:%S"
    )
    if saving:
        file_handler = logging.FileHandler(f"{log_dir}/run.log", mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    if displaying:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)


def print_args(args, str_num=80):
    for arg, val in args.__dict__.items():
        logger.info(arg + "." * (str_num - len(arg) - len(str(val))) + str(val))


def multi_label_metric(prob, gt, ddi_adj, threshold=0.5):
    """
    prob is the output of sigmoid
    gt is a binary matrix
    """

    def jaccard(prob, gt):
        score = []
        for b in range(gt.shape[0]):
            target = np.where(gt[b] == 1)[0]
            predicted = np.where(prob[b] >= threshold)[0]
            inter = set(predicted) & set(target)
            union = set(predicted) | set(target)
            jaccard_score = 0 if union == 0 else len(inter) / len(union)
            score.append(jaccard_score)
        return np.mean(score)

    def precision_auc(pre, gt):
        all_micro = []
        for b in range(gt.shape[0]):
            all_micro.append(average_precision_score(gt[b], pre[b], average="macro"))
        return np.mean(all_micro)

    def prc_recall(prob, gt):
        score_prc = []
        score_recall = []
        for b in range(gt.shape[0]):
            target = np.where(gt[b] == 1)[0]
            predicted = np.where(prob[b] >= threshold)[0]
            inter = set(predicted) & set(target)
            prc_score = 0 if len(predicted) == 0 else len(inter) / len(predicted)
            recall_score = 0 if len(target) == 0 else len(inter) / len(target)
            score_prc.append(prc_score)
            score_recall.append(recall_score)
        return score_prc, score_recall

    def average_f1(prc, recall):
        score = []
        for idx in range(len(prc)):
            if prc[idx] + recall[idx] == 0:
                score.append(0)
            else:
                score.append(2 * prc[idx] * recall[idx] / (prc[idx] + recall[idx]))
        return np.mean(score)

    def ddi_rate_score(medications, ddi_matrix):
        all_cnt = 0
        ddi_cnt = 0
        for sample in medications:
            for i, med_i in enumerate(sample):
                for j, med_j in enumerate(sample):
                    if j <= i:
                        continue
                    all_cnt += 1
                    if ddi_matrix[med_i, med_j] == 1 or ddi_matrix[med_j, med_i] == 1:
                        ddi_cnt += 1
        if all_cnt == 0:
            return 0
        return ddi_cnt / all_cnt

    ja = jaccard(prob, gt)
    prauc = precision_auc(prob, gt)
    prc_ls, recall_ls = prc_recall(prob, gt)
    f1 = average_f1(prc_ls, recall_ls)

    pred = prob.copy()
    pred[pred >= threshold] = 1
    pred[pred < threshold] = 0
    pred_med = [np.where(item)[0] for item in pred]
    ddi = ddi_rate_score(pred_med, ddi_adj)

    return {"ja": ja, "prauc": prauc, "f1": f1, "ddi": ddi}
