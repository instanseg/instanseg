import numpy as np
import torch
from instanseg.utils.pytorch_utils import torch_fastremap, torch_sparse_onehot, fast_sparse_dual_iou
from collections import namedtuple
import pandas as pd
from tqdm import tqdm

def _check_is_equal(stats1,stats2):
    assert len(stats1) == len(stats2), "Stats length mismatch"
    for s1,s2 in zip(stats1,stats2):
        for k in s1._asdict().keys():
            v1 = s1._asdict()[k]
            v2 = s2._asdict()[k]
            if isinstance(v1, str):
                assert v1 == v2, f"Mismatch in {k}: {v1} vs {v2}"
            elif isinstance(v1, int):
                assert v1 == v2, f"Mismatch in {k}: {v1} vs {v2}"
            elif isinstance(v1, float):
                assert abs(v1 - v2) < 1e-6, f"Mismatch in {k}: {v1} vs {v2}"
            else:
                assert v1 == v2, f"Mismatch in {k}: {v1} vs {v2}"


# Define the same structure as StarDist's Matching
Matching = namedtuple('Matching', [
    'criterion', 'thresh', 'fp', 'tp', 'fn',
    'precision', 'recall', 'accuracy', 'f1',
    'n_true', 'n_pred'])


def stats_at_thresholds(iou_matrix: torch.Tensor, thresholds):

    results = []

    assert min(thresholds) >= 0.5, "Thresholds needs to be greater than 0.5"

    for t in thresholds:
        # Match predictions to GTs where IoU > t
        matches = iou_matrix >= t

        tp = (matches).sum(1).sum().item()
        fp = ((matches).sum(0) == 0).sum().item()
        fn = ((matches).sum(1) == 0).sum().item()
        
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-10)
        accuracy = tp / (tp + fp + fn + 1e-10)

        N_true = iou_matrix.shape[0]
        N_pred = iou_matrix.shape[1]

        stats = Matching(
            criterion='iou',
            thresh=t,
            fp=fp,
            tp=tp,
            fn=fn,
            precision=precision,
            recall=recall,
            accuracy=accuracy,
            f1=f1,
            n_true=N_true,
            n_pred=N_pred,
        )
        results.append(stats)

    return results


def matching_torch(t, p, thresholds):
    if not torch.is_tensor(t):
        t = torch.from_numpy(t)
    if not torch.is_tensor(p):
        p = torch.from_numpy(p)

    assert t.shape[-2:] == p.shape[-2:], "Shape mismatch between ground truth and prediction."

    x = torch_fastremap(t)
    y = torch_fastremap(p)

    if y.max() ==0 or x.max() ==0:
        iou = torch.zeros([len(x[x>0].unique()),len(y[y>0].unique())])

    else:

        x_onehot, _ = torch_sparse_onehot(x, flatten=True)
        y_onehot, _ = torch_sparse_onehot(y, flatten=True)
        iou = fast_sparse_dual_iou(x_onehot, y_onehot)

    return stats_at_thresholds(iou, thresholds)


def matching_dataset_torch(y_true, y_pred, thresh, by_image = False, show_progress = False):
    #inspired by stardist matching

    stats_all = []
    for t, p in zip(y_true, y_pred):
        res = matching_torch(t, p, thresh)
        stats_all.append(res)

    n_images, n_threshs = len(stats_all), len(thresh)
    accumulate = [{} for _ in range(n_threshs)]

    for stats in stats_all:
        for i,s in enumerate(stats):
            acc = accumulate[i]
            for k,v in s._asdict().items():
                if not isinstance(v,str):
                    acc[k] = acc.setdefault(k,0) + v
                

    for thr,acc in zip(thresh,accumulate):

        acc['criterion'] = "iou"
        acc['thresh'] = thr
        acc['by_image'] = bool(by_image)

        if by_image:
            for k in ('precision', 'recall', 'accuracy', 'f1'):
                    acc[k] /= n_images
        else:
            tp, fp, fn = acc['tp'], acc['fp'], acc['fn']
            acc.update(
            precision          = tp / (tp + fp+ 1e-10),
            recall             = tp / (tp + fn+ 1e-10),
            accuracy           = tp / (tp + fp + fn+ 1e-10),
            f1                 = 2 * tp / (2 * tp + fp + fn+ 1e-10),
            )
    
    accumulate = tuple(namedtuple('DatasetMatching',acc.keys())(*acc.values()) for acc in accumulate)

    return accumulate


def test_matching_dataset_torch():

    from stardist import matching
    thresh = torch.arange(0.5, 1.0, 0.01)

    x = torch.randint(0,100,(256,256))
    y = torch.randint(0,10,(256,256))

    out_instanseg = matching_dataset_torch([x,x],[y,x], thresh = thresh, by_image = False)
    out_stardist = matching.matching_dataset([x.numpy(),x.numpy()], [y.numpy(),x.numpy()], thresh = thresh, by_image = False)

    _check_is_equal(out_instanseg,out_stardist)


    x = torch.randint(0,100,(256,256))
    y = torch.randint(0,10,(256,256))

    out_instanseg = matching_dataset_torch([x,x] * 100,[y,x] * 100, thresh = thresh, by_image = True)
    out_stardist = matching.matching_dataset([x.numpy(),x.numpy()] * 100, [y.numpy(),x.numpy()] * 100, thresh = thresh, by_image = True)

    _check_is_equal(out_instanseg,out_stardist)


    x = torch.randint(0,100,(256,256)) 
    y = torch.randint(0,10,(256,256)) *0

    out_instanseg = matching_dataset_torch([x,x],[y,x], thresh = thresh, by_image = False)
    out_stardist = matching.matching_dataset([x.numpy(),x.numpy()], [y.numpy(),x.numpy()], thresh = thresh, by_image = False)

    _check_is_equal(out_instanseg,out_stardist)


use_stardist = False
if use_stardist:
    from stardist.matching import matching_dataset as matching
else:
    matching = matching_dataset_torch

from typing import Union
def _robust_f1_mean_calculator(nan_list: Union[list, np.ndarray]):
    nan_list = np.array(nan_list)
    if len(nan_list) == 0:
        return np.nan
    elif np.isnan(nan_list).all():
        return np.nan
    else:
        return np.nanmean(nan_list)


def _robust_average_precision(labels, predicted, threshold):

    for i in range(len(labels)):
        if labels[i].min() < 0 and not (labels[i] < 0).all():
            labels[i][labels[i] < 0] = 0 #sparse labels
            predicted[i][labels[i] < 0] = 0 

    if labels[0].shape[0] != 2: #cells or nuclei
        labels = [labels[i].detach().cpu().numpy().astype(np.int32) for i, l in enumerate(labels) if labels[i].min() >= 0 and labels[i].max() > 0]
        predicted = [predicted[i].detach().cpu().numpy().astype(np.int32) for i, l in enumerate(labels) if labels[i].min() >= 0 and labels[i].max() > 0]

        if len(labels)==0:
            return np.nan
        
        stats = matching([l for l in labels], [p for p in predicted], thresh=threshold, show_progress = False)
        f1i = [stat.f1 for stat in stats]

        return _robust_f1_mean_calculator(f1i)
    else:
        f1is = [] 
        for i, _ in enumerate(["nuclei", "cells"]):
            labels_tmp = [(labels[j][i].detach().cpu().numpy())[0].astype(np.int32) for j, l in enumerate(labels) if labels[j][i].min() >= 0 and labels[j][i].max() > 0]
            predicted_tmp = [(predicted[j][i].detach().cpu().numpy())[0].astype(np.int32) for j, l in enumerate(labels) if labels[j][i].min() >= 0 and labels[j][i].max() > 0]

            if len(labels_tmp)==0:
                f1is.append(np.nan)
                continue

            stats = matching([l for l in labels_tmp], [p for p in predicted_tmp],thresh=threshold, show_progress = False)
            f1i = [stat.f1 for stat in stats]

            f1is.append(_robust_f1_mean_calculator(f1i))

        return f1is
    


def compute_and_export_metrics(gt_masks, pred_masks, output_path, target, return_metrics = False, show_progress = False, verbose = True):
    
    taus = [ 0.5, 0.6, 0.7, 0.8, 0.9]

    stats = matching(gt_masks, pred_masks, thresh=taus, show_progress=False, by_image = False)
    df_list = []

    for stat in stats:
        df_list.append(pd.DataFrame([stat]))

    df = pd.concat(df_list, ignore_index=True)

    mean_f1 = df[["thresh", "f1"]].iloc[:].mean()["f1"]
    #mean_panoptic_quality = df[["thresh", "panoptic_quality"]].iloc[:].mean()["panoptic_quality"]
    #panoptic_quality_05 = df[["thresh", "panoptic_quality"]].iloc[0]["panoptic_quality"]
    f1_05 = df[["thresh", "f1"]].iloc[0]["f1"]

    df["mean_f1"] = mean_f1
    df["f1_05"] = f1_05
    #df["mean_PQ"] = mean_panoptic_quality
    #df["SQ"] = panoptic_quality_05 / f1_05

    if verbose:
        print("Target:",target)
        print("Mean f1 score: ", mean_f1)
        print("f1 score at 0.5: ", f1_05)
        #print("SQ: ", panoptic_quality_05 / f1_05)

    if return_metrics:
        return mean_f1, f1_05#, panoptic_quality_05 / f1_05

    if output_path is not None:

        df.to_csv(output_path / str(target + "_matching_metrics.csv"))
