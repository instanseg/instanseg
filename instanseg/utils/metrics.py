import numpy as np
import fastremap
import pandas as pd

from stardist import matching
from tqdm import tqdm
from typing import Union

def compute_and_export_metrics(predictions: List[np.ndarray],
                               ground_truths: List[np.ndarray],
                               metrics: List[str],
                               output_dir: str,
                               file_prefix: str = "metrics",
                               iou_threshold: float = 0.5,
                               verbose: bool = True) -> None:
    """
    Compute and export metrics for predictions and ground truths.

    :param predictions: List of prediction arrays.
    :param ground_truths: List of ground truth arrays.
    :param metrics: List of metrics to compute.
    :param output_dir: Directory to save the metrics.
    :param file_prefix: Prefix for the output file names.
    :param iou_threshold: IoU threshold for computing metrics.
    :param verbose: Whether to print verbose output.
    """
    taus = [ 0.5, 0.6, 0.7, 0.8, 0.9]
    stats = [matching.matching_dataset(gt_masks, pred_masks, thresh=t, show_progress=False, by_image = False) for t in tqdm(taus, disable=not show_progress)]
    df_list = []

    for stat in stats:
        df_list.append(pd.DataFrame([stat]))
    df = pd.concat(df_list, ignore_index=True)

    mean_f1 = df[["thresh", "f1"]].iloc[:].mean()["f1"]
    mean_panoptic_quality = df[["thresh", "panoptic_quality"]].iloc[:].mean()["panoptic_quality"]
    panoptic_quality_05 = df[["thresh", "panoptic_quality"]].iloc[0]["panoptic_quality"]
    f1_05 = df[["thresh", "f1"]].iloc[0]["f1"]


    df["mean_f1"] = mean_f1
    df["f1_05"] = f1_05
    df["mean_PQ"] = mean_panoptic_quality
    df["SQ"] = panoptic_quality_05 / f1_05

    if verbose:
        print("Target:",target)
        print("Mean f1 score: ", mean_f1)
        print("f1 score at 0.5: ", f1_05)
        print("SQ: ", panoptic_quality_05 / f1_05)

    if return_metrics:
        return mean_f1, f1_05, panoptic_quality_05 / f1_05

    if output_path is not None:

        df.to_csv(output_path / str(target + "_matching_metrics.csv"))

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
        

        stats = matching.matching_dataset([l for l in labels], [p for p in predicted], thresh=threshold, show_progress = False)
        f1i = [stat.f1 for stat in stats]
        return _robust_f1_mean_calculator(f1i)
    else:
        f1is = [] 
        for i, _ in enumerate(["nuclei", "cells"]):
            labels_tmp = [fastremap.renumber(labels[j][i].detach().cpu().numpy())[0].astype(np.int32) for j, l in enumerate(labels) if labels[j][i].min() >= 0 and labels[j][i].max() > 0]
            predicted_tmp = [fastremap.renumber(predicted[j][i].detach().cpu().numpy())[0].astype(np.int32) for j, l in enumerate(labels) if labels[j][i].min() >= 0 and labels[j][i].max() > 0]

            if len(labels_tmp)==0:
                f1is.append(np.nan)
                continue

            stats = matching.matching_dataset([l for l in labels_tmp], [p for p in predicted_tmp],thresh=threshold, show_progress = False)
            f1i = [stat.f1 for stat in stats]


            f1is.append(_robust_f1_mean_calculator(f1i))

        return f1is
    
