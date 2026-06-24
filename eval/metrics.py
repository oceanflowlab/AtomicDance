"""Distribution metrics for AIST++ kinetic and manual motion features."""

from pathlib import Path

import numpy as np
from scipy import linalg
from scipy.spatial.distance import pdist

from eval.eval_bas import calculate_bas


def _load_feature_matrix(root, directory):
    paths = sorted((Path(root) / directory).glob("*.npy"))
    if not paths:
        raise FileNotFoundError("no features found under {}".format(Path(root) / directory))
    arrays = [np.asarray(np.load(str(path)), dtype=np.float64).reshape(-1) for path in paths]
    dimensions = {array.shape for array in arrays}
    if len(dimensions) != 1:
        raise ValueError("inconsistent feature dimensions under {}".format(directory))
    matrix = np.stack(arrays)
    if not np.isfinite(matrix).all():
        raise ValueError("non-finite values found under {}".format(directory))
    return matrix


def normalize_separately(features):
    """Standardize one distribution using its own statistics, as in the starter."""
    features = np.asarray(features, dtype=np.float64)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-10, 1.0, std)
    return (features - mean) / std


def normalize(reference, values):
    """Return independently standardized distributions for compatibility."""
    return normalize_separately(reference), normalize_separately(values)


def calc_fid(generated, ground_truth):
    generated = np.asarray(generated, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    if generated.ndim != 2 or ground_truth.ndim != 2:
        raise ValueError("FID inputs must be matrices")
    if generated.shape[1] != ground_truth.shape[1]:
        raise ValueError("FID feature dimensions do not match")
    if len(generated) < 2 or len(ground_truth) < 2:
        raise ValueError("FID requires at least two samples per distribution")
    mu_gen, mu_gt = generated.mean(axis=0), ground_truth.mean(axis=0)
    sigma_gen = np.atleast_2d(np.cov(generated, rowvar=False))
    sigma_gt = np.atleast_2d(np.cov(ground_truth, rowvar=False))
    covariance, _ = linalg.sqrtm(sigma_gen.dot(sigma_gt), disp=False)
    if not np.isfinite(covariance).all():
        offset = np.eye(sigma_gen.shape[0]) * 1e-5
        covariance = linalg.sqrtm((sigma_gen + offset).dot(sigma_gt + offset))
    if np.iscomplexobj(covariance):
        if not np.allclose(np.diagonal(covariance).imag, 0.0, atol=1e-3):
            raise ValueError("FID covariance has a large imaginary component")
        covariance = covariance.real
    difference = mu_gen - mu_gt
    value = difference.dot(difference) + np.trace(sigma_gen) + np.trace(sigma_gt)
    value -= 2.0 * np.trace(covariance)
    return float(max(value, 0.0))


def calculate_avg_distance(features, mean=None, std=None):
    features = np.asarray(features, dtype=np.float64)
    if mean is not None and std is not None:
        features = (features - mean) / std
    if features.ndim != 2 or len(features) < 2:
        raise ValueError("diversity requires at least two feature vectors")
    return float(pdist(features, metric="euclidean").mean())


def calc_diversity(feats):
    return calculate_avg_distance(feats)


def quantized_metrics(predicted_pkl_root, gt_pkl_root):
    pred_kinetic = _load_feature_matrix(predicted_pkl_root, "kinetic_features")
    pred_manual = _load_feature_matrix(predicted_pkl_root, "manual_features")
    gt_kinetic = _load_feature_matrix(gt_pkl_root, "kinetic_features")
    gt_manual = _load_feature_matrix(gt_pkl_root, "manual_features")

    pred_kinetic = normalize_separately(pred_kinetic)
    pred_manual = normalize_separately(pred_manual)
    gt_kinetic = normalize_separately(gt_kinetic)
    gt_manual = normalize_separately(gt_manual)
    return {
        "num_pred": int(len(pred_kinetic)),
        "num_gt": int(len(gt_kinetic)),
        "fid_k": calc_fid(pred_kinetic, gt_kinetic),
        "fid_m": calc_fid(pred_manual, gt_manual),
        "div_k": calculate_avg_distance(pred_kinetic),
        "div_m": calculate_avg_distance(pred_manual),
        "div_k_gt": calculate_avg_distance(gt_kinetic),
        "div_m_gt": calculate_avg_distance(gt_manual),
        "BAS_pred": calculate_bas(predicted_pkl_root),
        "BAS_gt": calculate_bas(gt_pkl_root),
    }


def calculate_BAS(predicted_pkl_root, gt_pkl_root=None):
    result = {"predict_BAS": calculate_bas(predicted_pkl_root)}
    if gt_pkl_root is not None:
        result["groundtruth_BAS"] = calculate_bas(gt_pkl_root)
    return result
