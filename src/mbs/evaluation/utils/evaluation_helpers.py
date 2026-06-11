from pathlib import Path
import argparse
import json
import time
import yaml

import numpy as np
import scipy.stats

from tqdm.auto import tqdm
import h5py

import torch

try:
    import sklearnex
    sklearnex.patch_sklearn()
    print("Successfully patched sklearn with sklearnex for optimized performance.")
except ImportError:
    print("sklearnex not found. Proceeding without it.")

from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

from tqdm.auto import tqdm
# from tqdm import tqdm

from mbs.core import str2bool
from mbs.metrics import (
    RidgeGCVTorch,
    RepresentationalSimilarityAnalysisTorch,
    CenteredKernelAlignmentTorch,
)
from mbs.metrics import (
    RidgeGCVTorch,
    RepresentationalSimilarityAnalysis,
    CenteredKernelAlignment,
)


def build_stimulus_id_mapping(stimulus_ids):
    mapping = {}
    for idx, stim_id in enumerate(stimulus_ids):
        mapping[stim_id] = idx
        if hasattr(stim_id, "decode"):
            mapping[stim_id.decode("utf-8")] = idx
    return mapping


def deep_update(d, u):
    """
    Recursively updates a dictionary `d` with another dictionary `u`.
    If the value of a key in `u` is a dictionary, the function will recursively update
    the corresponding dictionary in `d`. Otherwise, it will directly update the value
    in `d` with the value from `u`.
    Parameters:
    d (dict): The dictionary to be updated.
    u (dict): The dictionary with updates.
    Returns:
    dict: The updated dictionary `d`.
    """
    
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def load_yaml(yaml_file: str) -> dict:
    """
    Loads a yaml file and returns the content as a dictionary.
    
    Parameters:
    yaml_file (str): The path to the yaml file to be loaded.
    
    Returns:
    dict: The content of the yaml file as a dictionary.
    """
    with open(yaml_file, 'r') as f:
        config = yaml.safe_load(f)
        
    base_config = config.get('base_config', None)
    if base_config is not None:
        yaml_file = Path(yaml_file)
        base_config = yaml_file.parent / base_config
        # Load a base configuration files if exists
        defaults = load_yaml(base_config)
        # Update the defaults with the current configuration
        config = deep_update(defaults, config)

    return config

def compute_metrics(y_true, y_pred, noise_ceiling=None, verbose=True):
    try:
        r2_score_val = r2_score(y_true, y_pred)
        explained_variance_score_val = explained_variance_score(y_true, y_pred)
        mae_val = mean_absolute_error(y_true, y_pred)
        mse_val = mean_squared_error(y_true, y_pred)
        pearsonr_raw_values = scipy.stats.pearsonr(y_true, y_pred, axis=0)[0]
        pearsonr_ncorrected = None
        approx_exp_var_corrected = None
        if noise_ceiling is not None:
            pearsonr_ncorrected_values = pearsonr_raw_values / noise_ceiling
            pearsonr_ncorrected = pearsonr_ncorrected_values.mean()
            approx_exp_var_corrected = (pearsonr_ncorrected_values ** 2).mean()
        pearsonr_val = pearsonr_raw_values.mean()
        approx_exp_var = (pearsonr_raw_values ** 2).mean()
    except Exception as e:
        print(f"Error computing metrics: {e}")
        # print(y_true, y_pred, y_true.shape, y_pred.shape, np.isnan(y_true).any(), np.isnan(y_pred).any())
        print(y_true.shape, y_pred.shape, np.isnan(y_true).any(), np.isnan(y_pred).any())
        raise e
    
    
    if verbose:
        print(f'R2: {r2_score_val:.4f}, '
              f'EVS: {explained_variance_score_val:.4f}, '
              f'MAE: {mae_val:.4f}, '
              f'MSE: {mse_val:.4f}, '
              f'PearsonR: {pearsonr_val:.4f} '
              f'PearsonR (NC): {pearsonr_ncorrected:.4f}' if pearsonr_ncorrected is not None else ''
              
            )

    return r2_score_val, explained_variance_score_val, mae_val, mse_val, pearsonr_val, approx_exp_var, pearsonr_ncorrected, approx_exp_var_corrected

def compute_rsa_cka(X_test, y_test, y_test_pred, X_train=None, y_train=None, y_train_pred=None, verbose=False, use_gpu=False):
    
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    y_test_pred = torch.tensor(y_test_pred, dtype=torch.float32)
    if X_train is not None and y_train is not None and y_train_pred is not None:
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_train_pred = torch.tensor(y_train_pred, dtype=torch.float32)
        
    if use_gpu:
        X_test = X_test.cuda()
        y_test = y_test.cuda()
        y_test_pred = y_test_pred.cuda()
        if X_train is not None and y_train is not None and y_train_pred is not None:
            X_train = X_train.cuda()
            y_train = y_train.cuda()
            y_train_pred = y_train_pred.cuda()
    
    # print(X_test.shape, y_test.shape, y_test_pred.shape)
    # print(type(X_test), type(y_test), type(y_test_pred))
    try:
        
        if use_gpu:
            rsa_metric = RepresentationalSimilarityAnalysisTorch()
            cka_metric = CenteredKernelAlignmentTorch(unbiased=True)
        else:
            rsa_metric = RepresentationalSimilarityAnalysis()
            cka_metric = CenteredKernelAlignment(unbiased=True)
        
        if X_train is not None and y_train is not None and y_train_pred is not None:
            rsa_c_train = rsa_metric(X_train, y_train)
            rsa_ve_train = rsa_metric(y_train_pred, y_train)
        else:
            rsa_c_train = float('nan')
            rsa_ve_train = float('nan')
        
        rsa_c_test = rsa_metric(X_test, y_test)
        rsa_ve_test = rsa_metric(y_test_pred, y_test)
        
        if X_train is not None and y_train is not None and y_train_pred is not None:
            cka_c_train = cka_metric(X_train, y_train)
            cka_ve_train = cka_metric(y_train_pred, y_train)
        else:
            cka_c_train = float('nan')
            cka_ve_train = float('nan')
        cka_c_test = cka_metric(X_test, y_test)
        cka_ve_test = cka_metric(y_test_pred, y_test)
    except Exception as e:
        print(f"Error computing RSA/CKA metrics: {e}")
        raise e

    if verbose:
        print(f'RSA - Train C: {rsa_c_train:.4f}, Train VE: {rsa_ve_train:.4f}, Test C: {rsa_c_test:.4f}, Test VE: {rsa_ve_test:.4f}')
        print(f'CKA - Train C: {cka_c_train:.4f}, Train VE: {cka_ve_train:.4f}, Test C: {cka_c_test:.4f}, Test VE: {cka_ve_test:.4f}')

    return (
        float(rsa_c_train), float(rsa_ve_train), float(rsa_c_test), float(rsa_ve_test),
        float(cka_c_train), float(cka_ve_train), float(cka_c_test), float(cka_ve_test)
    )


def pearsonr_score(y_true, y_pred, noise_ceiling=None):
    pearsonr_raw_values = scipy.stats.pearsonr(y_true, y_pred, axis=0)[0]
    if noise_ceiling is not None:
        pearsonr_ncorrected_values = pearsonr_raw_values / noise_ceiling
        return pearsonr_ncorrected_values.mean()
    return pearsonr_raw_values.mean()



def load_layer_features(layer_name: str, features_folder: Path=None, features_file_path: Path=None):
    assert features_folder is not None or features_file_path is not None, \
        "Either features_folder or features_file_path must be provided."
    assert not (features_folder is not None and features_file_path is not None), \
        "Only one of features_folder or features_file_path should be provided."
    
    

    if features_folder:
        h5_files = sorted(features_folder.glob('*.h5'))
        assert len(h5_files) > 0, f"No h5 files found in features folder: {features_folder}"
        
        layer_feats = []
        stimulus_ids = []
        for h5_file in tqdm(h5_files, desc=f'Loading features for layer {layer_name}', leave=False):
            with h5py.File(h5_file, 'r') as f:
                layer_key = f'features/{layer_name.replace(".", "-")}'
                layer_feats.append(f[layer_key][:])
                stimulus_ids.extend(f['ids'][:])
        layer_feats = np.concatenate(layer_feats, axis=0)
    else:
        # Load from single h5 file
        h5_file = Path(features_file_path)
        assert h5_file.exists(), f"No h5 file found at: {features_file_path}"
        with h5py.File(h5_file, 'r') as f:
            layer_key = f'features/{layer_name.replace(".", "-")}'
            layer_feats = f[layer_key][:]
            stimulus_ids = f['ids'][:]
    
    stimulus_ids_mapping = build_stimulus_id_mapping(stimulus_ids)

    return layer_feats, stimulus_ids_mapping


def load_neural_metadata(data_path: Path):
    
    with h5py.File(data_path, 'r') as f:
        subjects = f.attrs['subjects']
        rois = f.attrs['rois']
        splits = f.attrs['splits']
        nc_max = f.attrs.get('max_nc', 100.0)

    return subjects, rois, splits, nc_max

def load_neural_data(data_path: Path, subject:str, roi:str, split:str):
    with h5py.File(data_path, 'r') as f:
        subject_specific_stimulus_set = f.attrs.get('subject_specific_stimulus_set', False)
        if subject_specific_stimulus_set:
            stimulus_ids = f[split]['stimulus_ids'][subject][()]
        else:
            stimulus_ids = f[split]['stimulus_ids'][()]
        nc_max = f.attrs.get('max_nc', 100.0)
        neural_data = f[split]['neural_data'][subject][roi][()]
        noise_ceiling = f['noise_ceilings'][subject][roi][()]/nc_max + 1e-6  # to avoid division by zero
        noise_ceiling = np.sqrt(noise_ceiling) # convert variance explained to correlation coefficient
    return stimulus_ids, neural_data, noise_ceiling


ALPHA_LIST = [
    *[0.01, 0.1, 0.5, 1.0],
    *np.linspace(1e1, 1e2, 4, endpoint=False),
    *np.linspace(1e2, 1e3, 18, endpoint=False),
    *np.linspace(1e3, 1e4, 18, endpoint=False),
    *np.linspace(1e4, 1e5, 18, endpoint=False),
    *np.linspace(1e5, 1e6, 18, endpoint=False),
    *np.linspace(1e6, 1e7, 19)
]

ALPHA_LIST_SHORT = np.logspace(1, 7, 10)


def get_pipeline(use_gpu: bool = False, alphas: np.ndarray = ALPHA_LIST_SHORT, use_wide_range_alphas: bool = True):
    if use_wide_range_alphas:
        print("Using wide range of alphas for Ridge regression.")
        alphas = ALPHA_LIST
    if use_gpu:
        regressor = RidgeGCVTorch(alphas=alphas, pbar=False, dtype=torch.float32, store_results_gpu=False)
    else:
        regressor = RidgeCV(alphas=alphas)

    pipeline = Pipeline([
        # ('scaler', StandardScaler()),
        # ('pca', pca),
        # ('rp', random_projection),
        ('regressor', regressor)
    ])
    return pipeline

def save_results(output_file_path: Path, config: dict, results: list):
    data = {
        'config': {k: v for k, v in config.items() if k != 'layer_id'}, # Exclude layer_id from saved config
        'results': results
    }
    with open(output_file_path, 'w') as f:
        json.dump(data, f, indent=2)
