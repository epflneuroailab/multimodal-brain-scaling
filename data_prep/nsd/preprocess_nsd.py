import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict
import json

import numpy as np
import pandas as pd
import h5py
from scipy.stats import zscore
import nibabel as nib
from tqdm.auto import tqdm

# ==========================================
# Constants & Configuration
# ==========================================

SUBJECT_TEMPLATE = "subj{idx:02d}"

ROI_FILES = [
    "streams",
    "prf-visualrois",
    "nsdgeneral",
    "floc-words",
    "floc-places",
    "floc-faces",
    "floc-bodies"
]

# Combined ROI mapping logic
ROI_MAPPING = {
    "V1": ["V1d", "V1v"],
    "V2": ["V2d", "V2v"],
    "V3": ["V3d", "V3v"],
    "V4": ["hV4"],
    "IT": [
        "midlateral",
        "midparietal",
        "midventral",
        "parietal",
        "lateral",
        "ventral"
    ],
    "early_vision": [
        "early"
    ],
    "mid_vision": [
        "midlateral",
        "midparietal",
        "midventral",  
    ],
    "high_vision": [
        "parietal",
        "lateral",
        "ventral"
    ],
    "category_selective" : [
        "OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words",
        "OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces",
        "EBA", "FBA-1", "FBA-2", "mTL-bodies",
        "OPA", "PPA", "RSC",
    ],
    "VWFA": ["OWFA", "VWFA-1", "VWFA-2", "mfs-words", "mTL-words"],
    "faces": ["OFA", "FFA-1", "FFA-2", "mTL-faces", "aTL-faces"],
    "bodies": ["EBA", "FBA-1", "FBA-2", "mTL-bodies"],
    "places": ["OPA", "PPA", "RSC"],
    "vision": ["nsdgeneral"]
}

SKIP_ROIS = [
    "mfs-words", "mTL-words",
    "mTL-faces", "aTL-faces",
    "mTL-bodies",
]
# SKIP_ROIS = []

BETA_TYPE = "betas_fithrf_GLMdenoise_RR"
NOISE_CEILING_THRESHOLD = 10

# ==========================================
# Helper Functions
# ==========================================

def ncsnr2nc(x: np.ndarray) -> np.ndarray:
    """Convert NCSNR to % noise ceiling."""
    return 100.0 * (x ** 2) / (x ** 2 + 1.0 / 3.0)

def load_ctab(file_path: Path) -> pd.DataFrame:
    """Read FreeSurfer-style .ctab files with ROI id / name."""
    ctab = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        comment="#",
        names=["roi_id", "roi_name"],
    )
    return ctab


def load_noise_ceiling(
    ds_dir: Path,
    subject: str,
    data_space: str,
    beta_type: str,
) -> np.ndarray:
    """
    Load noise ceiling for a subject in either volumetric or surface space.

    Returns:
        - func1pt8mm: 3D array (X, Y, Z)
        - nativesurface: 1D array (N_verts,)
        - fsaverage: 1D array (N_verts,)
    """
    if data_space == "func1pt8mm":
        nc_path = (
            ds_dir
            / "nsddata_betas"
            / "ppdata"
            / subject
            / data_space
            / beta_type
            / "ncsnr.nii.gz"
        )
        nc = nib.load(str(nc_path)).get_fdata()
        nc = np.asarray(nc, dtype=np.float32)
        return ncsnr2nc(nc)

    elif data_space == "nativesurface":
        base = ds_dir / "nsddata_betas" / "ppdata" / subject / data_space / beta_type
        nc_lh_path = base / "lh.ncsnr.mgh"
        nc_rh_path = base / "rh.ncsnr.mgh"

        nc_lh = np.squeeze(nib.load(str(nc_lh_path)).get_fdata())
        nc_rh = np.squeeze(nib.load(str(nc_rh_path)).get_fdata())
        nc = np.concatenate([nc_lh, nc_rh], axis=0)
        nc = np.asarray(nc, dtype=np.float32)
        return ncsnr2nc(nc)
    elif data_space == "fsaverage":
        base = ds_dir / "nsddata_betas" / "ppdata" / subject / data_space / beta_type
        nc_lh_path = base / "lh.ncsnr.mgh"
        nc_rh_path = base / "rh.ncsnr.mgh"

        nc_lh = np.squeeze(nib.load(str(nc_lh_path)).get_fdata())
        nc_rh = np.squeeze(nib.load(str(nc_rh_path)).get_fdata())
        nc = np.concatenate([nc_lh, nc_rh], axis=0)
        nc = np.asarray(nc, dtype=np.float32)
        return ncsnr2nc(nc)

    else:
        raise ValueError(f"Unsupported data_space: {data_space}")
    
def build_roi_raw_masks(
    ds_dir: Path,
    subject: str,
    data_space: str,
) -> Dict[str, np.ndarray]:
    """
    Build per-ROI boolean masks in either volume or surface space.

    Returns:
        dict roi_name -> bool array:
            - func1pt8mm: shape (X, Y, Z)
            - nativesurface: shape (N_verts,)
            - fsaverage: shape (N_verts,)
    """
    roi_files = ROI_FILES
    roi_raw_masks: Dict[str, np.ndarray] = {}

    for meta_roi in roi_files:
        if data_space == "func1pt8mm":
            roi_meta_mask_path = (
                ds_dir
                / "nsddata"
                / "ppdata"
                / subject
                / "func1pt8mm"
                / "roi"
                / f"{meta_roi}.nii.gz"
            )
            roi_meta_mask = np.squeeze(
                nib.load(str(roi_meta_mask_path)).get_fdata()
            )
        elif  data_space == "nativesurface":
            base = ds_dir / "nsddata" / "freesurfer" / subject / "label"
            roi_meta_mask_left = np.squeeze(
                nib.load(str(base / f"lh.{meta_roi}.mgz")).get_fdata()
            )
            roi_meta_mask_right = np.squeeze(
                nib.load(str(base / f"rh.{meta_roi}.mgz")).get_fdata()
            )
            roi_meta_mask = np.concatenate(
                [roi_meta_mask_left, roi_meta_mask_right],
                axis=0,
            )
        elif data_space == "fsaverage":
            base = ds_dir / "nsddata" / "freesurfer" / "fsaverage" / "mapped_labels" / subject
            roi_meta_mask_left = np.squeeze(
                nib.load(str(base / f"lh.{meta_roi}.mgz")).get_fdata()
            )
            roi_meta_mask_right = np.squeeze(
                nib.load(str(base / f"rh.{meta_roi}.mgz")).get_fdata()
            )
            roi_meta_mask = np.concatenate(
                [roi_meta_mask_left, roi_meta_mask_right],
                axis=0,
            )

        ctab_path = ds_dir / "nsddata" / "freesurfer" / subject / "label" / f"{meta_roi}.mgz.ctab"
        metadata = load_ctab(ctab_path)

        # Skip first row which just has number of entries
        for _, row in metadata.iloc[1:].iterrows():
            roi_name = str(row.roi_name)
            roi_id = int(row.roi_id)
            roi_mask = roi_meta_mask == roi_id
            roi_raw_masks[roi_name] = roi_mask

    # Whole-brain mask (for convenience)
    any_mask = next(iter(roi_raw_masks.values()))
    roi_raw_masks["whole_brain"] = np.ones_like(any_mask, dtype=bool)

    return roi_raw_masks




def build_roi_masks(
    roi_raw_masks: Dict[str, np.ndarray],
    noise_ceiling: np.ndarray,
    nc_threshold: float,
    roi_mapping: Dict[str, List[str]],
    apply_nsdgeneral_mask: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Combine fine-grained ROIs into higher-level ROIs and apply noise ceiling threshold.

    Returns:
        roi_masks: dict[roi_name] -> bool mask
        roi_masks_nc: dict[roi_name] -> bool mask & (NC >= threshold)
    """
    roi_masks: Dict[str, np.ndarray] = defaultdict(lambda: None)
    roi_masks_nc: Dict[str, np.ndarray] = defaultdict(lambda: None)

    nc_mask = noise_ceiling > nc_threshold

    for roi_name, roi_list in roi_mapping.items():
        mask = np.zeros_like(roi_raw_masks["nsdgeneral"], dtype=bool)
        for roi in roi_list:
            mask |= roi_raw_masks[roi]

        if apply_nsdgeneral_mask:
            # Apply nsdgeneral mask to restrict to visually responsive cortex
            if roi_name != "RSC":  # No overlap with nsdgeneral for RSC
                mask &= roi_raw_masks["nsdgeneral"]

        roi_masks[roi_name] = mask
        roi_masks_nc[roi_name] = mask & nc_mask

    # Whole brain ROI
    roi_masks["whole_brain"] = np.ones_like(roi_raw_masks["nsdgeneral"], dtype=bool)
    roi_masks_nc["whole_brain"] = roi_masks["whole_brain"] & nc_mask

    return roi_masks, roi_masks_nc


# ==========================================
# Helpers: betas loading
# ==========================================


def load_betas_volume(
    ds_dir: Path,
    subject: str,
    beta_type: str,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Load 4D betas for a subject in func1pt8mm space.

    Returns:
        np.ndarray of shape (n_trials, n_x, n_y, n_z)
    """
    sub_data_dir = (
        ds_dir
        / "nsddata_betas"
        / "ppdata"
        / subject
        / "func1pt8mm"
        / beta_type
    )
    func_data_paths = sorted(sub_data_dir.glob("betas_session*.hdf5"))

    all_trials: List[np.ndarray] = []
    for path in tqdm(func_data_paths, desc=f"{subject} volume sessions", leave=False):
        with h5py.File(path, "r") as f:
            func_data = f["betas"][:]  # (750, Z, Y, X)
            func_data = func_data.transpose((0, 3, 2, 1))  # (750, X, Y, Z)

        func_data = func_data.astype(dtype) / 300.0
        func_data = zscore(func_data, axis=0, ddof=1)
        all_trials.append(func_data)
        
    print("Loaded volume data, concatenating sessions...")
    all_trials = np.concatenate(all_trials, axis=0)
    print("Concatenated volume data shape:", all_trials.shape)

    return all_trials # (n_trials, n_x, n_y, n_z)



def load_betas_nativesurface(
    ds_dir: Path,
    subject: str,
    beta_type: str,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Load 2D betas for a subject in nativesurface space.

    Returns:
        np.ndarray of shape (n_trials, n_vertices)
    """
    sub_data_dir = (
        ds_dir
        / "nsddata_betas"
        / "ppdata"
        / subject
        / "nativesurface"
        / beta_type
    )
    lh_paths = sorted(sub_data_dir.glob("lh.betas_session*.hdf5"))
    rh_paths = sorted(sub_data_dir.glob("rh.betas_session*.hdf5"))

    assert len(lh_paths) == len(rh_paths), "LH/RH sessions mismatch"

    all_trials: List[np.ndarray] = []
    for lh_path, rh_path in tqdm(
        list(zip(lh_paths, rh_paths)),
        desc=f"{subject} surface sessions",
        leave=False,
    ):
        with h5py.File(lh_path, "r") as f_lh, h5py.File(rh_path, "r") as f_rh:
            lh_data = f_lh["betas"][:]  # (750, N_lh)
            rh_data = f_rh["betas"][:]  # (750, N_rh)

        surface_data = np.concatenate([lh_data, rh_data], axis=1)  # (750, N_vert)
        surface_data = surface_data.astype(dtype) / 300.0
        surface_data = zscore(surface_data, axis=0, ddof=1)
        all_trials.append(surface_data)

    print("Loaded surface data, concatenating sessions...")
    all_trials = np.concatenate(all_trials, axis=0)
    print("Concatenated surface data shape:", all_trials.shape)

    return all_trials # (n_trials, n_vertices)

def load_betas_fsaverage(
    ds_dir: Path,
    subject: str,
    beta_type: str,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Load 2D betas for a subject in fsaverage space.

    Returns:
        np.ndarray of shape (n_trials, n_vertices)
    """
    sub_data_dir = (
        ds_dir
        / "nsddata_betas"
        / "ppdata"
        / subject
        / "fsaverage"
        / beta_type
    )
    lh_paths = sorted(sub_data_dir.glob("lh.betas_session*.mgh"))
    rh_paths = sorted(sub_data_dir.glob("rh.betas_session*.mgh"))

    assert len(lh_paths) == len(rh_paths), "LH/RH sessions mismatch"

    all_trials: List[np.ndarray] = []
    for lh_path, rh_path in tqdm(
        list(zip(lh_paths, rh_paths)),
        desc=f"{subject} surface sessions",
        leave=False,
    ):
        lh_data = nib.load(str(lh_path)).get_fdata().squeeze()  # (163842, 750)
        rh_data = nib.load(str(rh_path)).get_fdata().squeeze()  # (163842, 750)

        surface_data = np.concatenate([lh_data.T, rh_data.T], axis=1)  # (750, 327684)
        # surface_data = surface_data.astype(dtype) / 300.0
        surface_data = zscore(surface_data, axis=0, ddof=1)
        all_trials.append(surface_data)

    print("Loaded fsaverage surface data, concatenating sessions...")
    all_trials = np.concatenate(all_trials, axis=0)
    print("Concatenated surface data shape:", all_trials.shape)

    return all_trials # (n_trials, n_vertices)



# ==========================================
# Helpers: stimuli handling
# ==========================================
def split_stimuli_by_reps(df_stimuli: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split stimulus presentations into two DataFrames based on repetition count per nsdId.

    Parameters:
        df_stimuli (pd.DataFrame): Long-form DataFrame of presentations with at least:
            - 'nsdId' (int): stimulus identifier
            - 'rep' (int): repetition index for that presentation

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - stimuli_with_all_reps: subset of rows whose nsdId appears exactly 3 times
            - stimuli_fewer_reps: subset of rows whose nsdId appears fewer than 3 times

    Notes:
        - Assumes up to 3 planned repetitions per stimulus and counts rows per 'nsdId'.
        - The returned DataFrames preserve the original columns and row order of the input.
        - To get unique stimulus IDs per set, use `.nsdId.unique()` on the returned DataFrames.
    """
    stimuli_with_all_reps = []
    stimuli_missing_reps = []
    for group, group_data in df_stimuli.groupby('nsdId'):
        if group_data.shape[0] == 3:
            stimuli_with_all_reps.append(group)
        else:
            stimuli_missing_reps.append(group)
    stimuli_with_all_reps = df_stimuli[df_stimuli.nsdId.isin(stimuli_with_all_reps)]
    stimuli_missing_reps = df_stimuli[df_stimuli.nsdId.isin(stimuli_missing_reps)]
    return stimuli_with_all_reps, stimuli_missing_reps


def process_trials_with_missing_reps(
    stimuli_missing_reps: pd.DataFrame,
    subject_brain_data: np.ndarray
) -> List[np.ndarray]:
    """
    Process trials with missing repetitions by averaging available brain data per stimulus.

    Parameters:
        stimuli_missing_reps (pd.DataFrame): DataFrame of presentations with missing reps, must include:
            - 'nsdId' (int): stimulus identifier
            - 'trial' (int): trial index corresponding to brain data columns
        subject_brain_data (np.ndarray): Brain data array of shape (n_x, n_y, n_z, n_trials) or (n_vertices, n_trials).

    Returns:
        np.ndarray: Array of shape (n_voxels, n_stimuli) containing averaged brain data for each stimulus.
    """
    data_missing_reps = []
    for nsdId, group_data in tqdm(stimuli_missing_reps.groupby('nsdId'), leave=False, desc=""):
        brain_data_stimulus = subject_brain_data[..., group_data.index.values].mean(axis=-1)
        data_missing_reps.append(brain_data_stimulus)
    
    return np.stack(data_missing_reps, axis=-1)


def process_sub_data_split(subject_brain_data: np.ndarray, df_stimuli: pd.DataFrame, keep_reps: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process subject brain data by splitting stimuli into those with all repetitions and those with missing reps.

    Parameters:
        subject_brain_data (np.ndarray): Brain data array of shape (n_x, n_y, n_z, n_trials) or (n_vertices, n_trials).
        df_stimuli (pd.DataFrame): DataFrame of presentations with at least:
            - 'nsdId' (int): stimulus identifier
            - 'trial' (int): trial index corresponding to brain data columns
            - 'rep' (int): repetition index for that presentation
        keep_reps (bool, optional): If True, do not average repetitions for stimuli with all reps. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Combined brain data array and unique stimulus IDs.
            np.ndarray: Combined brain data array of shape (n_voxels, n_stimuli) for all stimuli.
            np.ndarray: Array of unique stimulus IDs for all stimuli.
    """
    stimuli_with_all_reps, stimuli_missing_reps = split_stimuli_by_reps(df_stimuli)
    if len(stimuli_missing_reps)>0 and keep_reps:
        raise ValueError("Cannot keep repetitions when there are stimuli with missing repetitions.")

    stimulus_ids_all_reps = np.sort(stimuli_with_all_reps['nsdId'].unique())
    stimulus_ids_missing_reps = np.sort(stimuli_missing_reps['nsdId'].unique())
    stimulus_ids = np.concatenate([stimulus_ids_all_reps, stimulus_ids_missing_reps])

    # Process stimuli with all repetitions
    subject_data_all_reps = subject_brain_data[..., stimuli_with_all_reps.index.values]
    subject_data_all_reps = subject_data_all_reps.reshape(
        *subject_data_all_reps.shape[:-1],
        -1, 3
    )
    if not keep_reps:
        subject_data_all_reps = subject_data_all_reps.mean(axis=-1)

    # Process stimuli with missing repetitions
    if len(stimuli_missing_reps) > 0:
        subject_data_missing_reps = process_trials_with_missing_reps(
            stimuli_missing_reps,
            subject_brain_data
        )
    else:
        if keep_reps:
            subject_data_missing_reps = np.empty((*subject_data_all_reps.shape[:-2], 0, subject_data_all_reps.shape[-1]))
        else:
            subject_data_missing_reps = np.empty((*subject_data_all_reps.shape[:-1], 0))

    # Combine both sets of processed data
    if keep_reps:
        combined_data = np.concatenate([subject_data_all_reps, subject_data_missing_reps], axis=-2)
    else:
        combined_data = np.concatenate([subject_data_all_reps, subject_data_missing_reps], axis=-1)
    return combined_data, stimulus_ids

def get_subject_stimuli(df_stimuli: pd.DataFrame, subject_id: int = 1) -> pd.DataFrame:
    """
    Return a copy of rows for a single subject.

    Parameters:
        df_stimuli (pd.DataFrame): Long-form DataFrame containing stimulus presentations.
            Must include column:
            - 'subject' (int): subject identifier (1-8 for NSD).
        subject_id (int, optional): Subject to select. Defaults to 1.

    Returns:
        pd.DataFrame: Copy of df_stimuli filtered to the given subject.
    """
    subject_stimuli = df_stimuli[df_stimuli.subject == subject_id].copy()
    return subject_stimuli


def get_valid_stimuli(
    subject_brain_data: np.ndarray,
    df_stimuli: pd.DataFrame
) -> pd.DataFrame:
    """
    Filter presentations to those with a trial index within the completed scans.

    Parameters:
        subject_brain_data (np.ndarray): Brain data array of shape (n_x, n_y, n_z, n_trials) or (n_vertices, n_trials).
        df_stimuli (pd.DataFrame): Presentations DataFrame with at least:
            - 'trial' (int): 1-based trial index aligned to brain data columns.
            - 'nsdId' (int): stimulus identifier.
            - 'rep' (int): repetition index.

    Returns:
        pd.DataFrame: Copy of valid rows (trial <= completed n_trials), index reset and
        sorted by ['nsdId', 'rep'].
    """
    completed_trials = subject_brain_data.shape[-1]

    valid_stimuli = df_stimuli[df_stimuli.trial <= completed_trials].copy()
    valid_stimuli.reset_index(drop=True, inplace=True)
    valid_stimuli.sort_values(['nsdId', 'rep'], inplace=True)
    return valid_stimuli


def get_stimuli_train_test_splits(df_stimuli: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split presentations into train/test by the NSD shared1000 set.

    Parameters:
        df_stimuli (pd.DataFrame): Presentations DataFrame with:
            - 'shared1000' (bool): True if the stimulus belongs to the shared1000 test set.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - sub_stim_train: copy of rows with shared1000 == False
            - sub_stim_test: copy of rows with shared1000 == True
    """
    test_mask = df_stimuli["shared1000"] == True
    sub_stim_test = df_stimuli[test_mask].copy()
    sub_stim_train = df_stimuli[~test_mask].copy()
    return sub_stim_train, sub_stim_test


def process_subject_data(
    subject_brain_data: np.ndarray,
    df_stimuli: pd.DataFrame,
    keep_reps: bool = False,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing for a single subject (per ROI):

    - filters invalid trials
    - splits into train/test by shared1000
    - averages repetitions (unless keep_reps=True)
    """
    df_stimuli = get_valid_stimuli(subject_brain_data, df_stimuli)
    df_stimuli_train, df_stimuli_test = get_stimuli_train_test_splits(df_stimuli)

    subject_data_train, stimulus_ids_train = process_sub_data_split(
        subject_brain_data,
        df_stimuli_train,
        keep_reps=keep_reps,
    )
    subject_data_test, stimulus_ids_test = process_sub_data_split(
        subject_brain_data,
        df_stimuli_test,
        keep_reps=keep_reps,
    )

    return subject_data_train, subject_data_test, df_stimuli_train, df_stimuli_test, stimulus_ids_train, stimulus_ids_test




# ==========================================
# Main pipeline per subject
# ==========================================

def process_single_subject(
    subject: str,
    ds_dir: Path,
    data_space: str,
    beta_type: str,
    df_stim_all: pd.DataFrame,
    nc_threshold: float,
    keep_reps: bool,
    roi_mapping_type: str,
    out_path: Path,
    dtype: str,
    debug_mode: bool = False,
    apply_nsdgeneral_mask: bool = False,
    args: argparse.Namespace = None,
) -> None:
    """
    Full pipeline for a single subject and data_space:

    - load noise ceiling & ROI masks
    - extract ROI time series from betas
    - average repetitions & split into train/test
    - append to an output HDF5 file
    """
    subject_num = int(subject.replace("subj", ""))
    print(f"=== Processing {subject} ({data_space}) ===")

    # 1) Noise ceiling + ROIs
    noise_ceiling = load_noise_ceiling(ds_dir, subject, data_space, beta_type)
    roi_raw_masks = build_roi_raw_masks(ds_dir, subject, data_space)
    if roi_mapping_type == "individual":
        roi_mapping = {roi: [roi] for roi in roi_raw_masks.keys() if roi not in SKIP_ROIS}
    elif roi_mapping_type == "combined":
        roi_mapping = ROI_MAPPING
    else:
        raise ValueError(f"Unsupported roi_mapping_type: {roi_mapping_type}")
    print(f"Using ROI mapping: {roi_mapping}")
    
    roi_masks, roi_masks_nc = build_roi_masks(
        roi_raw_masks,
        noise_ceiling=noise_ceiling,
        nc_threshold=nc_threshold,
        roi_mapping=roi_mapping,
        apply_nsdgeneral_mask=apply_nsdgeneral_mask,
    )
    
    print(f"ROIs to be processed: {list(roi_masks_nc.keys())}")
    for roi_name, roi_mask in roi_masks_nc.items():
        n_voxels = np.sum(roi_mask)
        print(f"  - {roi_name}: {n_voxels}/{roi_masks[roi_name].sum()} units (after/before NC thresholding)")

    # 2) Stimulus table for this subject
    df_stim_sub = get_subject_stimuli(df_stim_all, subject_num)

    # 3) For each ROI, load betas, do averaging + train/test split, and write to disk
    if not out_path.parent.exists():
        out_path.parent.mkdir(parents=True, exist_ok=False)
        
    # 3.1) Load ROI voxels (trials x voxels_x x voxels_y x voxels_z) or (trials x vertices)
    dtype_np = np.dtype(dtype)
    
    if debug_mode:
        # Skip loading full data in debug mode
        neural_data = np.zeros((30_000, *noise_ceiling.shape), dtype=dtype_np)
    else:
        if data_space == "func1pt8mm":
            neural_data = load_betas_volume(
                ds_dir,
                subject,
                beta_type,
                dtype=dtype_np
            )
        elif data_space == "nativesurface":
            neural_data = load_betas_nativesurface(
                ds_dir,
                subject,
                beta_type,
                dtype=dtype_np
            )
        elif data_space == "fsaverage":
            neural_data = load_betas_fsaverage(
                ds_dir,
                subject,
                beta_type,
                dtype=dtype_np
            )
        else:
            raise ValueError(f"Unsupported data_space: {data_space}")
    print("Loaded neural data shape:", neural_data.shape)
    
    # Some voxels/vertices may be NaN after z-scoring; add those to the noise ceiling mask
    if data_space == "func1pt8mm":
        nan_mask = np.isnan(neural_data).any(axis=0)  # (X, Y, Z)
    elif data_space == "nativesurface":
        nan_mask = np.isnan(neural_data).any(axis=0)  # (N_verts,)
    elif data_space == "fsaverage":
        nan_mask = np.isnan(neural_data).any(axis=0)  # (N_verts,)
    if nan_mask.sum() > 0:
        print(f"Found {nan_mask.sum()} voxels/vertices with NaN values after z-scoring; masking them out.")
        noise_ceiling[nan_mask] = 0.0
        for roi_name in roi_masks.keys():
            roi_masks[roi_name][nan_mask] = False
        for roi_name in roi_masks_nc.keys():
            roi_masks_nc[roi_name][nan_mask] = False

    # 3.2) Average repetitions & split into train/test
    print("Processing subject data (averaging reps & train/test split)...")
    if data_space == "func1pt8mm":
        # Transpose to (v_x, v_y, v_z, trials)
        subject_brain_data = neural_data.transpose(1, 2, 3, 0)
        data_train, data_test, df_train, df_test, stimulus_ids_train, stimulus_ids_test = process_subject_data(
            subject_brain_data,
            df_stim_sub,
            keep_reps=keep_reps,
        )
        if keep_reps:
            # Transpose to (trials, reps, v_x, v_y, v_z)
            data_train = data_train.transpose(3, 4, 0, 1, 2)
            data_test = data_test.transpose(3, 4, 0, 1, 2)
        else:
            # Transpose to (trials, v_x, v_y, v_z)
            data_train = data_train.transpose(3, 0, 1, 2)
            data_test = data_test.transpose(3, 0, 1, 2)
    elif data_space in ["nativesurface", "fsaverage"]:
        # Transpose to (vertices, trials)
        subject_brain_data = neural_data.transpose(1, 0)
        data_train, data_test, df_train, df_test, stimulus_ids_train, stimulus_ids_test = process_subject_data(
            subject_brain_data,
            df_stim_sub,
            keep_reps=keep_reps,
        )
        if keep_reps:
            # Transpose to (trials, reps, vertices)
            data_train = data_train.transpose(1, 2, 0)
            data_test = data_test.transpose(1, 2, 0)
        else:
            # Transpose to (trials, vertices)
            data_train = data_train.transpose(1, 0)
            data_test = data_test.transpose(1, 0)
    print("Processed train data shape:", data_train.shape)
    print("Processed test data shape:", data_test.shape)

    if not debug_mode and not keep_reps:
        # Sort stimulus IDs
        print("Sorting stimulus IDs and data...")
        sorting_indices_train = np.argsort(stimulus_ids_train)
        sorting_indices_test = np.argsort(stimulus_ids_test)

        # Apply sorting
        print("Applying sorting to stimulus IDs and data...")
        stimulus_ids_train = stimulus_ids_train[sorting_indices_train]
        stimulus_ids_test = stimulus_ids_test[sorting_indices_test]
        data_train = data_train[sorting_indices_train]
        data_test = data_test[sorting_indices_test]
        print("Sorted train data shape:", data_train.shape)
        print("Sorted test data shape:", data_test.shape)

    # 4) Write to HDF5
    print(f"Writing processed data to {out_path}...")
    with h5py.File(out_path, "a") as f:
        # Set global attrs if not present
        if "data_space" not in f.attrs:
            f.attrs["data_space"] = data_space
        if "beta_type" not in f.attrs:
            f.attrs["beta_type"] = beta_type
        if "subjects" in f.attrs:
            subjects = list(f.attrs["subjects"])
            if subject not in subjects:
                subjects.append(subject)
                f.attrs["subjects"] = subjects
        else:
            f.attrs["subjects"] = [subject]
        if "rois" not in f.attrs:
            f.attrs["rois"] = list(roi_mapping.keys())
        if "ROI_mapping" not in f.attrs:
            f.attrs["ROI_mapping"] = json.dumps(roi_mapping)
        if "splits" not in f.attrs:
            f.attrs["splits"] = ["train", "test"]
        if "dtype" in f.attrs:
            assert f.attrs["dtype"] == dtype, "Inconsistent dtype across subjects"
        else:
            f.attrs["dtype"] = dtype
        if args is not None:
            f.attrs["last_args"] = json.dumps(vars(args))
        if "max_nc" not in f.attrs:
            f.attrs["max_nc"] = 100.0

        # In NSD, each subject has their own subject-specific stimulus set
        f.attrs["subject_specific_stimulus_set"] = True

        
        # Write stimulus data
        print("Writing stimulus IDs...")
        key = f"train/stimulus_ids/{subject}"
        if key in f:
            del f[key]
        f.create_dataset(
            key,
            data=np.array(stimulus_ids_train, dtype=np.int32),
        )
        key = f"test/stimulus_ids/{subject}"
        if key in f:
            del f[key]
        f.create_dataset(
            key,
            data=np.array(stimulus_ids_test, dtype=np.int32),
        )
            
        
        # Create noise ceiling dataset
        print("Writing noise ceilings...")
        
        # Full noise ceiling without filtering or masking
        key = f"noise_ceiling_full/{subject}"
        if key in f:
            del f[key]
        f.create_dataset(key, data=noise_ceiling)
        for roi_name, roi_mask in tqdm(roi_masks_nc.items(), leave=False, total=len(roi_masks_nc), desc=f"Write ROI noise ceilings"):
            key = f"noise_ceilings/{subject}/{roi_name}"
            if key in f:
                del f[key]            
            f.create_dataset(key, data=noise_ceiling[roi_mask])

        # Create ROI labels
        print("Writing ROI labels:")
        for roi_name, roi_mask in tqdm(roi_masks.items(), desc="Creating ROI labels", leave=False):
            key = f"roi_labels/{subject}/{roi_name}"
            if key in f:
                del f[key]
            f.create_dataset(key, data=roi_mask)

        # Create noise ceiled ROI labels
        print("Create noise ceiled ROI labels...")
        for roi_name, roi_mask in tqdm(roi_masks_nc.items(), leave=False, total=len(roi_masks_nc), desc=f"Create noise ceiled ROI labels"):
            key = f"roi_labels_nc/{subject}/{roi_name}"
            if key in f:
                del f[key]
            f.create_dataset(key, data=roi_mask)
            
        print("Writing data...")
        for roi_name, roi_mask in tqdm(roi_masks_nc.items(), leave=False, total=len(roi_masks_nc), desc=f"Write ROI data"):
            key = f"train/neural_data/{subject}/{roi_name}"
            if key in f:
                del f[key]
            f.create_dataset(key, data=data_train[..., roi_mask])
            
            key = f"test/neural_data/{subject}/{roi_name}"
            if key in f:
                del f[key]
            f.create_dataset(key, data=data_test[..., roi_mask])

        print(f"[{subject}] Finished writing to {out_path}")


# ==========================================
# CLI
# ==========================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "NSD preprocessing: volume/surface, one subject at a time.\n"
            "Loads betas for a single subject, applies ROI + noise ceiling, "
            "averages repetitions, performs train/test split, and appends "
            "results to an HDF5 file."
        )
    )
    parser.add_argument(
        "--subject",
        type=str,
        choices=[f"subj{idx:02d}" for idx in range(1, 9)] + [str(idx) for idx in range(1, 9)],
        required=True,
        help="Subject id, e.g. 'subj01' or '1'.",
    )
    parser.add_argument(
        "--data-space",
        choices=["func1pt8mm", "nativesurface", "fsaverage"],
        required=True,
        help="NSD beta space to use.",
    )
    parser.add_argument(
        "--beta-type",
        choices=["betas_fithrf_GLMdenoise_RR"],
        default="betas_fithrf_GLMdenoise_RR",
        help="Beta type subdirectory name.",
    )
    parser.add_argument(
        "--ds-dir",
        type=str,
        help="Root NSD dataset directory.",
    )
    parser.add_argument(
        "--stim-csv-path",
        type=str,
        help="Path to nsd_stim_mapping.csv (stimulus table for all subjects).",
    )
    parser.add_argument(
        "--roi-mapping",
        type=str,
        choices=["combined", "individual"],
        default="combined",
        help="ROI mapping strategy.",
    )
    parser.add_argument(
        "--nc-threshold",
        type=float,
        default=NOISE_CEILING_THRESHOLD,
        help="Noise ceiling threshold (%%). Voxels below are discarded.",
    )
    parser.add_argument(
        "--keep-reps",
        action="store_true",
        help="Keep repetitions instead of averaging (requires all stimuli to have 3 reps).",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32", "float64"],
        default="float32",
        help="Data type for output arrays.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for HDF5 file.",
    )
    parser.add_argument(
        "--apply-nsdgeneral-mask",
        action="store_true",
        help="Apply the nsdgeneral ROI mask to all ROIs (except RSC) to restrict to visually responsive cortex.",
    )
    parser.add_argument(
        "--debug-mode",
        action="store_true",
        help="Enable debug mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Arguments:", args)

    # Normalise subject id
    if args.subject.startswith("subj"):
        subject = args.subject
    else:
        subject = SUBJECT_TEMPLATE.format(idx=int(args.subject))

    # Load full stimulus mapping once
    df_stim_all = pd.read_csv(args.stim_csv_path)
    
    
    if args.roi_mapping == "individual":
        ROI_version = "individualROIs"
    else:
        ROI_version = "combinedROIs"

    ds_dir = Path(args.ds_dir)
    # output_path = Path(args.output_dir) / f"nsd_{args.data_space}_{ROI_version}_nc{int(args.nc_threshold)}.h5"
    output_path = Path(args.output_dir) / f"nsd_{args.data_space}_{ROI_version}.h5"
    
    if args.debug_mode:
        output_path = output_path.with_name(output_path.stem + "_debug" + output_path.suffix)
    
    if args.keep_reps:
        output_path = output_path.with_name(output_path.stem + "_reps" + output_path.suffix)

    process_single_subject(
        subject=subject,
        ds_dir=ds_dir,
        data_space=args.data_space,
        beta_type=args.beta_type,
        df_stim_all=df_stim_all,
        nc_threshold=args.nc_threshold,
        keep_reps=args.keep_reps,
        roi_mapping_type=args.roi_mapping,
        out_path=output_path,
        dtype=args.dtype,
        debug_mode=args.debug_mode,
        apply_nsdgeneral_mask=args.apply_nsdgeneral_mask,
        args=args,
    )


if __name__ == "__main__":
    main()
