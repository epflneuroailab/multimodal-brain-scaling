import numpy as np
from scipy.stats import zscore, pearsonr
from tqdm.auto import tqdm


def compute_ceiling_splithalf(
    responses: np.ndarray,
    folds: int = 10,
    seed: int = 0,
    spearman_brown: bool = True,
    equalize_halves: bool = True,
    clip_folds: bool = False
) -> np.ndarray:
    """
    Split-half reliability per unit (voxel/channel).

    Parameters
    ----------
    responses : np.ndarray
        Shape (n_units, n_stimuli, n_reps) or (n_channels, n_timepoints, n_stimuli, n_reps). 
        Last axis is repetitions/trials.
    folds : int, default=10
        Number of random split-halves to sample.
    seed : int, default=0
        Base RNG seed; each fold uses seed+fold_idx.
    spearman_brown : bool, default=True
        Apply Spearman-Brown correction: r_sb = 2r / (1+r).
    equalize_halves : bool, default=True
        If True, use equal-sized halves (drops one trial if n_reps is odd).
        If False, second half may be larger by one when n_reps is odd.
    clip_folds : bool, default=False
        If True, clip correlations to be between 0 and 100 after Spearman-Brown correction.
    Returns
    -------
    np.ndarray
        Array of shape (n_units, folds) with reliabilities in percent.
    """

    n_reps = responses.shape[-1]
    if n_reps < 2:
        raise ValueError("Need at least 2 repetitions for ceiling computation.")

    fold_corrs = []
    for f in tqdm(range(folds), total=folds, desc="Split-half folds", leave=False):
        rng = np.random.default_rng(seed + f)
        perm = rng.permutation(n_reps)

        if equalize_halves:
            k = n_reps // 2
            idx_A = perm[:k]
            idx_B = perm[k: 2 * k]  # drop leftover if odd
        else:
            idx_A = perm[: n_reps // 2]
            idx_B = perm[n_reps // 2 :]

        # Average within halves -> (n_units, n_stimuli)
        avg_A = np.nanmean(responses[..., idx_A], axis=-1)
        avg_B = np.nanmean(responses[..., idx_B], axis=-1)

        # Pearson r across stimuli, per unit
        r, _ = pearsonr(avg_A, avg_B, axis=-1)

        if spearman_brown:
            r_clip = np.clip(r, -0.999999, 0.999999)
            r = 2.0 * r_clip / (1.0 + r_clip)

        fold_corrs.append(r)
        
    fold_corrs = np.stack(fold_corrs, axis=-1)*100  # (n_units, folds)

    if clip_folds:
        fold_corrs = np.clip(fold_corrs, 0.0, 100.0)

    return fold_corrs  # (n_units, folds)

def compute_ceiling_variancebased(responses: np.ndarray, nan_policy: str = 'omit') -> np.ndarray:
    """
    Noise ceiling per unit using the method described in the NSD paper (Allen et al., 2021).

    Steps:
      1) z-score across stimuli (axis=1) for each (unit, rep) -> total var ≈ 1
      2) estimate noise variance across repetitions (axis=2), then average across stimuli
      3) signal variance = 1 - noise_var
      4) reliability (percent) for finite repeats: nc = 100 * (snr / (snr + 1 / n_reps))

    Parameters
    ----------
    responses : np.ndarray
        Shape (n_units, n_stimuli, n_reps) or (n_channels, n_timepoints, n_stimuli, n_reps).
    nan_policy : {'propagate', 'raise', 'omit'}, default='omit'
        Passed to scipy.stats.zscore for handling NaNs.
    Returns
    -------
    np.ndarray
        Per-unit noise ceilings in percent with shape (n_units,).
    """
    
    n_reps = responses.shape[-1]
    if n_reps < 2:
        raise ValueError("Need at least 2 repetitions for ceiling computation.")

    # 1) Normalize across stimuli so total variance per (unit, rep) across stimuli is ~1
    norm = zscore(responses, axis=-2, ddof=0, nan_policy=nan_policy)  # (n_units, n_stimuli, n_reps)

    # 2) Noise variance: variance across repetitions (axis=2), averaged across stimuli (axis=1)
    noise_var = np.nanmean(np.nanvar(norm, axis=-1, ddof=1), axis=-1)  # (n_units,)
    
    # 3) Signal variance: total(=1) - noise variance  (clipped for numerical safety)
    signal_var = np.clip(1.0 - noise_var, 0.0, None)

    # 4) Reliability in percent
    snr = signal_var / np.maximum(noise_var, 1e-12)
    nc = 100.0 * (snr / (snr + 1 / n_reps))

    return nc