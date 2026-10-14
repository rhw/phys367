import numpy as np
import utility_functions


def make_admixture(
    mags: np.ndarray,
    library: np.ndarray,
    admixture: float=0.01,
    zero_points: float = 31.4,
    seed: int | None = None,
) -> np.ndarray:
    """Add an admixture of fluxes to objects

    Parameters
    ----------
    mags:
        Input mags [N_objects, N_bands]

    library:
        Library for adding admixtures  [N_bkg_objects, N_bands]

    admixture:
        Admixture fraction

    zero_points:
        Flux to mag zero-point

    seed:
        Random number seed

    Returns
    -------
    Magnitudes with flux admixtures added
    """
    n_lib = len(library)
    n_obj = len(mags)

    fluxes = utility_functions.mags_to_fluxes(mags, zero_points)
    lib_fluxes = utility_functions.mags_to_fluxes(library, zero_points)
    lib_fluxes = np.nan_to_num(lib_fluxes)

    total_fluxes = fluxes.sum(axis=1)

    if seed is not None:
        np.random.seed(seed)

    picks = np.random.randint(n_lib, size=n_obj)
    pick_fluxes = lib_fluxes[picks]
    pick_totals = pick_fluxes.sum(axis=1)

    pick_weights = total_fluxes / pick_totals
    contamination = (pick_fluxes.T * (pick_weights * admixture)).T

    new_fluxes = fluxes + contamination
    new_mags = utility_functions.fluxes_to_mags(new_fluxes, zero_points)

    return new_mags


def gaussian_noise(
    mags: np.ndarray,
    noise_levels: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """Add noise of fluxes to objects

    Parameters
    ----------
    mags:
        Input mags [N_objects, N_bands]

    noise_levels:
        Noise levels (in mags)

    seed:
        Random number seed

    Returns
    -------
    Magnitudes with noise added
    """
    n_obj = len(mags)
    n_mags = mags.shape[-1]
    
    if seed is not None:
        np.random.seed(seed)

    mag_list = []
    for i, noise_level in enumerate(noise_levels):
        noise = random_array = np.random.normal(loc=0, scale=noise_level, size=n_obj)
        mag_list.append(mags[:,i] + noise)

    new_mags = np.vstack(mag_list)
    return new_mags
