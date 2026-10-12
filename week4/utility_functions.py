import os
from typing import Any, TypeAlias
import numpy as np
from numpy.polynomial import Polynomial
import urllib.request
import subprocess

Tablelike: TypeAlias = Any


def setup_data() -> int:  # pragma: no cover
    if not os.path.exists("phys367_week4_data.tgz"):
        urllib.request.urlretrieve(
            "http://s3df.slac.stanford.edu/people/echarles/to_download/phys367_week4_data.tgz",
            "phys367_week4_data.tgz",
        )
        print(f"File downloaded to phys367_week4_data.tgz")        
        if not os.path.exists("phys367_week4_data.tgz"):
            return 1

    if not os.path.exists(f"data/object.hdf5"):
        print(f"Uncompressing phys367_week4_data.tgz")
        status = subprocess.run(["tar", "zxvf", "phys367_week4_data.tgz"], check=False)
        if status.returncode != 0:
            return status.returncode
        if not os.path.exists("data/object.hdf5"):        
            return 2
    else:
        print("Data already downloaded and uncompressed")

    print("Data setup complete!")
    return 0


def teardown_datq() -> None:  # pragma: no cover
    home = os.environ['HOME']
    if os.environ.get("TEARDOWN"):
        os.system(f"\\rm -rf {home}/macss")
        try:
            os.unlink(f"{home}/macss.tgz")
        except FileNotFoundError:
            pass



def make_band_names(template: str, bands: list[str]) -> list[str]:
    """Make a set of band names from template and a list of bands

    Parameters
    ----------
    template:
        Template to make the names

    bands:
        List of the bands to apply to the template

    Returns
    -------
    Names of the bands
    """
    return [template.format(band=band_) for band_ in bands]


def extract_data_to_2d_array(data: Tablelike, column_names: list[str]) -> np.ndarray:
    """Extract a set of columns from a table to a 2D array

    Parameters
    ----------
    data:
        Input data

    column_names:
        Names of the columns to extract

    Returns
    -------
    Output 2D-Array
    """
    column_data = [data[column_] for column_ in column_names]
    return np.vstack(column_data).T


def make_mag_dict(
    data: Tablelike,
    band_name_template: str,
    bands: list[str],
) -> dict[str, np.ndarray]:
    """Extract magntidues from a table to a dict of numpy arrays

    Parameters
    ----------
    data:
        Input data

    band_name_template:
        Template to make the names

    bands:
        List of the bands to apply to the template

    Returns
    -------
    Output dict
    """
    mag_dict = {}
    for band_ in bands:
        col = band_name_template.format(band=band_)
        mags = fluxes_to_mags(data[col])
        mag_dict[band_] = mags
    return mag_dict


def make_colors(
    data: Tablelike,
    band_name_template: str,
    bands: list[str],
) -> dict[str, np.ndarray]:
    """Extract colors from a table of mags to a dict of numpy arrays

    Parameters
    ----------
    data:
        Input data, with magntidues

    band_name_template:
        Template to make the names

    bands:
        List of the bands to apply to the template

    Returns
    -------
    Output dict of adjacent band colors
    """
    colors = {}
    for i, band_ in enumerate(bands[0:-1]):
        col_a = band_name_template.format(band=band_)
        col_b = band_name_template.format(band=bands[i + 1])
        colors[f"{band_}-{bands[i+1]}"] = data[col_a] - data[col_b]
    return colors


def make_color_dict(
    data: Tablelike,
    band_name_template: str,
    bands: list[str],
) -> dict[str, np.ndarray]:
    """Extract colors from a table of fluxes to a dict of numpy arrays

    Parameters
    ----------
    data:
        Input data, with fluxes

    band_name_template:
        Template to make the names

    bands:
        List of the bands to apply to the template

    Returns
    -------
    Output dict
    """
    colors = {}
    for i, band_ in enumerate(bands[0:-1]):
        col_a = band_name_template.format(band=band_)
        col_b = band_name_template.format(band=bands[i + 1])
        try:
            mag_a = fluxes_to_mags(data[col_a].to_numpy())
            mag_b = fluxes_to_mags(data[col_b].to_numpy())
        except AttributeError:
            mag_a = fluxes_to_mags(data[col_a])
            mag_b = fluxes_to_mags(data[col_b])
        colors[f"{band_}-{bands[i+1]}"] = mag_a - mag_b
    return colors


def get_band_values(
    input_data: Tablelike,
    band_name_template: str,
    bands: list[str],
) -> np.ndarray:
    """Extract a set of columns from a table to a 2D array

    Parameters
    ----------
    input_data:
        Input data

    band_name_template:
        Template to make the names

    bands:
        List of the bands to apply to the template

    Returns
    -------
    Output 2D-Array
    """
    band_names = make_band_names(band_name_template, bands)
    values = extract_data_to_2d_array(input_data, band_names)
    return values


def fluxes_to_mags(
    fluxes: np.ndarray, zero_points: float | np.ndarray = 31.4
) -> np.ndarray:
    """Convert fluxes to magnitudes

    Parameters
    ----------
    fluxes:
        Input data

    zero_points:
        Zero-point magnitudes

    Returns
    -------
    Output magntidues
    """
    return -2.5 * np.log10(fluxes) + zero_points


def mags_to_fluxes(
    mags: np.ndarray, zero_points: float | np.ndarray = 31.4
) -> np.ndarray:
    """Convert magnitudes to fluxes

    Parameters
    ----------
    mags:
        Input data

    zero_points:
        Zero-point magnitudes

    Returns
    -------
    Output fluxes
    """
    return np.power(10, (zero_points - mags) / 2.5)


def adjacent_band_colors(mags: np.ndarray) -> np.ndarray:
    """Return a set of colors using magnitudes in adjacent bands

    I.e., u-g, g-r, r-i, i-z, z-y

    Note that there will be one less color than bands

    Parameters
    ----------
    mags:
        Input data

    Returns
    -------
    Output colors
    """
    n_bands = mags.shape[-1]
    colors = [mags[:, i] - mags[:, i + 1] for i in range(n_bands - 1)]
    return np.vstack(colors).T


def ref_band_colors(mags: np.ndarray, ref_band_index: int) -> np.ndarray:
    """Return a set of colors using magnitudes against a reference band

    I.e., u-i, g-i, r-i, z-i, y-i

    Note that there will be one less color than bands

    Parameters
    ----------
    mags:
        Input data

    Returns
    -------
    Output colors
    """
    ref_mags = mags[:, ref_band_index]
    n_bands = mags.shape[-1]
    colors_list = []
    for i in range(n_bands):
        if i == ref_band_index:
            continue
        colors_list.append(mags[:, i] - ref_mags)
    return np.vstack(colors_list).T


def polynomial_fits(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    y_errs: np.ndarray,
    x_pivot: float = 0,
    degree: int = 3,
) -> np.ndarray:
    """Fit data to n-degree polynomials and return the coefficents

    Parameters
    ----------
    x_vals:
        Input X data with N values

    y_vals:
        Input Y data with M,N values

    y_errs:
        Input Y Errors with M,N values

    x_pivot:
        Pivot value for X, (i.e., x_vals - x_pivot is used in fit)

    degree:
        Polynomial degee

    Returns
    -------
    M, Degree+1 coefficients
    """
    l_out = []
    for y_, yerr_ in zip(y_vals, y_errs):
        try:
            p = Polynomial.fit(x_vals - x_pivot, y_, w=1.0 / yerr_, deg=degree)
            l_out.append(p.coef)
        except Exception:
            l_out.append(np.array([np.nan] * (degree + 1)))
    return np.array(l_out)


def linear_fit_residuals(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    x_pivot: float = 0,
    degree: int = 2,
) -> np.ndarray:
    """Fit data to n-degree polynomials and return the coefficents

    Parameters
    ----------
    x_vals:
        Input X data with N values

    y_vals:
        Input Y data with M,N values

    x_pivot:
        Pivot value for X, (i.e., x_vals - x_pivot is used in fit)

    Returns
    -------
    M, N, residuals
    """
    l_out = []
    for y_ in y_vals:
        try:
            p = Polynomial.fit(x_vals - x_pivot, y_, deg=degree)
            resid = p(x_vals - x_pivot) - y_
            l_out.append(resid)
        except Exception:
            l_out.append(np.array([np.nan] * (degree + 1)))
    return np.array(l_out)


def color_excess(
    mags: np.ndarray,
) -> np.ndarray:
    """Extract excess each band w.r.t. adjancent bands

    Parameters
    ----------
    mags:
        Input data, with mags [N_band, N_objects]

    Returns
    -------
    Tables with excesses [N_band-2, N_objects]
    """
    n_mags = mags.shape[-1]
    color_excess_list = []
    for i in range(1, n_mags - 1):
        color_excess_list.append(0.5 * (mags[:, i - 1] + mags[:, i + 1]) - mags[:, i])

    return np.array(color_excess_list).T


def build_template_dict(
    seds: list[str], filters: list[str]
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """Extract AB templates

    Parameters
    ----------
    seds:
        Names of the seds to use

    filters:
        Name of the filters to use

    Returns
    -------
    Dict mapping sed name to a tuple with
    redshifts (N), mags (N, N_filter), colors (N, N_filter-1), sed_index
    """
    template_dict = {}

    pz_dir = f"."
    
    for ised, sed in enumerate(seds):
        mag_data_list = []
        for filter_ in filters:
            path = f"{pz_dir}/data/AB/{sed}.{filter_}.AB"
            data = np.loadtxt(path)
            _redshifts = data[:, 0]
            mags = fluxes_to_mags(data[:, 1], 31.4)
            mag_data_list.append(mags)
        mag_data = np.vstack(mag_data_list).T
        color_data = adjacent_band_colors(mag_data).T
        template_dict[sed] = (_redshifts, mag_data, color_data, ised)
    return template_dict


def prepare_data_total_mag_and_colors(
    input_data: np.ndarray,
    band_name_template: str,
    bands: list[str],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Extract data for Regression algorithms

    Parameters
    ----------
    input_data:
        Table with input data

    band_name_template:
        Template for the band names

    bands:
        List of the bands

    Returns
    -------
    Tuple with ndarray(N) of target redshift and
    ndarray(N,N_color+1) of summed magntiude and colors
    """
    band_names = make_band_names(band_name_template, bands)
    mags = extract_data_to_2d_array(input_data, band_names)
    fluxes = mags_to_fluxes(mags, 31.4)
    fluxes = np.where(np.isfinite(fluxes), fluxes, 0.0)
    total_fluxes = np.sum(fluxes, axis=1)
    mag_total = fluxes_to_mags(total_fluxes, 31.4)
    mag_total = np.where(np.isfinite(mag_total), mag_total, 25.0)
    mags = np.where(np.isfinite(mags), mags, 27.0)
    colors = adjacent_band_colors(mags).clip(-2, 2)
    try:
        targets = input_data["redshift"]
    except KeyError:
        targets = None
    features = np.vstack([mag_total, colors.T]).T
    return (targets, features)


def run_regression(
    regerssor,
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
) -> np.ndarray:
    """Run as scikit-learn regression algorithm

    Parameters
    ----------
    regerssor
        Scikit-learn regressor

    train_features:
        Features in training data

    train_targets:
        Targets in training data

    test_features:
        Features in test data

    Returns
    -------
    Predictions on test data
    """
    regerssor.fit(train_features, train_targets)
    return regerssor.predict(test_features)
