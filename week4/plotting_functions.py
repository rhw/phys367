import numpy as np
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
from matplotlib import colors, cm
from scipy.stats import sigmaclip
from astropy.stats import biweight_location, biweight_scale


def get_subplot_nrow_ncol(nfig: int) -> tuple[int, int]:
    """Get the number of rows and columns of sub-plots
    for a particular number of plots

    Parameters
    ----------
    nfig:
        Number of figures

    Returns
    -------
    Number of rows and columns as (nrow, ncol)
    """
    shape_dict = {
        1: (1, 1),
        2: (1, 2),
        3: (1, 3),
        4: (2, 2),
        5: (2, 3),
        6: (2, 3),
        7: (2, 4),
        8: (2, 4),
        9: (3, 3),
        10: (3, 4),
        11: (3, 4),
        12: (3, 4),
        13: (4, 4),
        14: (4, 4),
        15: (4, 4),
        16: (4, 4),
    }
    try:
        return shape_dict[nfig]
    except KeyError:
        raise ValueError(
            f"Sorry, Phillipe.  I'm not going to put {nfig} subplots in one figure"
        ) from None


def plot_feature_histograms(
    data: np.ndarray, labels: list[str] | None = None
) -> Figure:
    """Plot Histograms of the features being used to train
    a ML algorithm on a single, busy figure

    Parameters
    ----------
    data:
        Input data

    labels:
        Labels for the various features


    Returns
    -------
    Figure with requested plots
    """
    fig = plt.figure(figsize=(8, 8))
    n_features = data.shape[-1]
    nrow, ncol = get_subplot_nrow_ncol(n_features)
    axs = fig.subplots(nrow, ncol)

    for ifeature in range(n_features):
        icol = int(ifeature / ncol)
        irow = ifeature % ncol

        axs[icol][irow].hist(data[:, ifeature], bins=100)
        if labels is not None:
            axs[icol][irow].set_xlabel(labels[ifeature])

    fig.tight_layout()
    return fig


def plot_true_nz(targets: np.ndarray) -> Figure:
    """Plot the true NZ

    Parameters
    ----------
    targets:
        Input data

    Returns
    -------
    Figure with requested plot
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.subplots(1, 1)
    ax.hist(targets, bins=100)
    fig.tight_layout()
    return fig


def plot_pca_hist2d(
    data: np.ndarray, pca_out: np.ndarray, labels: list[str] | None = None
) -> Figure:
    """Plot input data v. principle componment analysis features

    Parameters
    ----------
    data:
        Input data

    pca_out:
        Output of principle compoments analysis

    lables:
        Labels for the data columns

    Returns
    -------
    Figure with requested plots


    Notes
    -----
    This will create N_features X N_components sub-plots
    """

    fig = plt.figure(figsize=(8, 8))
    n_features = data.shape[-1]
    n_comps = pca_out.shape[-1]
    nrow, ncol = n_features, n_comps
    axs = fig.subplots(nrow, ncol)

    for irow in range(nrow):
        row_data = data[:, irow]
        for icol in range(ncol):
            col_data = pca_out[:, icol]
            axs[irow][icol].hist2d(
                row_data, col_data, bins=(100, 100), norm="log", cmap="gray"
            )
        if labels is not None:
            axs[irow][0].set_xlabel(labels[irow])
    fig.tight_layout()
    return fig


def plot_feature_target_hist2d(
    data: np.ndarray, targets: np.ndarray, labels: list[str] | None = None
) -> Figure:
    """Plot input data v. target redshift value as 2D histogram

    Parameters
    ----------
    data:
        Input data [N_objects, N_features]

    targets:
        Target redshirt [N_objects]

    lables:
        Labels for the data columns [N_features]

    Returns
    -------
    Figure with requested plots

    Notes
    -----
    This will create N_features sub-plots
    """

    fig = plt.figure(figsize=(8, 8))
    n_features = data.shape[-1]
    nrow, ncol = get_subplot_nrow_ncol(n_features)
    axs = fig.subplots(nrow, ncol)

    for ifeature in range(n_features):
        icol = int(ifeature / ncol)
        irow = ifeature % ncol

        axs[icol][irow].hist2d(
            targets, data[:, ifeature], bins=100, norm="log", cmap="gray"
        )
        if labels is not None:
            axs[icol][irow].set_xlabel(labels[irow])
    fig.tight_layout()
    return fig


def plot_features_target_scatter(
    data: np.ndarray, targets: np.ndarray, labels: list[str] | None = None
) -> Figure:
    """Plot input data v. target redshift value as scatter plot

    Parameters
    ----------
    data:
        Input data [N_objects, N_features]

    targets:
        Target redshirt [N_objects]

    lables:
        Labels for the data columns [N_features]

    Returns
    -------
    Figure with requested plots

    Notes
    -----
    This will create N_features sub-plots
    """
    fig = plt.figure(figsize=(8, 8))
    n_features = data.shape[-1]
    nrow, ncol = get_subplot_nrow_ncol(n_features)
    axs = fig.subplots(nrow, ncol)

    for ifeature in range(n_features):
        irow = int(ifeature / ncol)
        icol = ifeature % ncol

        axs[irow][icol].scatter(targets, data[:, ifeature], marker=".", s=1)
        if labels is not None:
            axs[irow][0].set_ylabel(labels[ifeature])
    fig.tight_layout()
    return fig


def plot_features_pca_scatter(
    data: np.ndarray,
    pca_out: np.ndarray,
    targets: np.ndarray,
    labels: list[str] | None = None,
) -> Figure:
    """Plot input data v. pca with target redshift as color

    Parameters
    ----------
    data:
        Input data [N_objects, N_features]

    pca_out:
        PCA transformed data [N_objects, N_components]

    targets:
        Target redshirt [N_objects]

    lables:
        Labels for the data columns [N_features]

    Returns
    -------
    Figure with requested plots

    Notes
    -----
    This will create N_features sub-plots
    """
    fig = plt.figure(figsize=(8, 8))
    n_features = data.shape[-1]
    n_components = pca_out.shape[-1]
    nrow, ncol = n_features, n_components
    axs = fig.subplots(nrow, ncol)

    for irow in range(nrow):
        row_data = data[:, irow]
        for icol in range(ncol):
            axs[irow][icol].scatter(
                row_data, pca_out[:, icol], c=targets, marker=".", s=1
            )
            if labels is not None:
                axs[irow][0].set_xlabel(labels[irow])
    fig.tight_layout()
    return fig


def plot_true_predict_simple(targets: np.ndarray, predictions: np.ndarray) -> Figure:
    """Plot predicted redshift v. true redshift as a 2d histogram

    Parameters
    ----------
    targets:
        Target redshifts [N_objects]

    predictions:
        Predicted redshifts [N_objects]

    Returns
    -------
    Figure with requested plots
    """
    fig = plt.figure(figsize=(8, 8))
    ax = fig.subplots(1, 1)
    ax.hist2d(targets, predictions, bins=(100, 100), norm="log", cmap="gray")
    return fig


def get_biweight_mean_sigma_outlier(
    subset: np.ndarray, nclip: int = 3
) -> tuple[float, float, float, float]:
    """Return biweight stats with sigma clipping

    Parameters
    ----------
    subset:
        Input data, estimate - reference redshifts

    nclip:
        Value for sigma clipping

    Returns
    -------
    Mean, error on mean, std, outlier_rate
    """

    subset_clip, _, _ = sigmaclip(subset, low=nclip, high=nclip)
    for _j in range(nclip):
        subset_clip, _, _ = sigmaclip(subset_clip, low=nclip, high=nclip)

    mean = biweight_location(subset_clip)
    std = biweight_scale(subset_clip)
    outlier_rate = np.sum(np.abs(subset) > nclip * biweight_scale(subset_clip)) / len(
        subset
    )

    return mean, std / np.sqrt(len(subset_clip)), std, outlier_rate


def plot_true_predict_fancy(targets: np.ndarray, predictions: np.ndarray) -> Figure:
    """Plot predicted redshift v. true redshift as with nice overlayes

    Parameters
    ----------
    targets:
        Target redshifts [N_objects]

    predictions:
        Predicted redshifts [N_objects]

    Returns
    -------
    Figure with requested plots and nice overlays
    """
    z_min = 0.0
    z_max = 1.5
    figure, axes = plt.subplots(figsize=(7, 6))
    bin_edges = np.linspace(z_min, z_max, 151)
    dz = (predictions - targets) / (1 + targets)
    mean, _mean_err, std, outlier_rate = get_biweight_mean_sigma_outlier(dz, nclip=3)
    mean, std, outlier_rate = round(mean, 4), round(std, 4), round(outlier_rate, 4)
    h = axes.hist2d(
        targets,
        predictions,
        bins=(bin_edges, bin_edges),
        norm=colors.LogNorm(),
        cmap="gray",
    )
    axes.plot(
        [z_min, z_max],
        [z_min, z_max],
        "--",
        color="red",
    )
    axes.plot(
        [z_min, z_max],
        [z_min - 3 * std, z_max - 3 * std],
        "--",
        color="red",
    )
    axes.plot(
        [z_min, z_max],
        [z_min + 3 * std, z_max + 3 * std],
        "--",
        color="red",
    )
    axes.plot(
        [],
        [],
        ".",
        alpha=0.0,
        label=rf"$\Delta z = {mean} $"
        + "\n"
        + rf"$\sigma z = {std} $"
        + "\n"
        + f"outlier rate = {outlier_rate}",
    )

    plt.xlabel("True Redshift")
    plt.ylabel("Estimated Redshift")
    cb = figure.colorbar(h[3], ax=axes)
    cb.set_label("Density")

    plt.legend()
    figure.tight_layout()
    return figure


def plot_colors_v_redshifts_with_templates(
    redshifts: np.ndarray,
    color_data: np.ndarray,
    zmax: float = 4.0,
    templates: dict | None = None,
    labels: list[str] | None = None,
) -> Figure:
    """Plot colors v. redshift with overlaid template tracks

    Parameters
    ----------
    redshifts:
        Redshifts [N_objects]

    color_data:
        Color data [N_colors, N_objects]

    zmax:
        Maximum redshift for templates

    templates:
        Dictionary of templates to plot

    labels:
        Labels for the data columns [N_features]

    Returns
    -------
    Figure with requested plots and nice overlays
    """

    fig = plt.figure(figsize=(8, 8))
    n_colors = color_data.shape[-1]
    nrow, ncol = get_subplot_nrow_ncol(n_colors)
    axs = fig.subplots(nrow, ncol)

    for icolor in range(n_colors):
        icol = int(icolor / ncol)
        irow = icolor % ncol
        axs[icol][irow].hist2d(
            redshifts,
            color_data[:, icolor],
            bins=[np.linspace(0, zmax, 121), np.linspace(-3, 3, 121)],
            cmap="gray",
            norm="log",
        )
        axs[icol][irow].set_xlim(0, zmax)
        axs[icol][irow].set_ylim(-3.0, 3.0)
        if templates is not None:
            for key, val in templates.items():
                mask = val[0] < zmax
                _ = axs[icol][irow].plot(
                    val[0][mask],
                    val[2][icolor][mask],
                    label=key,
                    c=cm.rainbow(1.0 - val[3] / len(templates)),
                )
        # axs[icol][irow].legend()
        axs[icol][irow].set_xlabel("redshift")
        if labels is not None:
            axs[icol][irow].set_ylabel(labels[icolor])

    fig.tight_layout()
    return fig


def plot_colors_v_colors_with_templates(
    color_data: np.ndarray,
    zmax: float = 4.0,
    templates: dict | None = None,
    labels: list[str] | None = None,
) -> Figure:
    """Plot colors v. colors with overlaid template tracks

    Parameters
    ----------
    color_data:
        Color data [N_colors, N_objects]

    zmax:
        Maximum redshift for templates

    templates:
        Dictionary of templates to plot

    labels:
        Labels for the data columns [N_features]

    Returns
    -------
    Figure with requested plots and nice overlays
    """

    fig = plt.figure(figsize=(8, 8))
    n_colors = color_data.shape[-1]
    nrow, ncol = n_colors - 1, n_colors - 1
    axs = fig.subplots(nrow, ncol)

    for icol in range(n_colors - 1):
        for irow in range(n_colors - 1):
            axs[icol][irow].set_xlim(-3.0, 3.0)
            axs[icol][irow].set_ylim(-3.0, 3.0)
            if labels is not None:
                axs[icol][irow].set_ylabel(labels[icol])
                axs[icol][irow].set_xlabel(labels[irow + 1])
            if irow < icol:
                continue
            axs[icol][irow].scatter(
                color_data[:, icol], color_data[:, irow + 1], color="black", s=1
            )
            if templates is not None:
                for key, val in templates.items():
                    mask = val[0] < zmax
                    _ = axs[icol][irow].plot(
                        val[2][icol][mask],
                        val[2][irow + 1][mask],
                        label=key,
                        c=cm.rainbow(1.0 - val[3] / len(templates)),
                    )
            # axs[icol][irow].legend()
    fig.tight_layout()
    return fig


def process_data(
    zphot: np.ndarray,
    specz: np.ndarray,
    low: float = 0.01,
    high: float = 2.0,
    nclip: int = 3,
    nbin: int = 101,
) -> dict[str, list[float]]:
    dz = (zphot - specz) / (1 + specz)

    z_bins = np.linspace(low, high, nbin)
    # Bin the data
    bin_indices = np.digitize(zphot, bins=z_bins) - 1  # Assign each point to a bin

    biweight_mean: list[float] = []
    biweight_std: list[float] = []
    biweight_sigma: list[float] = []
    biweight_outlier: list[float] = []
    z_mean: list[float] = []
    qt_95_low: list[float] = []
    qt_68_low: list[float] = []
    median: list[float] = []
    qt_68_high: list[float] = []
    qt_95_high: list[float] = []
    for i in range(len(z_bins) - 1):
        subset = dz[bin_indices == i]
        if len(subset) < 1:
            continue
        subset_clip, _, _ = sigmaclip(subset, low=3, high=3)
        for _j in range(nclip):
            subset_clip, _, _ = sigmaclip(subset_clip, low=3, high=3)

        biweight_mean.append(biweight_location(subset_clip))
        biweight_std.append(biweight_scale(subset_clip) / np.sqrt(len(subset_clip)))
        biweight_sigma.append(biweight_scale(subset_clip))

        outlier_rate = np.sum(np.abs(subset) > 3 * biweight_scale(subset_clip)) / len(
            subset
        )
        biweight_outlier.append(outlier_rate)

        qt_95_low.append(np.percentile(subset, 2.5))
        qt_68_low.append(np.percentile(subset, 16))
        median.append(np.percentile(subset, 50))
        qt_68_high.append(np.percentile(subset, 84))
        qt_95_high.append(np.percentile(subset, 97.5))

        z_mean.append(np.mean(zphot[bin_indices == i]))
    return {
        "z_mean": z_mean,
        "biweight_mean": biweight_mean,
        "biweight_std": biweight_std,
        "biweight_sigma": biweight_sigma,
        "biweight_outlier": biweight_outlier,
        "qt_95_low": qt_95_low,
        "qt_68_low": qt_68_low,
        "median": median,
        "qt_68_high": qt_68_high,
        "qt_95_high": qt_95_high,
    }


def plot_biweight_stats_v_redshift(targets, predictions) -> Figure:

    n_zbins = 50
    z_min = 0.0
    z_max = 1.5
    n_clip = 3

    dz = (predictions - targets) / (1 + targets)

    results = process_data(
        predictions,
        targets,
        nbin=n_zbins,
        low=z_min,
        high=z_max,
        nclip=n_clip,
    )
    figure, axes = plt.subplots(2, 1, figsize=(8, 6))

    plt.subplots_adjust(wspace=0.1, hspace=0.0)

    axes[0].errorbar(
        results["z_mean"],
        results["biweight_mean"],
        results["biweight_std"],
        label="Bias",
    )

    axes[0].plot(results["z_mean"], results["biweight_sigma"], label=r"$\sigma_z$")

    axes[0].plot(results["z_mean"], results["biweight_outlier"], label=r"Outlier rate")
    axes[0].set_title(f"Bias, Sigma, and Outlier rates w/ {n_clip} sigma clipping")
    axes[0].set_ylabel("Statistics")
    axes[0].legend()
    axes[0].tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    axes[0].set_xlim(z_min, z_max)

    bin_edges_z = np.linspace(z_min, z_max, 100 + 1)
    bin_edges_dz = np.linspace(-0.3, 0.3, 100 + 1)
    axes[1].hist2d(
        predictions,
        dz,
        bins=(bin_edges_z, bin_edges_dz),
        norm=colors.LogNorm(),
        cmap="gray",
    )

    axes[1].set_xlim(z_min, z_max)
    for qt in ["qt_95_low", "qt_68_low", "median", "qt_68_high", "qt_95_high"]:
        axes[1].plot(results["z_mean"], results[qt], "--", color="blue", linewidth=2.0)

    axes[1].set_xlabel("Redshift")
    axes[1].set_ylabel(r"$(z_{phot} - z_{spec})/(1+z_{spec})$")
    figure.tight_layout()
    return figure


def plot_mag_spectra(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    y_errs: np.ndarray,
    targets: np.ndarray,
) -> Figure:

    figure, axes = plt.subplots(1, 1, figsize=(8, 6))

    for y_, yerr_, target_ in zip(y_vals, y_errs, targets):
        axes.errorbar(x_vals, y_, yerr_, color=cm.rainbow(target_))
    figure.tight_layout()
    return figure


def plot_mag_mag_scatter(t1, t2, bands, mask=None, norm="log"):
    fig = plt.figure()
    nrow, ncol = get_subplot_nrow_ncol(len(bands))
    axes = fig.subplots(nrow, ncol)
    for i, band in enumerate(bands):
        icol = i % ncol
        irow = int(i / ncol)
        axs = axes[irow][icol]
        if mask is None:
            axs.hist2d(
                t1[band],
                t2[band],
                bins=(np.linspace(16, 30, 101), np.linspace(16, 30, 101)),
                cmap="gray",
                norm=norm,
            )
        else:
            axs.hist2d(
                t1[band][mask],
                t2[band][mask],
                bins=(np.linspace(16, 30, 101), np.linspace(16, 30, 101)),
                cmap="gray",
                norm=norm,
            )
        axs.plot([16, 30], [16, 30])
        axs.set_xlabel(f"{band} [mag]")
        axs.set_ylabel(f"{band} [mag]")
    fig.tight_layout()


def plot_mag_mag_resid(t1, t2, bands, mask=None, norm="log"):
    fig = plt.figure()
    nrow, ncol = get_subplot_nrow_ncol(len(bands))
    axes = fig.subplots(nrow, ncol)
    for i, band in enumerate(bands):
        icol = i % ncol
        irow = int(i / ncol)
        axs = axes[irow][icol]
        if mask is None:
            axs.hist2d(
                t2[band],
                t1[band] - t2[band],
                bins=(np.linspace(16, 30, 101), np.linspace(-2, 2, 101)),
                cmap="gray",
                norm=norm,
            )
        else:
            axs.hist2d(
                t2[band][mask],
                t1[band][mask] - t2[band][mask],
                bins=(np.linspace(16, 30, 101), np.linspace(-2, 2, 101)),
                cmap="gray",
                norm=norm,
            )
        axs.set_xlabel(f"{band} [mag]")
        axs.set_ylabel(f"Delta {band} [mag]")
    fig.tight_layout()


def plot_color_scatter(t1, t2, bands, mask=None, norm="log"):
    fig = plt.figure()
    nrow, ncol = get_subplot_nrow_ncol(len(bands))
    axes = fig.subplots(nrow, ncol)
    for i, band in enumerate(bands):
        icol = i % ncol
        irow = int(i / ncol)
        axs = axes[irow][icol]
        if mask is None:
            axs.hist2d(
                t1[band],
                t2[band],
                bins=(np.linspace(-0.5, 2.0, 101), np.linspace(-0.5, 2.0, 101)),
                cmap="gray",
                norm=norm,
            )
        else:
            axs.hist2d(
                t1[band][mask],
                t2[band][mask],
                bins=(np.linspace(-0.5, 2.0, 101), np.linspace(-0.5, 2.0, 101)),
                cmap="gray",
                norm=norm,
            )
        axs.set_xlabel(f"{band} [mag]")
        axs.set_ylabel(f"{band} [mag]")
    fig.tight_layout()


def plot_color_resid(t1, t2, bands, mask=None, norm="log"):
    fig = plt.figure()
    nrow, ncol = get_subplot_nrow_ncol(len(bands))
    axes = fig.subplots(nrow, ncol)
    for i, band in enumerate(bands):
        icol = i % ncol
        irow = int(i / ncol)
        axs = axes[irow][icol]
        if mask is None:
            axs.hist2d(
                t2[band],
                t1[band] - t2[band],
                bins=(np.linspace(-0.5, 2.0, 101), np.linspace(-1.0, 1.0, 101)),
                cmap="gray",
                norm=norm,
            )
        else:
            axs.hist2d(
                t2[band][mask],
                (t1[band] - t2[band])[mask],
                bins=(np.linspace(-0.5, 2.0, 101), np.linspace(-1.0, 1.0, 101)),
                cmap="gray",
                norm=norm,
            )
        axs.set_xlabel(f"{band} [mag]")
        axs.set_ylabel(f"Delta {band} [mag]")
    fig.tight_layout()


def plot_DESI_scatter(t, flux_type, bands):
    fig = plt.figure()
    axes = fig.subplots(2, 2)
    for i, band in enumerate(bands):
        icol = i % 2
        irow = int(i / 2)
        axs = axes[irow][icol]
        axs.hist2d(
            t[f"{band}_{flux_type}Mag"],
            t[f"{band}_DESI"],
            bins=(np.linspace(16, 26, 101), np.linspace(16, 26, 101)),
            cmap="gray",
            norm="log",
        )
        axs.plot([16, 26], [16, 26])
        axs.set_xlabel(f"{band} {flux_type} [mag]")
        axs.set_ylabel(f"{band} DESI [mag]")
    fig.tight_layout()


def plot_DESI_resid(t, flux_type, bands):
    fig = plt.figure()
    axes = fig.subplots(2, 2)
    for i, band in enumerate(bands):
        icol = i % 2
        irow = int(i / 2)
        axs = axes[irow][icol]
        axs.hist2d(
            t[f"{band}_{flux_type}Mag"],
            t[f"{band}_DESI"] - t[f"{band}_{flux_type}Mag"],
            bins=(np.linspace(16, 26, 101), np.linspace(-2, 2, 101)),
            cmap="gray",
            norm="log",
        )
        axs.plot([16, 26], [16, 26])
        axs.set_xlabel(f"{band} {flux_type} [mag]")
        axs.set_ylabel(f"{band} DESI [mag]")
    fig.tight_layout()
