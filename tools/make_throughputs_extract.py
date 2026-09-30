"""Rubin throughput extract for Phys 367 Week 2 Notebook B ("Camera and telescope -> science").

Downloads the pinned component files of ``lsst-pst/syseng_throughputs`` (commit
``SYSENG_SHA``, release 1.9, triple-silver mirrors) by raw GitHub URL and writes two
small CSV files under ``week2/data/`` for the Colab notebook:

- ``throughputs.csv``: 300-1100 nm on a 1 nm grid. Columns
    wavelength_nm
    mirrors      -- M1*M2*M3 reflectance (protected silver, 6 deg), each x its loss curve
    lenses       -- L1*L2*L3: smoothed glass x BBAR coatings x losses, per lens
    detector     -- joint-minimum QE (``detector/joint_minimum/minDet_QE.dat``) x losses
    filter_u ... filter_y -- filter response x filter losses
    atmos_X1.0   -- ``siteProperties/atmos_10_aerosol.dat`` (X = 1.0)
    atmos_X1.2   -- ``siteProperties/pachonModtranAtm_12_aerosol.dat`` (X = 1.2)
  Hardware in band b = mirrors * lenses * detector * filter_b;
  system = hardware * atmos.
- ``darksky.csv``: the dark-sky spectrum ``siteProperties/darksky.dat``
  (wavelength in nm, F_lambda in erg s^-1 cm^-2 nm^-1), averaged into 1 nm bins
  centered on the same 300-1100 nm grid.

How syseng combines the components (``bandpassUtils.buildHardwareAndSystem(
setDefaultDirs(), addLosses=True)`` at the pinned commit), replicated here in numpy:

1. Every file is read with rubin_sim's ``Bandpass.read_throughput``: lines starting with
   ``#`` are skipped *entirely* (several syseng files put the first data point on the
   header line, so that point is dropped), the rest is sorted by wavelength.
2. Every curve is linearly interpolated onto a 300-1150 nm, 0.1 nm grid, with zero
   outside the tabulated range (``scipy.interpolate.interp1d(fill_value=0)``). Loss
   files are tabulated only over 300-1100 nm, so all hardware is zero above 1100 nm.
3. A ``*_Losses/`` (or ``*_Coatings/``) directory is the product of all its files.
4. Detector: ``setDefaultDirs`` points at ``detector/joint_minimum``, which holds a single
   ``minDet_QE.dat`` (already the minimum of the ITL and e2v vendor curves), so
   ``buildDetector`` treats it as one vendor: QE x ``joint_Losses/det_Losses.dat``.
5. Lens: the glass curve is smoothed with a Savitzky-Golay filter (window 31 samples on
   the 0.1 nm grid, cubic), then multiplied by the BBAR coating and the losses.
6. Mirror: ``m*_ProtAg_6deg.dat`` x ``m*_Losses/``. Filter: ``<b>_band_Response.dat`` x
   the product of the four ``filter_Losses/`` files.
7. Each component is clipped at 0 (values below -0.02 would be an error).
8. hardware_b = detector * L1 * L2 * L3 * M1 * M2 * M3 * filter_b; system_b = hardware_b *
   atmosphere.

Atmosphere: ``readAtmosphere``'s code default is ``pachonModtranAtm_12_aerosol.dat``,
which is the X = 1.2 atmosphere. For X = 1.0, syseng's own "Overview Paper" and "Repo
Demo" notebooks pass ``atmosphereOverride=readAtmosphere(..., 'atmos_10_aerosol.dat')``;
that is the X = 1.0 curve used here, and the one used for the reference m5 table
(``tools/make_m5_reference.py``).

The 1 nm columns are the 0.1 nm products sampled at integer wavelengths (the component
tables are themselves tabulated at 1 nm).

Usage:  python tools/make_throughputs_extract.py week2/data
"""

import argparse
import io
import pathlib
import urllib.request

import numpy as np
import pandas as pd

SYSENG_SHA = "00570b3d391b5a8671d55341ed51b5e534dab6b4"
RAW = f"https://raw.githubusercontent.com/lsst-pst/syseng_throughputs/{SYSENG_SHA}/"
BANDS = ("u", "g", "r", "i", "z", "y")

# syseng internal grid (bandpassUtils.WAVELEN_MIN/MAX/STEP)
FINE = np.arange(300, 1150 + 0.1 / 2.0, 0.1, dtype=float)
GRID = np.arange(300, 1101)          # output grid, nm
BELOW_ZERO = -0.02

# Files in each directory syseng globs, as listed in the git tree at SYSENG_SHA.
CAM = "components/camera/"
TEL = "components/telescope/"
DETECTOR = (CAM + "detector/joint_minimum/minDet_QE.dat",
            [CAM + "detector/joint_minimum/joint_Losses/det_Losses.dat"])
LENSES = {
    n: (CAM + f"lens{n}/l{n}_Glass.dat",
        [CAM + f"lens{n}/l{n}_Coatings/{'L' if n != 2 else 'l'}{n}_BBAR.dat"],
        [CAM + f"lens{n}/l{n}_Losses/l{n}_{s}_{k}.dat"
         for s in ("S1", "S2") for k in ("Condensation", "Contamination")])
    for n in (1, 2, 3)
}
MIRRORS = {n: (TEL + f"mirror{n}/m{n}_ProtAg_6deg.dat", [TEL + f"mirror{n}/m{n}_Losses/m{n}_Losses.dat"])
           for n in (1, 2, 3)}
FILTERS = {b: CAM + f"filters/{b}_band_Response.dat" for b in BANDS}
FILTER_LOSSES = [CAM + f"filters/filter_Losses/filter_{s}_{k}.dat"
                 for s in ("S1", "S2") for k in ("Condensation", "Contamination")]
ATMOS = {"X1.0": "siteProperties/atmos_10_aerosol.dat",
         "X1.2": "siteProperties/pachonModtranAtm_12_aerosol.dat"}
DARKSKY = "siteProperties/darksky.dat"


# ---------------------------------------------------------------------------
# Core functions (mirror rubin_sim Bandpass / syseng bandpassUtils)
# ---------------------------------------------------------------------------
def read_throughput_text(text):
    """Parse a two-column table the way rubin_sim ``Bandpass.read_throughput`` does."""
    w, sb = [], []
    for line in io.StringIO(text):
        if line.startswith(("#", "$", "!")):
            continue
        v = line.split()
        if len(v) < 2 or v[0] in ("$", "#", "!"):
            continue
        w.append(float(v[0]))
        sb.append(float(v[1]))
    w, sb = np.array(w), np.array(sb)
    if len(w) != len(np.unique(w)):
        raise ValueError("non-unique wavelengths")
    p = np.argsort(w)
    return w[p], sb[p]


def resample(w, sb, grid):
    """Linear interpolation onto ``grid``, zero outside the tabulated range."""
    return np.interp(grid, w, sb, left=0.0, right=0.0)


def savitzky_golay(y, window_size=31, order=3):
    """syseng ``savitzky_golay`` (Chuck Claver's glass smoothing), deriv=0."""
    half = (window_size - 1) // 2
    b = np.array([[k**i for i in range(order + 1)] for k in range(-half, half + 1)], dtype=float)
    m = np.linalg.pinv(b)[0]
    first = y[0] - np.abs(y[1:half + 1][::-1] - y[0])
    last = y[-1] + np.abs(y[-half - 1:-1][::-1] - y[-1])
    return np.convolve(m[::-1], np.concatenate((first, y, last)), mode="valid")


def _clip(sb, what):
    if np.any(sb < BELOW_ZERO):
        raise ValueError(f"{what}: values significantly below zero")
    return np.where(sb < 0, 0.0, sb) + 0.0      # "+ 0.0" turns -0.0 into 0.0


def build_components(fetch):
    """Return a dict of component curves on the 0.1 nm grid ``FINE``.

    ``fetch(path) -> str`` returns the text of a repository file.
    """
    def curve(path):
        return resample(*read_throughput_text(fetch(path)), FINE)

    def product(paths):
        out = np.ones_like(FINE)
        for p in paths:
            out = out * curve(p)
        return out

    comp = {}
    qe, det_losses = DETECTOR
    comp["detector"] = _clip(curve(qe) * product(det_losses), "detector")

    lenses = np.ones_like(FINE)
    for n, (glass, coatings, losses) in LENSES.items():
        lens = savitzky_golay(curve(glass), 31, 3) * product(coatings) * product(losses)
        lenses = lenses * _clip(lens, f"lens{n}")
    comp["lenses"] = lenses

    mirrors = np.ones_like(FINE)
    for n, (refl, losses) in MIRRORS.items():
        mirrors = mirrors * _clip(curve(refl) * product(losses), f"mirror{n}")
    comp["mirrors"] = mirrors

    floss = product(FILTER_LOSSES)
    for b, path in FILTERS.items():
        comp[f"filter_{b}"] = _clip(curve(path) * floss, f"filter {b}")
    for key, path in ATMOS.items():
        comp[f"atmos_{key}"] = _clip(curve(path), f"atmosphere {key}")
    return comp


def to_output_grid(fine_curve):
    """Sample a 0.1 nm-grid curve at the integer wavelengths of ``GRID``."""
    idx = np.rint((GRID - FINE[0]) / 0.1).astype(int)
    return fine_curve[idx]


def darksky_1nm(text):
    """Average the dark-sky F_lambda into 1 nm bins centered on ``GRID``."""
    w, fl = read_throughput_text(text)
    lo = GRID - 0.5
    out = np.empty(len(GRID))
    for i, a in enumerate(lo):
        sel = (w >= a - 1e-6) & (w < a + 1.0 - 1e-6)
        out[i] = fl[sel].mean()
    return out


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def fetch_raw(path):
    with urllib.request.urlopen(RAW + path, timeout=60) as r:
        return r.read().decode()


def _write_csv(df, path, header_lines, fmt):
    with open(path, "w") as f:
        for line in header_lines:
            f.write(f"# {line}\n")
        df.to_csv(f, index=False, float_format=fmt, lineterminator="\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=str, help="Output directory (e.g. week2/data)")
    args = parser.parse_args(argv)
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    comp = build_components(fetch_raw)
    cols = ["mirrors", "lenses", "detector"] + [f"filter_{b}" for b in BANDS] + ["atmos_X1.0", "atmos_X1.2"]
    t = pd.DataFrame({"wavelength_nm": GRID})
    for c in cols:
        t[c] = to_output_grid(comp[c])
    src = f"github.com/lsst-pst/syseng_throughputs @ {SYSENG_SHA} (release 1.9, triple-silver mirrors)"
    _write_csv(t, out / "throughputs.csv", [
        "Rubin throughput components, fractional (0-1), 300-1100 nm on a 1 nm grid.",
        f"Source: {src}",
        "Built by tools/make_throughputs_extract.py, replicating syseng buildHardwareAndSystem(addLosses=True).",
        "mirrors = M1*M2*M3 (with losses); lenses = L1*L2*L3 (glass x BBAR x losses);",
        "detector = joint-minimum QE x losses; filter_<b> = response x losses.",
        "atmos_X1.0 = atmos_10_aerosol.dat; atmos_X1.2 = pachonModtranAtm_12_aerosol.dat.",
        "hardware_b = mirrors*lenses*detector*filter_b ; system_b = hardware_b*atmos.",
    ], "%.6g")

    s = pd.DataFrame({"wavelength_nm": GRID, "flambda_erg_s_cm2_nm": darksky_1nm(fetch_raw(DARKSKY))})
    _write_csv(s, out / "darksky.csv", [
        "Rubin fiducial dark-sky spectrum at zenith: F_lambda in erg s^-1 cm^-2 nm^-1 per arcsec^2.",
        f"Source: siteProperties/darksky.dat, {src}",
        "Averaged into 1 nm bins centered on each wavelength_nm (tools/make_throughputs_extract.py).",
    ], "%.5e")
    for p in ("throughputs.csv", "darksky.csv"):
        print(f"wrote {out / p} ({(out / p).stat().st_size / 1024:.1f} kB)")


if __name__ == "__main__":
    main()
