"""
Generate lat-lon zonal-wind (u) movies for the Jupiter GCM runs under
notebooks/jupiter/runs/, for embedding in the results log notebook.

One movie per run, at a fixed representative sigma level (closest to 0.5,
i.e. roughly mid-atmosphere) for cross-run comparability. Color scale is
fixed from the first N_STABLE_DAYS of each run so a late blow-up saturates
the color scale rather than washing out the whole movie.
"""

import glob
import os

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

RUNS_DIR = "notebooks/jupiter/runs"
MOVIES_DIR = "notebooks/jupiter/movies"
os.makedirs(MOVIES_DIR, exist_ok=True)

RUNS = [
    "lian_showman_t85_1000day",
    "lian_showman_t85_halfdt_1000day",
    "lian_showman_t85_shorttau_1000day",
    "lian_showman_t85_highorder_1000day",
    "lian_showman_t85_hdiff_1000day",
    "lian_showman_t213_1000day",
    "lian_showman_t85_stability_test",
    "held_suarez_jupiter_t85_1000day",
    "held_suarez_jupiter_lowdty_t85_1000day",
]

N_STABLE_DAYS = 150  # window used to set the color scale
FPS = 8


def make_movie(run_name):
    files = sorted(glob.glob(f"{RUNS_DIR}/{run_name}/*_chunk_*.nc"))
    if not files:
        print(f"[skip] {run_name}: no chunk files found")
        return
    ds = xr.open_mfdataset(files, combine="by_coords")
    ds = ds.sortby("time")

    level_idx = int(np.argmin(np.abs(ds.level.values - 0.5)))
    sigma_val = float(ds.level.values[level_idx])

    u = ds["u"].isel(level=level_idx).transpose("time", "lat", "lon")
    times = ds.time.values
    lon = ds.lon.values
    lat = ds.lat.values

    stable_mask = times <= N_STABLE_DAYS
    stable_vals = u.isel(time=stable_mask).values
    finite = stable_vals[np.isfinite(stable_vals)]
    vmax = float(np.nanpercentile(np.abs(finite), 99.5)) if finite.size else 50.0
    vmax = max(vmax, 1.0)

    n_finite_total = 0
    first_nan_day = None

    fig, ax = plt.subplots(figsize=(8, 4), dpi=110)
    pcm = ax.pcolormesh(
        lon, lat, np.zeros((len(lat), len(lon))),
        cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto",
    )
    cbar = fig.colorbar(pcm, ax=ax, label="u [m/s]", shrink=0.85)
    title = ax.set_title("")
    ax.set_xlabel("longitude [deg]")
    ax.set_ylabel("latitude [deg]")
    fig.tight_layout()

    out_path = f"{MOVIES_DIR}/{run_name}_u.mp4"
    writer = imageio.get_writer(out_path, fps=FPS, codec="libx264", quality=7, macro_block_size=None)

    for i, t in enumerate(times):
        frame = u.isel(time=i).values
        finite_frame = np.isfinite(frame)
        if finite_frame.any():
            n_finite_total += 1
        elif first_nan_day is None:
            first_nan_day = int(t)
        pcm.set_array(np.ma.masked_invalid(frame))
        title.set_text(f"{run_name}  |  u @ sigma={sigma_val:.3f}  |  day {int(t)}")
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        writer.append_data(img)

    writer.close()
    plt.close(fig)

    status = f"NaN from day {first_nan_day}" if first_nan_day is not None else "stayed finite throughout"
    print(f"[ok] {run_name}: {len(times)} frames, level sigma={sigma_val:.3f}, vmax={vmax:.1f} m/s, {status} -> {out_path}")


if __name__ == "__main__":
    for run in RUNS:
        try:
            make_movie(run)
        except Exception as e:
            print(f"[error] {run}: {e}")
