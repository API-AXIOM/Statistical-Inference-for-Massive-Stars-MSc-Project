"""Load and evaluate the FASTWIND line emulators (13 stellar and wind parameters).

Each spectral line window has its own trained network, stored as one
checkpoint file (emulator_OUT.<line>_VTV010.zst.pth) in the models/ folder.
The checkpoint files are not part of this repository; emulators/README.md
explains where to get them. Apart from the checkpoints, this module needs
only numpy and PyTorch.

Basic usage
-----------
    from line_emulators_13par.line_emulator import LineEmulator, load_all_lines

    emu = LineEmulator("models/emulator_OUT.HALPHAHEII6527_VTV010.zst.pth")
    theta = [36000, 3.8, 12.0, 3e-6, 0.12, 8.0, 7.8, 8.6,
             1.2, 2200, 0.3, 0.5, 15.0]          # see PARAM_COLS
    lam = np.linspace(*emu.lambda_range, 400)     # any grid inside range
    flux = emu.predict(theta, lam)                # normalised flux, (400,)

    # batched, e.g. for MCMC walkers: theta_batch shape (N, 13)
    flux_batch = emu.predict(theta_batch, lam)    # (N, 400)

    emus = load_all_lines("models")               # every line window
    emus["HBETA"].predict(theta, lam_hbeta)       # look one up by its name

Parameter vector (this exact order; mdot in LINEAR M_sun/yr):
    [teff, logg, radius, mdot, yhe, C, N, O,
     beta, vinf, fic, fvel, fclump]

The prediction is the continuum-normalised flux (continuum = 1). It
contains no rotational / macroturbulent / instrumental broadening and no
radial-velocity shift; apply those yourself before comparing with data.
Predictions are only valid inside param_bounds() and lambda_range.
"""

import glob
import math
import os
import warnings

import numpy as np
import torch
import torch.nn as nn

PARAM_COLS = ["teff", "logg", "radius", "mdot", "yhe", "C", "N", "O",
              "beta", "vinf", "fic", "fvel", "fclump"]

# Recommended fitting intervals for lines whose stored lambda_range is much
# wider than the region actually constrained by the training data. The
# CIV1169b window spans 800-1400 A but each FASTWIND model has only 161
# native points there, almost all around the C IV 1169 / C III 1176 blend,
# so dense evaluation of the far continuum produces spurious oscillations
# (interpolation artifacts, not spectral features). Predict() warns when a
# requested grid extends beyond the recommended interval of such a line.
RECOMMENDED_FIT_RANGES = {
    "CIV1169b": (1140.0, 1210.0),
}


# ---------------- network definition (must match training) ----------------
def _build_mlp(input_dim, output_dim, hidden_widths):
    layers, in_dim = [], input_dim
    for out_dim in hidden_widths:
        layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        in_dim = out_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class _BranchNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_widths):
        super().__init__()
        self.net = _build_mlp(input_dim, output_dim, hidden_widths)

    def forward(self, x):
        return self.net(x)


class _TrunkNet(nn.Module):
    def __init__(self, output_dim, fourier_modes, hidden_widths):
        super().__init__()
        self.fourier_modes = fourier_modes
        self.net = _build_mlp(2 * fourier_modes + 1, output_dim, hidden_widths)

    def forward(self, x):                      # x: (N, P) normalised in [0,1]
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        batch, npts, _ = x.shape
        freqs = 2.0 * math.pi * torch.arange(
            1, self.fourier_modes + 1, dtype=x.dtype, device=x.device
        ).view(1, 1, -1)
        feats = torch.cat(
            [torch.ones(batch, npts, 1, dtype=x.dtype, device=x.device),
             torch.sin(freqs * x), torch.cos(freqs * x)], dim=-1)
        out = self.net(feats.view(-1, feats.shape[-1]))
        return out.view(batch, npts, -1)


class _DeepONet(nn.Module):
    def __init__(self, ckpt):
        super().__init__()
        self.branch = _BranchNet(len(ckpt["param_cols"]), ckpt["latent_dim"],
                                 ckpt["branch_widths"])
        self.trunk = _TrunkNet(ckpt["latent_dim"], int(ckpt["fourier_modes"]),
                               ckpt["trunk_widths"])

    def forward(self, params, coords):
        return (self.trunk(coords) * self.branch(params).unsqueeze(1)).sum(-1)


# ---------------- loading and evaluating the emulators ----------------
class LineEmulator:
    """One trained emulator for one diagnostic line window."""

    def __init__(self, pth_path, device="cpu"):
        ckpt = torch.load(pth_path, map_location="cpu", weights_only=True)
        self.device = torch.device(device)
        self.model = _DeepONet(ckpt)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()

        self.line = os.path.basename(pth_path)[len("emulator_"):-len(".pth")]
        self.param_cols = list(ckpt["param_cols"])
        self._mdot_idx = self.param_cols.index("mdot")
        # _p_min, _p_rng and _flux_offset are also read by src/gpu_emulator.py
        self._p_min = np.asarray(ckpt["param_mins_after_mdot_log"], np.float64)
        self._p_rng = np.asarray(ckpt["param_range_after_mdot_log"], np.float64)
        self.lambda_range = (float(ckpt["lambda_min"]), float(ckpt["lambda_max"]))
        self._flux_offset = float(ckpt.get("flux_offset", 1.0))

        rec = RECOMMENDED_FIT_RANGES.get(short_name(self.line))
        if rec is None:
            self.recommended_lambda_range = self.lambda_range
        else:
            self.recommended_lambda_range = (
                max(self.lambda_range[0], rec[0]),
                min(self.lambda_range[1], rec[1]))
        self._warned_recommended = False

    # -- parameter handling -------------------------------------------------
    def param_bounds(self):
        """dict name -> (min, max) in NATURAL units (mdot linear M_sun/yr)."""
        lo = self._p_min.copy()
        hi = (self._p_min + self._p_rng).copy()
        bounds = {}
        for k, name in enumerate(self.param_cols):
            a, b = lo[k], hi[k]
            if k == self._mdot_idx:
                a, b = 10.0 ** a, 10.0 ** b
            bounds[name] = (float(a), float(b))
        return bounds

    def in_bounds(self, theta):
        """True where a parameter vector lies inside the trained box.
        theta: (13,) or (N, 13), natural units. Returns bool or (N,) bools."""
        t = np.atleast_2d(np.asarray(theta, np.float64)).copy()
        t[:, self._mdot_idx] = np.log10(t[:, self._mdot_idx])
        n = (t - self._p_min) / self._p_rng
        ok = np.all((n >= -1e-9) & (n <= 1 + 1e-9), axis=1)
        return ok if np.asarray(theta).ndim == 2 else bool(ok[0])

    def _normalise(self, theta):
        t = np.atleast_2d(np.asarray(theta, np.float64)).copy()
        if t.shape[1] != len(self.param_cols):
            raise ValueError(f"theta must have {len(self.param_cols)} columns "
                             f"in the order {self.param_cols}")
        t[:, self._mdot_idx] = np.log10(t[:, self._mdot_idx])
        return (t - self._p_min) / self._p_rng

    # -- prediction ---------------------------------------------------------
    @torch.no_grad()
    def predict(self, theta, wavelengths, batch=4096, check_bounds=False):
        """Continuum-normalised flux.

        theta       : (13,) or (N, 13) parameter vectors, natural units,
                      order PARAM_COLS, mdot LINEAR in M_sun/yr.
        wavelengths : (P,) physical wavelengths in Angstrom, shared by all
                      vectors, or (N, P) one grid per vector. Must lie
                      inside self.lambda_range.
        returns     : (P,) if theta was 1-D, else (N, P).
        """
        single = np.asarray(theta).ndim == 1
        pn = self._normalise(theta)
        n = pn.shape[0]

        lam = np.asarray(wavelengths, np.float64)
        if lam.ndim == 1:
            lam = np.broadcast_to(lam, (n, lam.size))
        lmin, lmax = self.lambda_range
        if check_bounds:
            if not np.all(self.in_bounds(theta if not single else [theta])):
                raise ValueError("theta outside the trained parameter box")
            if lam.min() < lmin - 1e-6 or lam.max() > lmax + 1e-6:
                raise ValueError(f"wavelengths outside [{lmin}, {lmax}] A")
        rmin, rmax = self.recommended_lambda_range
        if (not self._warned_recommended
                and (rmin, rmax) != self.lambda_range
                and (lam.min() < rmin - 1e-6 or lam.max() > rmax + 1e-6)):
            warnings.warn(
                f"{short_name(self.line)}: requested wavelengths extend "
                f"beyond the recommended interval [{rmin:.0f}, {rmax:.0f}] A. "
                "Outside it the sparse training sampling of this broad "
                "window is unconstrained and dense grids show spurious "
                "oscillations; restrict the evaluation to the recommended "
                "interval.")
            self._warned_recommended = True
        ln = (lam - lmin) / (lmax - lmin)

        pn_t = torch.tensor(pn, dtype=torch.float32, device=self.device)
        ln_t = torch.tensor(ln, dtype=torch.float32, device=self.device)
        out = np.empty(lam.shape, dtype=np.float32)
        for i in range(0, n, batch):
            mu = self.model(pn_t[i:i + batch], ln_t[i:i + batch])
            out[i:i + batch] = mu.cpu().numpy() + self._flux_offset
        return out[0] if single else out


def short_name(line):
    name = str(line)
    if name.endswith(".zst"):
        name = name[:-4]
    if name.startswith("OUT."):
        name = name[len("OUT."):]
    for suffix in ("_VTV010", "_VT010"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


def load_all_lines(models_dir="models", device="cpu"):
    """Load every checkpoint in models_dir. Returns {short_name: LineEmulator}."""
    out = {}
    for p in sorted(glob.glob(os.path.join(models_dir, "emulator_*.pth"))):
        emu = LineEmulator(p, device=device)
        out[short_name(emu.line)] = emu
    if not out:
        raise FileNotFoundError(f"no emulator_*.pth found in {models_dir}")
    return out


if __name__ == "__main__":
    emus = load_all_lines(os.path.join(os.path.dirname(__file__) or ".",
                                       "models"))
    print(f"loaded {len(emus)} line emulators: {sorted(emus)}")
    emu = emus.get("HALPHAHEII6527", next(iter(emus.values())))
    theta = [36000, 3.8, 12.0, 3e-6, 0.12, 8.0, 7.8, 8.6,
             1.2, 2200, 0.3, 0.5, 15.0]
    lam = np.linspace(*emu.lambda_range, 200)
    flux = emu.predict(theta, lam, check_bounds=True)
    print(f"{short_name(emu.line)}: lambda {emu.lambda_range}, "
          f"flux range {flux.min():.3f} to {flux.max():.3f}")
