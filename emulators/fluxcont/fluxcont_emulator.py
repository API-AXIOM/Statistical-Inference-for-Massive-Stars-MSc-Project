"""Load and evaluate the FASTWIND continuum emulator (13 stellar and wind parameters).

It predicts the continuum spectral energy distribution between 1000 and
99,840 Angstrom. The trained network is one checkpoint file,
model/emulator_FLUXCONT.pth, which is not part of this repository;
emulators/README.md explains where to get it. Apart from the checkpoint,
this module needs only numpy and PyTorch.

Basic usage
-----------
    from fluxcont.fluxcont_emulator import FluxcontEmulator
    import numpy as np

    emu = FluxcontEmulator("model/emulator_FLUXCONT.pth")
    theta = [36000, 3.8, 12.0, 3e-6, 0.12, 8.0, 7.8, 8.6,
             1.2, 2200, 0.3, 0.5, 15.0]           # see PARAM_COLS
    lam = np.geomspace(*emu.lambda_range, 300)     # any grid inside range
    log_L = emu.predict_log_flux(theta, lam)       # log10 F_lambda
    L = emu.predict_flux(theta, lam)               # F_lambda itself

    # batched, e.g. MCMC walkers: theta shape (N, 13) -> output (N, P)

Output quantity
---------------
F_lambda here is a monochromatic luminosity L_lambda: the emergent FASTWIND
continuum flux F_nu, converted to per Angstrom and multiplied by the area
of the model's outer boundary, as in V. Kolympiris's MSc thesis (see
emulators/README.md):
    F_lambda = (3.00e18 / lambda^2) * 10**(log F_nu)
               * 4 pi (R_sun R_*)^2 * r_max^2      [erg s^-1 AA^-1]
with lambda in Angstrom, R_* the stellar radius (an input parameter) and
r_max the outer radius of the model atmosphere in units of R_*. Typical
values: log10 F_lambda about 27-37.
To compare with a DEREDDENED observed flux at distance d:
    f_lambda_observed = F_lambda / (4 pi d^2).

Parameter vector (exact order; mdot LINEAR in M_sun/yr):
    [teff, logg, radius, mdot, yhe, C, N, O,
     beta, vinf, fic, fvel, fclump]

Only valid inside param_bounds() and lambda_range. Largest errors occur
near the two wavelength edges; according to the emulator's author, the
typical accuracy is 1-2 per cent.
"""

import math
import os

import numpy as np
import torch
import torch.nn as nn

PARAM_COLS = ["teff", "logg", "radius", "mdot", "yhe", "C", "N", "O",
              "beta", "vinf", "fic", "fvel", "fclump"]


def _activation(name):
    key = str(name).lower()
    if key == "relu":
        return nn.ReLU()
    if key in ("leaky_relu", "leakyrelu"):
        return nn.LeakyReLU(negative_slope=0.01)
    if key in ("silu", "swish"):
        return nn.SiLU()
    if key == "gelu":
        return nn.GELU()
    raise ValueError(f"unknown activation {name}")


def _build_mlp(input_dim, output_dim, hidden_widths, activation_sequence,
               dropout, dropout_after_layer_indices):
    layers, in_dim = [], input_dim
    for idx, out_dim in enumerate(hidden_widths):
        layers.append(nn.Linear(in_dim, out_dim))
        act = activation_sequence[idx] if idx < len(activation_sequence) \
            else activation_sequence[-1]
        layers.append(_activation(act))
        if dropout > 0.0 and idx in dropout_after_layer_indices:
            layers.append(nn.Dropout(dropout))
        in_dim = out_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class _BranchNet(nn.Module):
    def __init__(self, ckpt):
        super().__init__()
        self.net = _build_mlp(len(ckpt["param_cols"]), ckpt["latent_dim"],
                              ckpt["branch_widths"],
                              ckpt["activation_sequence"], ckpt["dropout"],
                              ckpt["dropout_after_layer_indices"])

    def forward(self, x):
        return self.net(x)


class _TrunkNet(nn.Module):
    def __init__(self, ckpt):
        super().__init__()
        self.fourier_modes = int(ckpt["fourier_modes"])
        self.net = _build_mlp(2 * self.fourier_modes + 1, ckpt["latent_dim"],
                              ckpt["trunk_widths"],
                              ckpt["activation_sequence"], ckpt["dropout"],
                              ckpt["dropout_after_layer_indices"])

    def forward(self, x):
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
        self.branch = _BranchNet(ckpt)
        self.trunk = _TrunkNet(ckpt)

    def forward(self, params, coords):
        return (self.trunk(coords) * self.branch(params).unsqueeze(1)).sum(-1)


class FluxcontEmulator:
    """The FASTWIND continuum emulator."""

    def __init__(self, pth_path, device="cpu"):
        ckpt = torch.load(pth_path, map_location="cpu", weights_only=True)
        if ckpt.get("use_batch_norm") or ckpt.get("residual_blocks"):
            raise RuntimeError("this checkpoint uses batch normalisation or "
                               "residual blocks, which this loader does not "
                               "support")
        self.device = torch.device(device)
        self.model = _DeepONet(ckpt)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()

        self.param_cols = list(ckpt["param_cols"])
        self._mdot_idx = self.param_cols.index("mdot")
        # _p_min, _p_rng, _llog_min, _llog_max and the three _bin_ arrays are
        # also read by src/gpu_emulator.py
        self._p_min = np.asarray(ckpt["param_mins_after_mdot_log"], np.float64)
        self._p_rng = np.asarray(ckpt["param_range_after_mdot_log"], np.float64)

        self._llog_min = float(ckpt["lambda_log_min"])
        self._llog_max = float(ckpt["lambda_log_max"])
        self.lambda_range = (10.0 ** self._llog_min, 10.0 ** self._llog_max)

        self._bin_edges = np.asarray(ckpt["bin_edges_wavelength_log_norm"],
                                     np.float64)
        self._bin_means = np.asarray(ckpt["bin_mean_log_flam_train"], np.float64)
        self._bin_stds = np.asarray(ckpt["bin_std_log_flam_train"], np.float64)

    # -- parameter handling (identical conventions to the line emulators) --
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
    def predict_log_flux(self, theta, wavelengths, batch=2048,
                         check_bounds=False):
        """log10 F_lambda at the requested wavelengths (Angstrom).

        theta       : (13,) or (N, 13), natural units, mdot LINEAR.
        wavelengths : (P,) shared grid or (N, P). Inside lambda_range.
        returns     : (P,) or (N, P).
        """
        single = np.asarray(theta).ndim == 1
        pn = self._normalise(theta)
        n = pn.shape[0]

        lam = np.asarray(wavelengths, np.float64)
        if lam.ndim == 1:
            lam = np.broadcast_to(lam, (n, lam.size))
        if check_bounds:
            if not np.all(self.in_bounds(theta if not single else [theta])):
                raise ValueError("theta outside the trained parameter box")
            if (lam.min() < self.lambda_range[0] - 1e-6
                    or lam.max() > self.lambda_range[1] + 1e-6):
                raise ValueError(f"wavelengths outside {self.lambda_range} A")

        coords = (np.log10(lam) - self._llog_min) \
            / (self._llog_max - self._llog_min)
        bins = np.clip(np.digitize(coords, self._bin_edges) - 1,
                       0, self._bin_means.size - 1)
        mu, sd = self._bin_means[bins], self._bin_stds[bins]

        pn_t = torch.tensor(pn, dtype=torch.float32, device=self.device)
        co_t = torch.tensor(coords, dtype=torch.float32, device=self.device)
        y = np.empty(lam.shape, dtype=np.float64)
        for i in range(0, n, batch):
            out = self.model(pn_t[i:i + batch], co_t[i:i + batch])
            y[i:i + batch] = out.cpu().numpy()
        log_flam = y * sd + mu
        return log_flam[0] if single else log_flam

    def predict_flux(self, theta, wavelengths, **kw):
        """F_lambda in erg s^-1 Angstrom^-1 (see module docstring)."""
        return 10.0 ** self.predict_log_flux(theta, wavelengths, **kw)


if __name__ == "__main__":
    here = os.path.dirname(__file__) or "."
    emu = FluxcontEmulator(os.path.join(here, "model", "emulator_FLUXCONT.pth"))
    theta = [36000, 3.8, 12.0, 3e-6, 0.12, 8.0, 7.8, 8.6,
             1.2, 2200, 0.3, 0.5, 15.0]
    lam = np.geomspace(*emu.lambda_range, 200)
    logf = emu.predict_log_flux(theta, lam, check_bounds=True)
    print(f"lambda range {emu.lambda_range[0]:.0f} to "
          f"{emu.lambda_range[1]:.0f} A")
    print(f"log10 F_lambda from {logf.min():.2f} to {logf.max():.2f}")
