"""
Inference helper for the FASTWIND HG per-line emulators.

What to send to a colleague:
1. This file: `emulator_inference.py`
2. The checkpoint folder: `emulators_per_line_hg/`

Why this file is needed even if the `.pth` checkpoints already exist:
- The checkpoints store the learned weights and metadata.
- PyTorch still needs the network definition in Python to rebuild the model
  before those weights can be loaded.

Main public functions:
- `emulate_hg_spectrum(...)`: return one prediction per line.
- `emulate_hg_spectrum_from_indat(...)`: read FASTWIND parameters from INDAT,
  call the per-line emulators, and merge them into one unified spectrum.

Default behavior:
- If no wavelength grid is supplied, each line is evaluated on a uniform grid
  of `num_points` points between the checkpoint's saved `lambda_min` and
  `lambda_max`.

Custom wavelength grids:
- If the user wants a specific wavelength grid for one or more lines, pass
  `wavelengths_by_line`.
- The keys must be the line names, for example `OUT.HGAMMA_VTV010`.
- The values must be arrays/lists of wavelengths in Angstrom.

Example:
    import numpy as np
    from emulator_inference import emulate_hg_spectrum_from_indat

    spectrum = emulate_hg_spectrum_from_indat(
        "path/to/model/INDAT",
        emulator_dir="emulators_per_line_hg",
        output_wavelength=np.linspace(4000.0, 7000.0, 5000),
    )
"""

import glob
import math
import os
from typing import Dict, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn


USE_RESIDUAL = True
FLUX_OFFSET = 1.0
DEFAULT_PARAM_ORDER = ("Teff", "logg", "R", "Mdot", "v_inf", "Y_He", "v_turb")
__all__ = [
    "default_wavelength_grid",
    "emulate_hg_spectrum",
    "emulate_hg_spectrum_from_indat",
    "emulate_line",
    "merge_line_spectra",
    "normalize_params",
    "read_indat_params",
]


class BranchNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrunkNet(nn.Module):
    def __init__(self, output_dim: int = 128, fourier_modes: int = 32):
        super().__init__()
        self.fourier_modes = fourier_modes
        in_dim = 2 * fourier_modes + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        batch_size, num_points, _ = x.shape
        freqs = 2.0 * math.pi * torch.arange(
            1, self.fourier_modes + 1, device=x.device, dtype=x.dtype
        ).view(1, 1, self.fourier_modes)

        sin_feats = torch.sin(freqs * x)
        cos_feats = torch.cos(freqs * x)
        ones = torch.ones(batch_size, num_points, 1, device=x.device, dtype=x.dtype)
        feats = torch.cat([ones, sin_feats, cos_feats], dim=-1)
        feats = feats.view(-1, feats.shape[-1])
        out = self.net(feats)
        return out.view(batch_size, num_points, -1)


class BranchNet128(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrunkNet128(nn.Module):
    def __init__(self, output_dim: int = 128, fourier_modes: int = 32):
        super().__init__()
        self.fourier_modes = fourier_modes
        in_dim = 2 * fourier_modes + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        batch_size, num_points, _ = x.shape
        freqs = 2.0 * math.pi * torch.arange(
            1, self.fourier_modes + 1, device=x.device, dtype=x.dtype
        ).view(1, 1, self.fourier_modes)

        sin_feats = torch.sin(freqs * x)
        cos_feats = torch.cos(freqs * x)
        ones = torch.ones(batch_size, num_points, 1, device=x.device, dtype=x.dtype)
        feats = torch.cat([ones, sin_feats, cos_feats], dim=-1)
        feats = feats.view(-1, feats.shape[-1])
        out = self.net(feats)
        return out.view(batch_size, num_points, -1)


class BranchNet64(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrunkNet64(nn.Module):
    def __init__(self, output_dim: int = 64, fourier_modes: int = 32):
        super().__init__()
        self.fourier_modes = fourier_modes
        in_dim = 2 * fourier_modes + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(-1)

        batch_size, num_points, _ = x.shape
        freqs = 2.0 * math.pi * torch.arange(
            1, self.fourier_modes + 1, device=x.device, dtype=x.dtype
        ).view(1, 1, self.fourier_modes)

        sin_feats = torch.sin(freqs * x)
        cos_feats = torch.cos(freqs * x)
        ones = torch.ones(batch_size, num_points, 1, device=x.device, dtype=x.dtype)
        feats = torch.cat([ones, sin_feats, cos_feats], dim=-1)
        feats = feats.view(-1, feats.shape[-1])
        out = self.net(feats)
        return out.view(batch_size, num_points, -1)


class DeepONetModel(nn.Module):
    def __init__(self, branch_net: nn.Module, trunk_net: nn.Module):
        super().__init__()
        self.branch = branch_net
        self.trunk = trunk_net

    def forward(self, params: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        branch_out = self.branch(params)
        trunk_out = self.trunk(coords)
        return (trunk_out * branch_out.unsqueeze(1)).sum(-1)


def _as_param_vector(
    params: Union[Sequence[float], Mapping[str, float]],
    param_order: Sequence[str] = DEFAULT_PARAM_ORDER,
) -> np.ndarray:
    if isinstance(params, Mapping):
        return np.asarray([params[name] for name in param_order], dtype=np.float32)
    return np.asarray(params, dtype=np.float32)


def _resolve_indat_path(indat_path_or_model_dir: str) -> str:
    """Return the INDAT file path from either a file path or a model directory."""
    if os.path.isfile(indat_path_or_model_dir):
        return indat_path_or_model_dir

    if not os.path.isdir(indat_path_or_model_dir):
        raise FileNotFoundError(f"INDAT path or model directory not found: {indat_path_or_model_dir}")

    for filename in ("INDAT", "INDAT.DAT"):
        candidate = os.path.join(indat_path_or_model_dir, filename)
        if os.path.isfile(candidate):
            return candidate

    raise FileNotFoundError(f"No INDAT or INDAT.DAT found in {indat_path_or_model_dir}")


def read_indat_params(
    indat_path_or_model_dir: str,
    default_v_turb: Optional[float] = None,
) -> Dict[str, float]:
    """
    Read the seven emulator parameters from a FASTWIND INDAT/INDAT.DAT file.

    Parameters are returned in physical units and with the names expected by
    `emulate_hg_spectrum`: `Teff`, `logg`, `R`, `Mdot`, `v_inf`, `Y_He`,
    and `v_turb`.
    """
    indat_path = _resolve_indat_path(indat_path_or_model_dir)
    with open(indat_path, encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    try:
        teff, logg, radius = map(float, lines[3].split()[:3])
        mdot, _vmin, v_inf, _beta, _vtrans = map(float, lines[5].split()[:5])
        y_he = float(lines[6].split()[0])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Could not parse FASTWIND parameters from {indat_path}") from exc

    v_turb = default_v_turb
    if len(lines) > 8:
        try:
            v_turb = float(lines[8].split()[0])
        except (IndexError, ValueError):
            pass

    if v_turb is None:
        raise ValueError(
            f"Could not parse v_turb from {indat_path}; pass default_v_turb if it is fixed."
        )

    return {
        "Teff": teff,
        "logg": logg,
        "R": radius,
        "Mdot": mdot,
        "v_inf": v_inf,
        "Y_He": y_he,
        "v_turb": v_turb,
    }


def normalize_params(
    params: Union[Sequence[float], Mapping[str, float]],
    param_mins: Sequence[float],
    param_maxs: Sequence[float],
    param_order: Sequence[str] = DEFAULT_PARAM_ORDER,
) -> np.ndarray:
    """Normalize the 7 stellar parameters exactly as in the training notebook."""
    params_vec = _as_param_vector(params, param_order=param_order)
    param_mins = np.asarray(param_mins, dtype=np.float32)
    param_maxs = np.asarray(param_maxs, dtype=np.float32)

    if params_vec.shape[0] != len(param_order):
        raise ValueError(f"Expected {len(param_order)} parameters, got {params_vec.shape[0]}.")

    param_range = np.where(param_maxs - param_mins == 0.0, 1.0, param_maxs - param_mins)
    params_norm = (params_vec - param_mins) / param_range

    mdot_idx = param_order.index("Mdot")
    log_mdot = np.log10(params_vec[mdot_idx])
    log_mdot_min = np.log10(param_mins[mdot_idx])
    log_mdot_max = np.log10(param_maxs[mdot_idx])
    log_mdot_range = 1.0 if log_mdot_max == log_mdot_min else (log_mdot_max - log_mdot_min)
    params_norm[mdot_idx] = (log_mdot - log_mdot_min) / log_mdot_range
    return params_norm.astype(np.float32)


def _pick_architecture(config: Mapping[str, object]):
    architecture_name = config.get("architecture_name", "original")
    latent_dim = int(config.get("latent_dim", 128))
    fourier_modes = int(config.get("fourier_modes", 32))

    if architecture_name == "original":
        branch = BranchNet(input_dim=7, output_dim=latent_dim)
        trunk = TrunkNet(output_dim=latent_dim, fourier_modes=fourier_modes)
    elif architecture_name == "latent128":
        branch = BranchNet128(input_dim=7, output_dim=latent_dim)
        trunk = TrunkNet128(output_dim=latent_dim, fourier_modes=fourier_modes)
    elif architecture_name == "latent64":
        branch = BranchNet64(input_dim=7, output_dim=latent_dim)
        trunk = TrunkNet64(output_dim=latent_dim, fourier_modes=fourier_modes)
    else:
        raise ValueError(f"Unsupported architecture_name: {architecture_name}")

    return DeepONetModel(branch, trunk)

_CHECKPOINT_CACHE = {}
def load_emulator_checkpoint(
    checkpoint_path: str,
    device: Optional[Union[str, torch.device]] = None,
):
    """Load one saved per-line emulator checkpoint and rebuild its network."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    cache_key = (str(checkpoint_path), str(device))
    if cache_key in _CHECKPOINT_CACHE:
        return _CHECKPOINT_CACHE[cache_key]
    
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = state.get("config") or {}
    model = _pick_architecture(config).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()
    _CHECKPOINT_CACHE[cache_key] = (model, state, device)
    return model, state, device


def default_wavelength_grid(
    lambda_min: float,
    lambda_max: float,
    num_points: int = 161,
) -> np.ndarray:
    """Return a uniform wavelength grid in Angstrom."""
    return np.linspace(lambda_min, lambda_max, num_points, dtype=np.float32)


def emulate_line(
    checkpoint_path: str,
    params: Union[Sequence[float], Mapping[str, float]],
    wavelengths: Optional[Iterable[float]] = None,
    device: Optional[Union[str, torch.device]] = None,
    num_points: int = 161,
    param_order: Sequence[str] = DEFAULT_PARAM_ORDER,
) -> Dict[str, np.ndarray]:
    """
    Emulate one spectral line from one checkpoint.

    Parameters
    ----------
    checkpoint_path
        Path to one `emulator_*.pth` file.
    params
        Either a 7-element sequence or a dict with keys:
        `Teff`, `logg`, `R`, `Mdot`, `v_inf`, `Y_He`, `v_turb`.
    wavelengths
        Wavelength grid in Angstrom. If omitted, a default uniform grid is used.
    num_points
        Number of default grid points if `wavelengths` is not provided.

    Returns
    -------
    dict with keys `line_name`, `wavelength`, `flux`
    """
    model, state, device = load_emulator_checkpoint(checkpoint_path, device=device)

    params_norm = normalize_params(
        params=params,
        param_mins=state["param_mins"],
        param_maxs=state["param_maxs"],
        param_order=param_order,
    )

    lambda_min = float(state["lambda_min"])
    lambda_max = float(state["lambda_max"])
    if wavelengths is None:
        wavelengths = default_wavelength_grid(lambda_min, lambda_max, num_points=num_points)

    wavelengths = np.asarray(list(wavelengths), dtype=np.float32)
    wavelengths_norm = (wavelengths - lambda_min) / (lambda_max - lambda_min)

    params_tensor = torch.tensor(params_norm[None, :], dtype=torch.float32, device=device)
    waves_tensor = torch.tensor(wavelengths_norm[None, :], dtype=torch.float32, device=device)

    with torch.no_grad():
        flux = model(params_tensor, waves_tensor).detach().cpu().numpy().ravel()

    if USE_RESIDUAL:
        flux = flux + FLUX_OFFSET

    return {
        "line_name": (state.get("config") or {}).get(
            "line_file",
            os.path.basename(checkpoint_path).removeprefix("emulator_").removesuffix(".pth"),
        ),
        "wavelength": wavelengths,
        "flux": flux.astype(np.float32),
    }


def emulate_hg_spectrum(
    params: Union[Sequence[float], Mapping[str, float]],
    emulator_dir: str = "emulators_per_line_hg",
    wavelengths_by_line: Optional[Mapping[str, Iterable[float]]] = None,
    device: Optional[Union[str, torch.device]] = None,
    num_points: int = 161,
    param_order: Sequence[str] = DEFAULT_PARAM_ORDER,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Emulate all lines found in `emulator_dir`.

    Parameters
    ----------
    params
        Either a 7-element sequence or a dict with keys:
        `Teff`, `logg`, `R`, `Mdot`, `v_inf`, `Y_He`, `v_turb`.
    emulator_dir
        Folder containing `emulator_*.pth` checkpoints.
    wavelengths_by_line
        Optional custom wavelength grids in Angstrom.
        Example:
        `{"OUT.HGAMMA_VTV010": np.linspace(4310.0, 4370.0, 300)}`

        Any line not listed here uses the default uniform grid saved by its
        checkpoint wavelength range.
    num_points
        Number of grid points for lines that use the default grid.

    Returns
    -------
    dict
        A dictionary keyed by line name. Each value contains:
        - `wavelength`: wavelength grid in Angstrom
        - `flux`: predicted normalized flux
    """
    checkpoint_paths = sorted(glob.glob(os.path.join(emulator_dir, "emulator_*.pth")))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No emulator_*.pth files found in {emulator_dir}")

    results: Dict[str, Dict[str, np.ndarray]] = {}
    for checkpoint_path in checkpoint_paths:
        line_name = os.path.basename(checkpoint_path).removeprefix("emulator_").removesuffix(".pth")
        wavelengths = None if wavelengths_by_line is None else wavelengths_by_line.get(line_name)
        results[line_name] = emulate_line(
            checkpoint_path=checkpoint_path,
            params=params,
            wavelengths=wavelengths,
            device=device,
            num_points=num_points,
            param_order=param_order,
        )

    return results


def merge_line_spectra(
    line_results: Mapping[str, Mapping[str, np.ndarray]],
    output_wavelength: Optional[Iterable[float]] = None,
    fill_value: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Merge per-line emulator outputs onto one wavelength grid.

    The per-line spectra are interpolated only inside their own wavelength
    windows. Points outside all line windows are set to `fill_value`.
    Overlapping line windows are averaged.
    """
    if not line_results:
        raise ValueError("line_results is empty.")

    if output_wavelength is None:
        output_wavelength = np.unique(
            np.concatenate(
                [np.asarray(result["wavelength"], dtype=np.float32) for result in line_results.values()]
            )
        )

    wavelength = np.asarray(list(output_wavelength), dtype=np.float32)
    if wavelength.ndim != 1:
        raise ValueError("output_wavelength must be one-dimensional.")

    order = np.argsort(wavelength)
    wavelength_sorted = wavelength[order]

    flux_sum = np.zeros_like(wavelength_sorted, dtype=np.float64)
    count = np.zeros_like(wavelength_sorted, dtype=np.float64)

    for result in line_results.values():
        line_wavelength = np.asarray(result["wavelength"], dtype=np.float32)
        line_flux = np.asarray(result["flux"], dtype=np.float32)
        if line_wavelength.ndim != 1 or line_flux.ndim != 1:
            raise ValueError("Each line result must contain one-dimensional wavelength and flux arrays.")
        if line_wavelength.size != line_flux.size:
            raise ValueError("Line wavelength and flux arrays must have the same length.")
        if line_wavelength.size == 0:
            continue

        line_order = np.argsort(line_wavelength)
        line_wavelength = line_wavelength[line_order]
        line_flux = line_flux[line_order]

        mask = (wavelength_sorted >= line_wavelength[0]) & (wavelength_sorted <= line_wavelength[-1])
        if not np.any(mask):
            continue

        flux_sum[mask] += np.interp(wavelength_sorted[mask], line_wavelength, line_flux)
        count[mask] += 1.0

    flux_sorted = np.full_like(wavelength_sorted, fill_value, dtype=np.float64)
    covered = count > 0
    flux_sorted[covered] = flux_sum[covered] / count[covered]

    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(order.size)

    return {
        "wavelength": wavelength_sorted[inverse_order],
        "flux": flux_sorted[inverse_order].astype(np.float32),
        "coverage": count[inverse_order].astype(np.int16),
    }


def emulate_hg_spectrum_from_indat(
    indat_path_or_model_dir: str,
    emulator_dir: str = "emulators_per_line_hg",
    output_wavelength: Optional[Iterable[float]] = None,
    wavelengths_by_line: Optional[Mapping[str, Iterable[float]]] = None,
    device: Optional[Union[str, torch.device]] = None,
    num_points: int = 161,
    default_v_turb: Optional[float] = None,
    fill_value: float = 1.0,
    return_lines: bool = True,
) -> Dict[str, object]:
    """
    Read parameters from INDAT, run each per-line emulator, and merge the lines.

    Returns a dictionary with:
    - `params`: the physical parameters read from INDAT
    - `spectrum`: merged unified spectrum with `wavelength`, `flux`, `coverage`
    - `lines`: individual per-line results, included when `return_lines=True`
    """
    params = read_indat_params(indat_path_or_model_dir, default_v_turb=default_v_turb)
    line_results = emulate_hg_spectrum(
        params=params,
        emulator_dir=emulator_dir,
        wavelengths_by_line=wavelengths_by_line,
        device=device,
        num_points=num_points,
    )
    spectrum = merge_line_spectra(
        line_results=line_results,
        output_wavelength=output_wavelength,
        fill_value=fill_value,
    )

    result: Dict[str, object] = {
        "params": params,
        "spectrum": spectrum,
    }
    if return_lines:
        result["lines"] = line_results
    return result


if __name__ == "__main__":
    print("Example usage:")
    print("1. Read INDAT, call all per-line emulators, and merge onto a final grid:")
    print("   import numpy as np")
    print("   result = emulate_hg_spectrum_from_indat(")
    print("       'path/to/model/INDAT',")
    print("       emulator_dir='emulators_per_line_hg',")
    print("       output_wavelength=np.linspace(4000.0, 7000.0, 5000),")
    print("   )")
    print("   wavelength = result['spectrum']['wavelength']")
    print("   flux = result['spectrum']['flux']")
    print()
    print("2. If parameters are already known, call the per-line emulators directly:")
    print("   import numpy as np")
    print("   params = {")
    print("       'Teff': 35000.0, 'logg': 3.6, 'R': 12.0, 'Mdot': 1e-6,")
    print("       'v_inf': 2000.0, 'Y_He': 0.10, 'v_turb': 10.0,")
    print("   }")
    print("   results = emulate_hg_spectrum(")
    print("       params,")
    print("       emulator_dir='emulators_per_line_hg',")
    print("       wavelengths_by_line={")
    print("           'OUT.HGAMMA_VTV010': np.linspace(4310.0, 4370.0, 300),")
    print("           'OUT.HEI4471_VTV010': np.linspace(4455.0, 4485.0, 250),")
    print("       },")
    print("   )")
