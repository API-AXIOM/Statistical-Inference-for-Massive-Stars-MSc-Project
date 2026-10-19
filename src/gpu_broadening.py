"""Rotational, macroturbulent and instrumental broadening for a batch of models.

Broadening a spectrum means convolving it with three kernels: a Gray rotation
profile set by ``vsini``, a radial-tangential macroturbulence profile set by
``vmacro``, and a Gaussian instrumental profile set by the resolving power.

Doing that directly is awkward to batch, because ``vsini`` and ``vmacro``
differ from one parameter vector to the next and control how wide the kernels
are, so every member of a batch would need a differently sized convolution.
This module avoids that by working in the Fourier domain.

The quantity computed for each line is

    out[j] = sum_k  kernel[k] * flux_ext[j + c - k]

where ``kernel`` is the full convolution ``instrument * rotation * macro``,
``c = (len(kernel) - 1) // 2`` is its centre, and ``flux_ext`` is the model
flux extended with 1.0 beyond its own range (the spectrum is continuum
normalised, so 1.0 is the correct value outside it).

Because the same kernel is applied at every pixel, the convolution can be
done with FFTs. Each kernel is stored *zero-delay*: the weight for offset 0
sits at index 0, positive offsets follow it, and negative offsets wrap round
to the end of the array. Stored this way, kernels can be combined simply by
multiplying their Fourier transforms, and the result is not shifted.

A zero-delay kernel always has the same length, the transform length, no
matter how wide it is; a narrow kernel just contains more zeros. So a whole
batch of models with different vsini and vmacro can be broadened in a single
FFT call. The transform length is fixed once, from the largest vsini and vmacro
allowed, and every kernel up to that width fits inside it.

Everything independent of the sampled parameters is computed once in
``LineBroadeningPlan``: the instrumental profile and its transform, the
transform length, and the sample-offset grid.
"""

import math
import numpy as np
import torch
from scipy.stats import norm
from gpu_config import C_KMS, INSTRUMENT_NSIG, LIMB_DARKENING, MACRO_EXTENT


FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))


def grid_regularity(axis):
    """How far an axis departs from perfect regularity, in units of one step.

    Returns 0 for an exactly evenly spaced axis. Values well below 1 mean the
    departure is a small fraction of a pixel, which is typical of a grid
    stored in single precision.
    """
    axis = np.asarray(axis, dtype=np.float64)
    if axis.size < 3:
        return 0.0
    step = (axis[-1] - axis[0]) / (axis.size - 1)
    if step == 0:
        return np.inf
    predicted = axis[0] + step * np.arange(axis.size)
    return float(np.max(np.abs(axis - predicted)) / abs(step))


def sampling_mode_and_axis(wavelength):
    """Decide whether a grid is evenly spaced in wavelength or in velocity.

    Spectra are commonly stored on one of two grids: constant ``d(lambda)``,
    or constant ``d(lambda)/lambda``, which is a constant step in velocity and
    equivalently in ``ln(lambda)``. Rotation, macroturbulence and a fixed
    resolving power are all velocity broadenings, so the second is the more
    natural grid for them.

    Returns ``(mode, axis, irregularity)`` where ``mode`` is ``"linear"`` or
    ``"log"``, ``axis`` is the evenly spaced coordinate, and ``irregularity``
    is that axis's departure from regularity in units of one step.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    linear = grid_regularity(wavelength)
    logarithmic = (grid_regularity(np.log(wavelength))
                   if np.all(wavelength > 0) else np.inf)

    if linear <= logarithmic:
        return "linear", wavelength, linear
    return "log", np.log(wavelength), logarithmic


def detect_sampling(wavelength, tolerance=0.25):
    """Decide how a grid is spaced, and refuse grids that are neither kind.

    Returns ``(mode, axis, irregularity)`` exactly as
    ``sampling_mode_and_axis`` does. Raises ``ValueError`` if the grid is
    evenly spaced neither in wavelength nor in velocity, that is, if even
    the better of the two fits departs from regularity by more than
    ``tolerance`` pixels.
    """
    mode, axis, irregularity = sampling_mode_and_axis(wavelength)

    if irregularity > tolerance:
        wavelength = np.asarray(wavelength, dtype=np.float64)
        linear = grid_regularity(wavelength)
        logarithmic = (grid_regularity(np.log(wavelength))
                       if np.all(wavelength > 0) else np.inf)
        raise ValueError(
            f"the wavelength grid is neither evenly spaced in wavelength "
            f"(irregularity {linear:.2f} pixels) nor in velocity "
            f"({logarithmic:.2f} pixels). Split it into pieces that are each "
            f"evenly spaced.")
    return mode, axis, irregularity


def instrumental_profile(wave, resolution, nsig=INSTRUMENT_NSIG):
    """Gaussian instrumental profile on an evenly spaced wavelength grid.

    The width is ``lambda / resolution`` (FWHM) at the mean wavelength, and
    the profile is sampled at the grid's average step.
    """
    wave = np.asarray(wave, dtype=np.float64)
    mean_wavelength = np.mean(wave)
    sigma = mean_wavelength / float(resolution) * FWHM_TO_SIGMA
    step = (wave[-1] - wave[0]) / (wave.size - 1)
    return _gaussian_kernel(sigma, step, nsig)


def instrumental_profile_velocity(velocity_step, resolution,
                                  nsig=INSTRUMENT_NSIG):
    """Gaussian instrumental profile on an evenly spaced velocity grid.

    At a fixed resolving power the instrumental width is a constant fraction
    of the wavelength, so in velocity it is the same Gaussian everywhere.
    """
    sigma = C_KMS / float(resolution) * FWHM_TO_SIGMA
    return _gaussian_kernel(sigma, velocity_step, nsig)


def _gaussian_kernel(sigma, step, nsig):
    """Normalised Gaussian sampled on a grid of the given step."""
    x = np.arange(-nsig * sigma, nsig * sigma + step, step)
    profile = norm.pdf(x, scale=sigma, loc=x[len(x) // 2 + len(x) % 2 - 1])
    return profile / np.sum(profile)


def _instrument_centre_index(length):
    """Index of the profile peak, which sets the kernel's delay."""
    return length // 2 + length % 2 - 1


def _half_width_samples(velocity_max, velocity_step):
    """Number of positive kernel samples out to ``velocity_max``."""
    return max(1, int(math.ceil(velocity_max / velocity_step)))


class LineBroadeningPlan:
    """Fixed-cost broadening machinery for one line on one wavelength grid.

    Build one plan per line window, then call ``broaden`` with a batch of
    model spectra and their ``vsini`` and ``vmacro``.

    Parameters
    ----------
    wavelength : array (P,)
        Wavelengths of the line window, in Angstrom, evenly spaced either in
        wavelength or in ``ln(wavelength)``.
    resolution : float
        Resolving power ``R = lambda / FWHM`` of the instrument.
    vsini_max, vmacro_max : float
        Largest ``vsini`` and ``vmacro`` (km/s) that will be passed to
        ``broaden``. They fix the padding and the transform length.
    device : torch.device or str
        Device the kernels and transforms live on.
    dtype : torch.dtype
        Floating-point type used for the transforms.
    nsig : float
        Half-width of the instrumental Gaussian, in standard deviations.
    limb_darkening : float
        Linear limb-darkening coefficient ``epsilon`` of the rotation profile.
    mode : {"linear", "log"} or None
        How the grid is spaced. ``None`` detects it with ``detect_sampling``.
    macro_extent : float
        The macroturbulence profile is cut at ``macro_extent * vmacro``.
    """

    def __init__(self, wavelength, resolution, vsini_max, vmacro_max,
                 device, dtype=torch.float32, nsig=INSTRUMENT_NSIG,
                 limb_darkening=LIMB_DARKENING, mode=None,
                 macro_extent=MACRO_EXTENT):
        wavelength = np.asarray(wavelength, dtype=np.float64)
        if wavelength.size < 2:
            raise ValueError("a line window needs at least two wavelength points")

        self.device = device
        self.dtype = dtype
        self.limb_darkening = float(limb_darkening)
        self.macro_extent = float(macro_extent)
        self.n_points = int(wavelength.size)

        if mode is None:
            mode, _, irregularity = detect_sampling(wavelength)
        else:
            axis = wavelength if mode == "linear" else np.log(wavelength)
            irregularity = grid_regularity(axis)
        self.mode = mode
        self.irregularity = irregularity

        # All three kernels are velocity broadenings, so build them on a grid
        # of velocity offsets. On a constant-velocity grid the step is exact
        # and the same everywhere; on a constant-wavelength grid it is fixed
        # by a reference wavelength, which is what makes that grid an
        # approximation for a wide line window.
        if mode == "log":
            step = float((np.log(wavelength[-1]) - np.log(wavelength[0]))
                         / (self.n_points - 1))
            self.velocity_step_rotation = C_KMS * step
            self.velocity_step_macro = C_KMS * step
            self.velocity_step_instrument = C_KMS * step
            instrument = instrumental_profile_velocity(
                self.velocity_step_instrument, resolution, nsig=nsig)
        else:
            wdiff = float((wavelength[-1]-wavelength[0])/(self.n_points-1))
            w0 = float(np.mean(wavelength))
            velocity_step = C_KMS * wdiff / w0
            self.velocity_step_rotation = velocity_step
            self.velocity_step_macro = velocity_step
            self.velocity_step_instrument = velocity_step

            instrument = instrumental_profile(wavelength, resolution, nsig=nsig)

        n_instrument = instrument.size
        centre_instrument = _instrument_centre_index(n_instrument)

        # Widest kernels the prior can ask for, so one transform length serves
        # every step of the fit. The rotation profile ends at vsini; the
        # macroturbulence profile is cut at macro_extent * vmacro.
        self.k_rotation = _half_width_samples(vsini_max, self.velocity_step_rotation)
        self.k_macro = _half_width_samples(self.macro_extent * vmacro_max,
                                           self.velocity_step_macro)

        kernel_length = n_instrument + 2 * self.k_rotation + 2 * self.k_macro
        centre = (kernel_length - 1) // 2
        assert centre == centre_instrument + self.k_rotation + self.k_macro

        # Ones padding around the model flux.
        self.pad_left = kernel_length - 1 - centre
        self.pad_right = centre
        self.extended_length = self.n_points + kernel_length - 1
        self.output_offset = self.pad_left

        self.n_fft = int(next_fast_length(self.extended_length))

        # Wrapped integer sample offsets: 0, 1, ..., n//2, then the negative
        # offsets up to -1 (e.g. n=8: 0..4, -3..-1; n=9: 0..4, -4..-1).
        offsets = np.arange(self.n_fft)
        offsets[offsets > self.n_fft // 2] -= self.n_fft
        self.offsets = torch.as_tensor(offsets, dtype=dtype, device=device)
        absolute = self.offsets.abs()
        self.abs_velocity_rotation = absolute * self.velocity_step_rotation
        self.abs_velocity_macro = absolute * self.velocity_step_macro

        # The instrumental kernel does not depend on the sampled parameters,
        # so transform it once.
        instrument_zero_delay = np.zeros(self.n_fft, dtype=np.float64)
        indices = (np.arange(n_instrument) - centre_instrument) % self.n_fft
        np.add.at(instrument_zero_delay, indices, instrument)
        self.instrument_spectrum = torch.fft.rfft(
            torch.as_tensor(instrument_zero_delay, dtype=dtype, device=device)
        )

        self.instrument_profile = instrument

    # -- parameter dependent kernels ---------------------------------------
    def rotation_kernel(self, vsini):
        """(N, n_fft) zero-delay Gray rotation profiles for a batch of vsini."""
        limit = vsini.clamp_min(1e-12).unsqueeze(-1)
        epsilon = self.limb_darkening
        norm_factor = 1.0 - epsilon / 3.0
        c1 = 2.0 * (1.0 - epsilon) / (math.pi * limit * norm_factor)
        c2 = epsilon / (2.0 * limit * norm_factor)

        velocity = self.abs_velocity_rotation.unsqueeze(0)
        inside = velocity < limit
        x_squared = (velocity / limit) ** 2
        one_minus = (1.0 - x_squared).clamp_min(0.0)
        profile = c1 * torch.sqrt(one_minus) + c2 * one_minus
        profile = torch.where(inside, profile, torch.zeros_like(profile))
        return profile / profile.sum(dim=-1, keepdim=True)

    def macro_kernel(self, vmacro):
        """(N, n_fft) zero-delay radial-tangential macroturbulence profiles.

        Assumes equal radial and tangential components with the same
        characteristic velocity ``vmacro``. The profile has wings at every
        velocity, so it is cut at ``macro_extent * vmacro`` and renormalised.
        """
        limit = vmacro.clamp_min(1e-12).unsqueeze(-1)
        ccr = 2.0 / (math.sqrt(math.pi) * limit)

        velocity = self.abs_velocity_macro.unsqueeze(0)
        x = velocity / limit
        inside = x < self.macro_extent
        profile = ccr * (torch.exp(-x * x)
                         + math.sqrt(math.pi) * x * (torch.erf(x) - 1.0))
        profile = torch.where(inside, profile, torch.zeros_like(profile))
        return profile / profile.sum(dim=-1, keepdim=True)

    # -- the convolution itself --------------------------------------------
    def broaden(self, flux, vsini, vmacro):
        """Broaden a batch of model spectra.

        flux   : (N, P) continuum-normalised model flux on this line's grid
        vsini  : (N,) km/s
        vmacro : (N,) km/s
        returns: (N, P)
        """
        n_batch = flux.shape[0]
        extended = flux.new_ones((n_batch, self.n_fft))
        extended[:, self.pad_left:self.pad_left + self.n_points] = flux
        extended[:, self.extended_length:] = 0.0

        spectrum = torch.fft.rfft(extended, n=self.n_fft)
        spectrum = spectrum * self.instrument_spectrum.unsqueeze(0)
        spectrum = spectrum * torch.fft.rfft(self.rotation_kernel(vsini), n=self.n_fft)
        spectrum = spectrum * torch.fft.rfft(self.macro_kernel(vmacro), n=self.n_fft)

        convolved = torch.fft.irfft(spectrum, n=self.n_fft)
        start = self.output_offset
        return convolved[:, start:start + self.n_points]


def next_fast_length(n):
    """Smallest 5-smooth integer >= n (a size cuFFT handles efficiently)."""
    if n <= 1:
        return 1
    best = None
    power5 = 1
    while power5 < 2 * n:
        power3 = power5
        while power3 < 2 * n:
            value = power3
            while value < n:
                value *= 2
            if best is None or value < best:
                best = value
            power3 *= 3
        power5 *= 5
    return best
