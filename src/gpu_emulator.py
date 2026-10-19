"""Batched GPU wrappers around the FASTWIND DeepONet emulators.

A DeepONet predicts ``sum_l trunk(lambda)[l] * branch(theta)[l]``: the trunk
depends only on wavelength, the branch only on the stellar parameters.
During a fit the wavelength grid is fixed, so the trunk is evaluated once
when a wrapper is built and stored as a basis matrix of shape
``(latent_dim, n_points)``. Predicting a batch of spectra is then one small
branch network and one matrix multiply:

    line emulators:  flux       = branch(theta) @ basis + flux_offset
    SED emulator:    log10 F_l  = (branch(theta) @ basis) * bin_std + bin_mean

The parameter vector has log10 of the mass-loss rate at index 3, as in
``gpu_config.PARAMETER_BOUNDS``; the wrappers normalise from that directly.
"""

import copy

import numpy as np
import torch


def _device_copy(model, device, dtype):
    """A detached copy of a DeepONet on `device`, leaving the input untouched."""
    copied = copy.deepcopy(model).to(device=device, dtype=dtype).eval()
    for parameter in copied.parameters():
        parameter.requires_grad_(False)
    return copied


class BatchedLineEmulator:
    """One line emulator, specialised to a fixed wavelength grid.

    The emulator is a DeepONet: two small neural networks whose outputs are
    combined to give the flux. The *branch* network turns the 13 stellar
    parameters into a list of ``latent_dim`` numbers; the *trunk* network
    turns a wavelength into a list of the same length. The flux at that
    wavelength is the dot product of the two lists (multiply element by
    element and add up), plus a constant offset.

    The trunk only sees wavelengths, and the observed wavelength grid does not
    change during a fit, so it is run once here and its output kept as the
    ``basis`` matrix. Each prediction then needs only the branch network and
    one matrix multiplication, for a whole batch of parameter vectors at once.
    """

    def __init__(self, emulator, wavelength, device, dtype=torch.float32):
        self.name = emulator.line
        self.device = device
        self.dtype = dtype
        self.n_points = int(np.size(wavelength))

        # The network was trained to predict the flux minus a constant
        # (usually 1, the continuum level); the constant is added back at the
        # end. It is put on the GPU once here, because copying it from the
        # CPU on every call would be slow and would stop the likelihood from
        # being recorded as a CUDA graph.
        self.flux_offset = float(emulator._flux_offset)
        self.flux_offset_tensor = torch.tensor(
            self.flux_offset, dtype=dtype, device=device)

        # A copy of the network on the GPU, in the chosen precision. A copy,
        # because moving a model changes it in place and the original emulator
        # should stay usable. Only the branch is kept; the trunk is used once
        # below to build the basis and then discarded with `model`.
        model = _device_copy(emulator.model, device, dtype)
        self.branch = model.branch

        # The trunk was trained on wavelengths rescaled to run from 0 to 1
        # across the line window, so the observed wavelengths are rescaled the
        # same way (e.g. for H beta, 4770 A -> 0 and 4965 A -> 1).
        lambda_min, lambda_max = emulator.lambda_range
        coordinates = (np.asarray(wavelength, np.float64) - lambda_min) \
            / (lambda_max - lambda_min)
        # The networks work on batches, so add a leading batch dimension,
        # giving shape (1, P).
        coordinate_tensor = torch.as_tensor(
            coordinates, dtype=dtype, device=device).unsqueeze(0)

        # Run the trunk once for all P wavelengths. `no_grad` skips the
        # bookkeeping PyTorch keeps for training, which is not needed here.
        with torch.no_grad():
            basis = model.trunk(coordinate_tensor)[0]      # (P, latent_dim)
        # Store it as (latent_dim, P), so a prediction is `latent @ basis`;
        # `contiguous` keeps it in one block of memory for a faster multiply.
        self.basis = basis.t().contiguous()
        self.latent_dim = int(self.basis.shape[0])

        # Before entering the network, each parameter is rescaled to roughly
        # 0..1 with the minimum and range of the training set. These are
        # stored for log10(mdot) (index 3), which is what the fit samples.
        self.parameter_min = torch.as_tensor(
            emulator._p_min, dtype=torch.float64, device=device)
        self.parameter_range = torch.as_tensor(
            emulator._p_rng, dtype=torch.float64, device=device)

    def normalise(self, theta_log_mdot):
        """(N, 13) sampled parameters (log10 mdot) -> network inputs.

        The rescaling is done in float64 and only the result is converted to
        the network's precision.
        """
        normalised = (theta_log_mdot - self.parameter_min) / self.parameter_range
        return normalised.to(self.dtype)

    def predict(self, theta_log_mdot):
        """(N, 13) -> (N, n_points) continuum-normalised flux, unbroadened."""
        # Branch network on the whole batch: (N, 13) -> (N, latent_dim).
        latent = self.branch(self.normalise(theta_log_mdot))
        # addmm(a, B, C) = a + B @ C in one GPU operation:
        # (N, latent_dim) @ (latent_dim, P) gives the dot product of the two
        # lists for every model and wavelength, then the offset is added.
        return torch.addmm(self.flux_offset_tensor, latent, self.basis)


class BatchedSedEmulator:
    """FLUXCONT continuum emulator, specialised to a fixed wavelength grid.

    Built the same way as ``BatchedLineEmulator`` (see there for how the two
    networks combine), with two differences. It predicts log10 F_lambda, the
    luminosity-like continuum flux in erg/s/A (see ``gpu_photometry``), over a
    wide wavelength range, so its wavelength coordinate is log10(lambda).
    And its output is standardised (see ``predict_log_flux``) instead of
    offset by a constant.
    """

    def __init__(self, emulator, wavelength, device, dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.n_points = int(np.size(wavelength))

        # Copy of the network on the GPU; only the branch is kept (see
        # BatchedLineEmulator).
        model = _device_copy(emulator.model, device, dtype)
        self.branch = model.branch

        # Rescale log10(lambda) to 0..1 over the emulator's wavelength range,
        # as in training, and run the trunk once to build the basis.
        wavelength = np.asarray(wavelength, np.float64)
        coordinates = (np.log10(wavelength) - emulator._llog_min) \
            / (emulator._llog_max - emulator._llog_min)
        coordinate_tensor = torch.as_tensor(
            coordinates, dtype=dtype, device=device).unsqueeze(0)
        with torch.no_grad():
            basis = model.trunk(coordinate_tensor)[0]      # (P, latent_dim)
        self.basis = basis.t().contiguous()                 # (latent_dim, P)

        # log10 F_lambda spans many orders of magnitude across the spectrum.
        # To make it easier to learn, the training split the wavelength range
        # into bins and, in each bin, subtracted the mean of log10 F_lambda
        # over the training spectra and divided by its standard deviation.
        # The network therefore predicts numbers of order 1; here we look up
        # which bin each observed wavelength falls in, so that the mean and
        # standard deviation can be put back after every prediction.
        bins = np.clip(np.digitize(coordinates, emulator._bin_edges) - 1,
                       0, emulator._bin_means.size - 1)
        self.bin_mean = torch.as_tensor(
            emulator._bin_means[bins], dtype=torch.float64, device=device)
        self.bin_std = torch.as_tensor(
            emulator._bin_stds[bins], dtype=torch.float64, device=device)

        # Rescaling of the parameters to 0..1, as for the line emulators;
        # stored for log10(mdot).
        self.parameter_min = torch.as_tensor(
            emulator._p_min, dtype=torch.float64, device=device)
        self.parameter_range = torch.as_tensor(
            emulator._p_rng, dtype=torch.float64, device=device)

    def predict_log_flux(self, theta_log_mdot):
        """(N, 13) -> (N, n_points) log10 F_lambda, in float64."""
        normalised = ((theta_log_mdot - self.parameter_min)
                      / self.parameter_range).to(self.dtype)
        # Branch network, then the matrix multiply with the basis: the
        # standardised log flux for every model and wavelength.
        latent = self.branch(normalised)
        raw = (latent @ self.basis).to(torch.float64)
        # Undo the standardisation: multiply by the bin's standard deviation
        # and add its mean. Done in float64, since the result feeds 10**x.
        return raw * self.bin_std + self.bin_mean
