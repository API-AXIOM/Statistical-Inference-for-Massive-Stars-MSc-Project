"""Vectorised log-posterior for the 15-parameter FASTWIND fit, on one GPU.

``GPUPosterior`` evaluates a whole batch of parameter vectors at once: the
emulators become batched matrix multiplies and the broadening becomes one
batched Fourier-domain product per line window. Call it with an array of shape
``(n, 15)`` and it returns ``(n,)`` log-posterior values.

The statistical model is:

* a uniform prior over the box in ``gpu_config.PARAMETER_BOUNDS``, or an
  informative prior from ``priors.py``;
* a Gaussian spectral term with sigma = 1 / SNR, or a per-pixel uncertainty,
  one residual per covered observed wavelength;
* a Gaussian term on a broadband magnitude, with a configurable uncertainty.

Each observed wavelength belongs to at most one line window: the windows in
``gpu_config.LINE_WINDOW_OVERRIDES`` do not overlap, and each is taken as
``low <= wavelength < high`` so neighbouring windows can share an edge. Every
covered wavelength therefore contributes exactly one residual. Building a
posterior from windows that do overlap raises an error.

The class is deliberately independent of any particular sampler. Alongside
``__call__`` it offers ``log_likelihood`` and ``prior_transform``, which is
the interface nested samplers expect.
"""

import numpy as np
import torch

from gpu_broadening import LineBroadeningPlan, sampling_mode_and_axis
from gpu_config import (BOUNDS_ARRAY, LMC_DISTANCE_PC, MDOT_INDEX, N_DIM,
                        N_THETA, PARAMETER_BOUNDS, VMACRO_INDEX, VSINI_INDEX,
                        line_window, resolve_resolution)
from gpu_emulator import BatchedLineEmulator
from gpu_photometry import PhotometryPlan


class _Line:
    """One line window and the observed points inside it.

    name        : line name, e.g. "HEII4541"
    indices     : positions of the window's points in the observed spectrum
    wavelength  : the observed wavelengths at those positions
    emulator    : the line's ``BatchedLineEmulator`` on this grid
    broadening  : the ``LineBroadeningPlan`` for this grid
    columns     : the columns of the window's points in the covered-points
                  array, i.e. where its model values go in ``merged_model``
    """

    __slots__ = ("name", "indices", "wavelength", "emulator", "broadening",
                 "columns")


def check_window(wavelength, low, high, min_points=8, tolerance=0.25):
    """Decide whether a line window can be fitted, and how it is sampled.

    ``wavelength`` holds the observed points inside the window
    (``low <= wavelength < high``). A line is fitted only if these points
    cover its whole window and are evenly spaced, either in wavelength or in
    velocity (log wavelength), because the broadening needs evenly spaced
    flux and broadening only part of a line would distort its profile.
    ``tolerance`` is in units of one step, as in
    ``gpu_broadening.detect_sampling``.

    Returns ``(problem, mode)``: ``problem`` is None if the window can be
    fitted and otherwise says why not; ``mode`` is "linear" or "log", for
    ``LineBroadeningPlan``.
    """
    wavelength = np.asarray(wavelength, dtype=np.float64)
    if wavelength.size < min_points:
        return f"fewer than {min_points} observed points in the window", None
    mode, _, irregularity = sampling_mode_and_axis(wavelength)
    if irregularity > tolerance:
        return "not evenly sampled (a gap or a change of sampling)", mode
    first_step = wavelength[1] - wavelength[0]
    last_step = wavelength[-1] - wavelength[-2]
    if (wavelength[0] - low > (1 + tolerance) * first_step
            or high - wavelength[-1] > (1 + tolerance) * last_step):
        return "only partly covered by the observed spectrum", mode
    return None, mode


class GPUPosterior:
    """Batched log-posterior. Call it with (N, 15) and get (N,) back."""

    def __init__(self, observed_wavelength, observed_flux, observed_magnitude,
                 spectrum_snr, resolution, line_emulators, sed_emulator=None,
                 filter_name="2MASS_Ks", zp_system="vega", distance_pc=LMC_DISTANCE_PC,
                 magnitude_uncertainty=0.1, device=None, dtype=torch.float32,
                 use_magnitude=True, filter_directory=None, min_line_points=8,
                 flux_uncertainty=None, prior=None, extinction=0.0,
                 verbose=True):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.dtype = dtype
        self.resolution = resolution if isinstance(resolution, list) else float(resolution)
        self.spectrum_snr = float(spectrum_snr)
        self.spectrum_sigma = 1.0 / float(spectrum_snr)
        self.magnitude_uncertainty = float(magnitude_uncertainty)
        self.observed_magnitude = observed_magnitude
        self.use_magnitude = bool(use_magnitude)
        # None means the uniform prior over PARAMETER_BOUNDS, which `forward`
        # already applies as a box mask. An explicit prior is evaluated on the
        # host and added afterwards, so the GPU part stays graph-capturable.
        self.prior = prior

        observed_wavelength = np.asarray(observed_wavelength, dtype=np.float64)
        observed_flux = np.asarray(observed_flux, dtype=np.float64)
        self.observed_wavelength = observed_wavelength
        self.observed_flux = observed_flux

        vsini_max = PARAMETER_BOUNDS["vsini"][1]
        vmacro_max = PARAMETER_BOUNDS["vmacro"][1]

        self.lines = []
        skipped = {}
        for name, emulator in line_emulators.items():
            # Positions of the observed points inside this line's window.
            low, high = line_window(name, emulator)
            window = np.flatnonzero((observed_wavelength >= low)
                                    & (observed_wavelength < high))
            # Leave the line out unless its whole window is evenly covered.
            # Whether that is evenly in wavelength or in velocity is decided
            # per window, so a spectrum stitched from differently sampled
            # parts still works.
            problem, mode = check_window(observed_wavelength[window], low,
                                         high, min_line_points)
            if problem:
                skipped[name] = problem
                continue

            line = _Line()
            line.name = name
            line.indices = window
            line.wavelength = observed_wavelength[window]
            # With several instruments the resolving power depends on
            # wavelength; pick the one that applies to this window.
            line_resolution = resolve_resolution(
                resolution, line.wavelength, line.name)
            # Both are built once for this window's wavelengths: the
            # emulator's wavelength basis and the broadening's FFT set-up
            # depend only on the grid, not on the stellar parameters.
            line.emulator = BatchedLineEmulator(
                emulator, line.wavelength, self.device, dtype)
            line.broadening = LineBroadeningPlan(
                line.wavelength, line_resolution, vsini_max, vmacro_max,
                self.device, dtype, mode=mode)
            self.lines.append(line)

        if not self.lines:
            raise ValueError(
                "no line window is fully and evenly covered by the observed "
                "spectrum: " + "; ".join(f"{name}: {reason}"
                                         for name, reason in skipped.items()))
        self.skipped_lines = skipped

        # The observed points the model covers, each in exactly one window.
        stacked = np.concatenate([line.indices for line in self.lines])
        covered, inverse, counts = np.unique(
            stacked, return_inverse=True, return_counts=True)
        if (counts > 1).any():
            shared = set(covered[counts > 1].tolist())
            pairs = sorted({line.name for line in self.lines
                            if shared.intersection(line.indices.tolist())})
            raise ValueError(
                f"line windows overlap: {int((counts > 1).sum())} observed "
                f"point(s) fall in more than one of {pairs}. Edit "
                f"LINE_WINDOW_OVERRIDES in gpu_config.py so the windows "
                f"meet without overlapping.")
        self.covered_indices = covered
        self.n_covered_points = int(covered.size)

        # For each window, the columns its points occupy in the covered-points
        # array (np.unique's `inverse` maps every stacked point to its column).
        offset = 0
        for line in self.lines:
            size = line.indices.size
            line.columns = torch.as_tensor(
                inverse[offset:offset + size], dtype=torch.long,
                device=self.device)
            offset += size

        self._graphs = {}
        self._likelihood_graphs = {}
        self.covered_observed_flux = torch.as_tensor(
            observed_flux[covered], dtype=dtype, device=self.device)

        # Flux uncertainty: either one number for the whole spectrum, or one
        # per observed wavelength. Per-pixel values weight each measurement by
        # its own inverse variance, which is what the data deserve when the
        # noise varies across the spectrum.
        self.per_pixel_uncertainty = flux_uncertainty is not None
        # Kept on the full observed grid so a run can store exactly what it
        # was given, and a rebuilt posterior can be handed the same array.
        self.flux_uncertainty = (None if flux_uncertainty is None else
                                 np.asarray(flux_uncertainty, dtype=np.float64))
        if self.per_pixel_uncertainty:
            uncertainty = np.asarray(flux_uncertainty, dtype=np.float64)
            if uncertainty.shape != observed_flux.shape:
                raise ValueError(
                    "flux_uncertainty must have one value per observed point")
            uncertainty = uncertainty[covered]
            usable = np.isfinite(uncertainty) & (uncertainty > 0)
            if not usable.any():
                raise ValueError(
                    "every flux uncertainty is zero or non-finite; drop "
                    "--use-errors, or supply a usable error array")
            self.n_unusable_uncertainties = int((~usable).sum())
            # Points without a usable uncertainty are excluded by giving them
            # infinite variance, i.e. zero weight.
            weight = np.zeros_like(uncertainty)
            weight[usable] = 1.0 / uncertainty[usable] ** 2
            self.inverse_variance = torch.as_tensor(
                weight, dtype=torch.float64, device=self.device)
            self.median_uncertainty = float(np.median(uncertainty[usable]))
            self.n_weighted_points = int(usable.sum())
        else:
            self.inverse_variance = None
            self.n_unusable_uncertainties = 0
            self.median_uncertainty = self.spectrum_sigma
            self.n_weighted_points = int(covered.size)

        self.photometry = None
        if self.use_magnitude:
            if sed_emulator is None:
                raise ValueError("use_magnitude=True needs the SED emulator")
            self.photometry = PhotometryPlan(
                sed_emulator, filter_name, zp_system, distance_pc,
                self.device, dtype, filter_directory, extinction)

        self.lower_bound = torch.as_tensor(
            BOUNDS_ARRAY[:, 0], dtype=torch.float64, device=self.device)
        self.upper_bound = torch.as_tensor(
            BOUNDS_ARRAY[:, 1], dtype=torch.float64, device=self.device)

        if verbose:
            self.describe()

    def set_observed_flux(self, observed_flux, observed_magnitude=None):
        """Swap in a different observation without rebuilding the GPU state.

        The cached trunk bases, FFT plans and instrument profiles depend only
        on the wavelength grid, so a mock observation generated from this very
        model can be installed in place instead of paying the setup cost twice.
        """
        observed_flux = np.asarray(observed_flux, dtype=np.float64)
        if observed_flux.shape != self.observed_flux.shape:
            raise ValueError("the new flux must lie on the same wavelength grid")
        self.observed_flux = observed_flux
        self.covered_observed_flux.copy_(torch.as_tensor(
            observed_flux[self.covered_indices], dtype=self.dtype,
            device=self.device))
        if observed_magnitude is not None:
            self.observed_magnitude = observed_magnitude

    # -- introspection ---------------------------------------------------
    def describe(self):
        fft_lengths = [line.broadening.n_fft for line in self.lines]
        points = [line.indices.size for line in self.lines]
        print(f"[GPUPosterior] device={self.device} dtype={self.dtype}")
        print(f"[GPUPosterior] {len(self.lines)} lines fitted "
              f"({min(points)}-{max(points)} points per line window)")
        print(f"[GPUPosterior] FFT lengths {min(fft_lengths)}-{max(fft_lengths)}")
        for mode, described in (("linear", "wavelength"), ("log", "velocity")):
            plans = [line.broadening for line in self.lines
                     if line.broadening.mode == mode]
            if plans:
                steps = [plan.velocity_step_rotation for plan in plans]
                worst = max(plan.irregularity for plan in plans)
                print(f"[GPUPosterior] {len(plans)} windows evenly spaced in "
                      f"{described} ({min(steps):.3f}-{max(steps):.3f} km/s "
                      f"per pixel, regular to {worst:.2f} of a pixel)")
        reached = {name: reason for name, reason in self.skipped_lines.items()
                   if not reason.startswith("fewer")}
        for name, reason in sorted(reached.items()):
            print(f"[GPUPosterior] left out {name}: {reason}")
        missed = sorted(set(self.skipped_lines) - set(reached))
        if missed:
            print(f"[GPUPosterior] not covered by the data: {', '.join(missed)}")
        print(f"[GPUPosterior] magnitude term: "
              f"{'on' if self.use_magnitude else 'off'}")
        print(f"[GPUPosterior] prior: "
              f"{'uniform over the parameter bounds' if self.prior is None else self.prior.name}")
        if self.photometry is not None and self.photometry.extinction:
            print(f"[GPUPosterior] extinction in {self.photometry.filter_name}: "
                  f"{self.photometry.extinction:.3f} mag")
        if self.per_pixel_uncertainty:
            print(f"[GPUPosterior] flux uncertainty: per pixel, median "
                  f"{self.median_uncertainty:.5f} (SNR "
                  f"{1 / self.median_uncertainty:.0f})")
            if self.n_unusable_uncertainties:
                print(f"[GPUPosterior] {self.n_unusable_uncertainties} point(s) "
                      f"had a zero or non-finite uncertainty and carry no "
                      f"weight")
        else:
            print(f"[GPUPosterior] flux uncertainty: {self.spectrum_sigma:.5f} "
                  f"everywhere (SNR {1 / self.spectrum_sigma:.0f})")
        print(f"[GPUPosterior] {self.n_covered_points} observed points enter "
              f"the likelihood, one residual each")
        cached = sum(line.emulator.basis.numel() for line in self.lines)
        if self.photometry is not None:
            cached += self.photometry.emulator.basis.numel()
        print(f"[GPUPosterior] cached trunk bases: "
              f"{cached * self.lines[0].emulator.basis.element_size() / 2**20:.1f} MiB")

    # -- the model -------------------------------------------------------
    def merged_model(self, parameters):
        """(N, 15) -> (N, n_covered) model flux, one value per covered point.

        Each line window's broadened prediction is written into the columns
        of the points it covers (``line.columns``). This is the model compared
        with the observed spectrum, and the one used for plotting.
        """
        theta = parameters[:, :N_THETA]
        vsini = parameters[:, VSINI_INDEX].to(self.dtype)
        vmacro = parameters[:, VMACRO_INDEX].to(self.dtype)

        total = torch.empty(parameters.shape[0], self.n_covered_points,
                            dtype=self.dtype, device=self.device)
        for line in self.lines:
            flux = line.emulator.predict(theta)
            broadened = line.broadening.broaden(flux, vsini, vmacro)
            total.index_copy_(1, line.columns, broadened)
        return total

    def _weighted_squared_residual(self, parameters):
        """(N, 15) device tensor -> (N, n_covered) per-point chi-squared.

        One term per covered observed point, before summing across points --
        summing all of them is ``_chi_squared_spectrum``; summing just the
        points belonging to one line window is ``chi_squared_per_line``.
        """
        residual = (self.covered_observed_flux.unsqueeze(0)
                    - self.merged_model(parameters)).to(torch.float64)
        if self.inverse_variance is None:
            return residual.pow(2) / self.spectrum_sigma ** 2
        return residual.pow(2) * self.inverse_variance.unsqueeze(0)

    def chi_squared_spectrum(self, parameters):
        """(N, 15) or (15,) array -> total spectroscopic chi-squared.

        The chi-squared of the spectrum alone, without photometry or prior;
        ``chi_squared_per_line`` gives the same split by line. Parameters
        outside the prior box are moved to its nearest edge first.
        """
        array = np.ascontiguousarray(parameters, dtype=np.float64)
        single = array.ndim == 1
        if single:
            array = array[None, :]
        with torch.no_grad():
            batch = torch.as_tensor(array, dtype=torch.float64,
                                    device=self.device)
            clamped = torch.clamp(batch, self.lower_bound, self.upper_bound)
            result = self._chi_squared_spectrum(clamped).cpu().numpy()
        return float(result[0]) if single else result

    def _chi_squared_spectrum(self, parameters):
        """(N, 15) device tensor -> (N,) spectroscopic chi-squared."""
        return self._weighted_squared_residual(parameters).sum(dim=-1)

    def log_likelihood_spectrum(self, parameters):
        """(N, 15) -> (N,) Gaussian spectral log-likelihood."""
        return -0.5 * self._chi_squared_spectrum(parameters)

    def log_likelihood_magnitude(self, parameters):
        if self.photometry is None:
            return torch.zeros(parameters.shape[0], dtype=torch.float64,
                               device=self.device)
        simulated = self.photometry.magnitude(parameters[:, :N_THETA])
        residual = self.observed_magnitude - simulated
        return -0.5 * (residual / self.magnitude_uncertainty) ** 2

    # -- GPU evaluation --------------------------------------------------
    def forward(self, batch):
        """(N, 15) device tensor -> (N,) log-posterior, entirely on the GPU.

        Free of host synchronisation and data-dependent control flow, so it can
        be captured into a CUDA graph.
        """
        inside = ((batch > self.lower_bound) & (batch < self.upper_bound)).all(dim=1)

        # Clamp before evaluating so rejected walkers cannot produce inf/NaN
        # that would poison the reductions; they are masked out afterwards.
        clamped = torch.clamp(batch, self.lower_bound, self.upper_bound)

        log_probability = self.log_likelihood_spectrum(clamped)
        log_probability = log_probability + self.log_likelihood_magnitude(clamped)
        return torch.where(
            inside & torch.isfinite(log_probability),
            log_probability,
            torch.full_like(log_probability, -np.inf),
        )

    def forward_likelihood(self, batch):
        """(N, 15) device tensor -> (N,) log-likelihood, entirely on the GPU.

        The likelihood ``log_likelihood`` returns, in a form that can be
        captured into a CUDA graph: points outside the prior box are evaluated
        at the nearest point on its boundary rather than rejected.
        """
        clamped = torch.clamp(batch, self.lower_bound, self.upper_bound)
        return (self.log_likelihood_spectrum(clamped)
                + self.log_likelihood_magnitude(clamped))

    # -- sampler entry points --------------------------------------------
    @torch.no_grad()
    def __call__(self, parameters, allow_graph=True):
        """(N, 15) or (15,) array -> (N,) or scalar log-posterior."""
        array = np.ascontiguousarray(parameters, dtype=np.float64)
        single = array.ndim == 1
        if single:
            array = array[None, :]
        if array.shape[1] != N_DIM:
            raise ValueError(f"expected {N_DIM} parameters, got {array.shape[1]}")

        n_rows = array.shape[0]
        size, entry = self._graph_for(n_rows) if allow_graph else (None, None)
        if entry is not None:
            result = self._replay(size, entry, array)
        else:
            batch = torch.as_tensor(array, dtype=torch.float64,
                                    device=self.device)
            result = self.forward(batch).cpu().numpy()

        if self.prior is not None:
            finite = np.isfinite(result)
            if finite.any():
                extra = np.asarray(self.prior.log_prob(array), dtype=np.float64)
                result = np.where(finite, result + extra, result)
                result = np.where(np.isfinite(result), result, -np.inf)

        return float(result[0]) if single else result

    def log_likelihood(self, parameters, allow_graph=True):
        """(N, 15) array -> (N,) log-likelihood, ignoring the prior.

        Samplers that handle the prior themselves -- nested samplers in
        particular -- want this rather than the posterior. Values outside the
        prior box are evaluated at the nearest point on the boundary.

        Uses a captured CUDA graph when one large enough has been set up with
        ``enable_cuda_graph(..., kind="likelihood")``.
        """
        array = np.ascontiguousarray(parameters, dtype=np.float64)
        single = array.ndim == 1
        if single:
            array = array[None, :]
        size, entry = (self._graph_for(array.shape[0], self._likelihood_graphs)
                       if allow_graph else (None, None))
        if entry is not None:
            result = self._replay(size, entry, array)
        else:
            with torch.no_grad():
                batch = torch.as_tensor(array, dtype=torch.float64,
                                        device=self.device)
                result = self.forward_likelihood(batch).cpu().numpy()
        return float(result[0]) if single else result

    def prior_transform(self, unit_cube):
        """Map points from the unit hypercube onto the prior.

        This is the form nested samplers such as UltraNest and dynesty expect.
        Accepts ``(15,)`` or ``(N, 15)`` and returns the same shape.

        With an informative prior this uses its marginal distributions. Any
        relations that prior carries cannot be inverted, so a nested sampler
        must add ``prior.relation_log_prob`` to its likelihood instead; see
        ``samplers.run_nested``.
        """
        cube = np.asarray(unit_cube, dtype=np.float64)
        if self.prior is not None:
            return self.prior.transform(cube)
        lower, upper = BOUNDS_ARRAY[:, 0], BOUNDS_ARRAY[:, 1]
        return lower + cube * (upper - lower)

    # -- analysis --------------------------------------------------------
    def chi_squared_per_line(self, parameters):
        """(N, 15) or (15,) array -> {line name: chi-squared of that line}.

        The spectroscopic chi-squared, without photometry or prior, split by
        line; adding up the values gives ``chi_squared_spectrum``. Parameters
        outside the prior box are moved to its nearest edge first.
        """
        array = np.ascontiguousarray(parameters, dtype=np.float64)
        single = array.ndim == 1
        if single:
            array = array[None, :]
        with torch.no_grad():
            batch = torch.as_tensor(array, dtype=torch.float64,
                                    device=self.device)
            clamped = torch.clamp(batch, self.lower_bound, self.upper_bound)
            weighted = self._weighted_squared_residual(clamped)
            per_line = {line.name: weighted.index_select(1, line.columns)
                        .sum(dim=-1).cpu().numpy() for line in self.lines}
        if single:
            return {name: float(value[0]) for name, value in per_line.items()}
        return per_line

    def merged_model_per_line(self, parameters):
        """(N, 15) or (15,) array -> {line name: (wavelength, flux)}.

        The values ``merged_model`` returns, split by line and paired with
        that line's observed wavelengths, for saving or plotting a model's
        profile line by line. ``flux`` is (N, n_points), or (n_points,) for a
        single parameter vector.
        """
        array = np.ascontiguousarray(parameters, dtype=np.float64)
        single = array.ndim == 1
        if single:
            array = array[None, :]
        with torch.no_grad():
            batch = torch.as_tensor(array, dtype=torch.float64,
                                    device=self.device)
            clamped = torch.clamp(batch, self.lower_bound, self.upper_bound)
            model = self.merged_model(clamped)
            result = {}
            for line in self.lines:
                flux = model.index_select(1, line.columns).cpu().numpy()
                result[line.name] = (line.wavelength, flux[0] if single else flux)
        return result

    def covered_point_counts(self):
        """{line name: number of observed points in that line's window}.

        Dividing ``chi_squared_per_line`` by these gives a reduced
        chi-squared per line.
        """
        return {line.name: int(line.indices.size) for line in self.lines}

    # -- CUDA graph capture ----------------------------------------------
    def _graph_kind(self, kind):
        """(function to capture, public evaluator, graph store) for ``kind``."""
        if kind == "posterior":
            return self.forward, self.__call__, self._graphs
        if kind == "likelihood":
            return (self.forward_likelihood, self.log_likelihood,
                    self._likelihood_graphs)
        raise ValueError("kind must be 'posterior' or 'likelihood'")

    def _capture(self, batch_size, tolerance=1e-9, kind="posterior"):
        """Record one CUDA graph for batches of ``batch_size`` and test it.

        The function recorded is ``forward`` (log-posterior) or
        ``forward_likelihood`` (log-likelihood), depending on ``kind``. Here
        "normal evaluation" means running that function as usual, with the
        CPU sending each GPU step one by one; that result is taken as correct.

        1. Draw ``batch_size`` test parameter vectors uniformly inside the
           prior box (always with the same random seed) and evaluate them
           normally. These values are what the graph has to reproduce.
        2. Warm-up: run the function three times. The first runs make PyTorch
           and the FFT library set aside the GPU memory and working space
           they need; that cannot happen while a graph is being recorded.
        3. Record: run the function once more, but with the GPU steps stored
           in the graph instead of executed. The graph remembers which GPU
           arrays each step reads and writes, so it always takes its input
           from one fixed array and writes to one fixed output array.
        4. Replay: copy the test vectors into that input array, run the
           stored steps with a single instruction, and compare the output
           with step 1. A replay is exactly how the graph is used later, so
           this checks the graph itself, not just the recording.
        5. Check padding. Which graph a call uses is decided by
           ``_graph_for``: the smallest recorded graph at least as large as
           the batch, or normal evaluation if the batch is larger than every
           recording. With graphs for 64 and 128, for example, a batch of 20
           or of 64 uses the 64 graph, a batch of 100 the 128 graph, and a
           batch of 2000 is evaluated normally.

           So this graph also receives batches smaller than ``batch_size``.
           Those are padded: a batch of 20 in the 64 graph has its other 44
           rows filled with copies of its first vector, and their results
           are discarded. This step tests that the real rows still come out
           right: it pads a third of the test vectors up to ``batch_size`` in
           the same way, replays them, and compares the real rows with a
           normal evaluation of the same padded batch. (Not with a normal
           evaluation of the real rows alone: float32 sums can differ in the
           last digit between batch sizes, which would look like an error.)

           Batches larger than ``batch_size`` are never sent to this graph,
           so they need no test here; every larger graph runs these same
           tests when it is recorded.

        Returns the stored graph, or None if recording fails or a check
        differs by more than ``tolerance`` (in log-probability); that batch
        size then runs the normal way.
        """
        function, evaluate, graphs = self._graph_kind(kind)
        lower = self.lower_bound.cpu().numpy()
        upper = self.upper_bound.cpu().numpy()
        probe = np.random.default_rng(0).uniform(
            lower, upper, size=(batch_size, N_DIM))
        expected = evaluate(probe, allow_graph=False)

        try:
            static_input = torch.zeros(batch_size, N_DIM, dtype=torch.float64,
                                       device=self.device)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                with torch.no_grad():
                    for _ in range(3):
                        function(static_input)
            torch.cuda.current_stream().wait_stream(stream)

            graph = torch.cuda.CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(graph):
                static_output = function(static_input)
        except Exception:
            return None

        entry = {
            "graph": graph, "input": static_input, "output": static_output,
            # A CPU-side buffer for the input, in "pinned" memory that the GPU
            # can read directly, so copying each new batch in is fast.
            "staging": torch.empty(batch_size, N_DIM, dtype=torch.float64,
                                   pin_memory=True),
        }
        graphs[batch_size] = entry

        got = evaluate(probe)
        difference = float(np.max(np.abs(got - expected)))
        if not np.isfinite(difference) or difference > tolerance:
            del graphs[batch_size]
            return None

        # Step 5: the padding check.
        short = max(1, batch_size // 3)
        padded = np.vstack([probe[:short]]
                           + [probe[0][None, :]] * (batch_size - short))
        with torch.no_grad():
            reference = function(torch.as_tensor(
                padded, dtype=torch.float64,
                device=self.device)).cpu().numpy()[:short]
        padded_difference = float(np.max(np.abs(
            self._replay(batch_size, entry, probe[:short]) - reference)))
        if not np.isfinite(padded_difference) or padded_difference > tolerance:
            del graphs[batch_size]
            return None

        entry["error"] = max(difference, padded_difference)
        return entry

    def enable_cuda_graph(self, *batch_sizes, tolerance=1e-9, verbose=True,
                          kind="posterior"):
        """Speed up small batches by recording the GPU work once, then replaying.

        Why it helps. One evaluation of the posterior is not one big GPU
        calculation but several hundred small ones (emulator, broadening and
        residuals for every line window). The CPU has to send each of them to
        the GPU separately, and each send has a fixed cost. For small batches
        the GPU finishes every step faster than the CPU can send the next, so
        the run is limited by the sending, not by the calculation: a batch of
        16 parameter vectors takes about as long as a batch of 256.

        What it does. A CUDA graph is a recording of that whole sequence of
        steps. After recording it once, the CPU replays all of it with a
        single instruction, which removes most of the sending cost.

        The catch. A recording only works for the batch size it was made
        with. So graphs are recorded for several sizes (``batch_sizes``), and
        a batch is padded up to the nearest recorded size with copies of its
        first row, whose results are thrown away. This padding is almost free,
        because at these sizes the cost hardly depends on the batch size.
        Larger batches than any recording run the normal way. The batch size
        is the number of parameter vectors the sampler asks for at once: for
        emcee half the walkers (it moves one half of the ensemble while the
        other half stays put), for nested sampling the number of walkers still
        taking a step, which changes from call to call.

        Each recording is checked against the normal evaluation before it is
        used (see ``_capture``), so this changes the speed, never the result.

        ``kind="posterior"`` records ``forward``, used by ``__call__`` (emcee);
        ``kind="likelihood"`` records ``forward_likelihood``, used by
        ``log_likelihood`` (nested sampling). Returns True if any size was
        recorded; on a CPU it does nothing and returns False.
        """
        if self.device.type != "cuda":
            return False
        _, _, graphs = self._graph_kind(kind)
        sizes = sorted({int(size) for size in batch_sizes if int(size) > 0})
        captured, failed = [], []
        for size in sizes:
            entry = self._capture(size, tolerance, kind=kind)
            (captured if entry is not None else failed).append(size)

        if verbose:
            if captured:
                worst = max(graphs[size]["error"] for size in captured)
                print(f"[GPUPosterior] CUDA graphs ({kind}) captured for "
                      f"batches {captured} (match eager to {worst:.1e})")
            if failed:
                print(f"[GPUPosterior] {kind} capture failed for {failed}; "
                      f"those sizes fall back to eager evaluation")
        return bool(captured)

    def _graph_for(self, n_rows, graphs=None):
        """Smallest captured graph that can hold ``n_rows``, or None.

        Looks among the posterior graphs unless another store is given.
        """
        graphs = self._graphs if graphs is None else graphs
        for size in sorted(graphs):
            if size >= n_rows:
                return size, graphs[size]
        return None, None

    def _replay(self, size, entry, array):
        """Evaluate ``array`` on a captured graph of batch ``size``."""
        n_rows = array.shape[0]
        # Pad the unused rows by repeating the first, so they are always
        # valid input; their results are discarded.
        staging = entry["staging"].numpy()
        staging[:n_rows] = array
        if size > n_rows:
            staging[n_rows:] = array[0]
        entry["input"].copy_(entry["staging"], non_blocking=True)
        entry["graph"].replay()
        return entry["output"].cpu().numpy()[:n_rows].copy()

    def disable_cuda_graph(self):
        """Drop every captured graph and go back to eager evaluation."""
        self._graphs = {}
        self._likelihood_graphs = {}
