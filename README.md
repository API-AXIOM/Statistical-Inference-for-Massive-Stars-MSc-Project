# Statistical inference for massive stars

Fits the 15-parameter FASTWIND model to the spectrum of an OB star on a single
GPU, with either ensemble MCMC (`emcee`) or nested sampling (UltraNest). The
model spectrum comes from neural-network emulators of FASTWIND, so a fit that
would take weeks with FASTWIND itself finishes in under an hour.

New to working on a GPU cluster or outside notebooks? Start with
**[GETTING_STARTED.md](GETTING_STARTED.md)**. It walks through a first fit step
by step and explains the vocabulary.

```bash
source env.sh                                      # once per terminal
python3 inspect_spectrum.py my_star.fits --plot check.png

sbatch scripts/submit_sampler.slurm --data my_star.fits --instrument LMC_stars_R6700 \
       --snr 100 --magnitude 13.4 --nwalkers 512 --nsteps 20000 \
       --outdir my_star          # -> results/mcmc/my_star/

# the same fit with nested sampling
sbatch scripts/submit_sampler.slurm --data my_star.fits --instrument LMC_stars_R6700 \
       --snr 120 --magnitude 13.4 \
       --sampler nested --n-live 600 --ns-steps 8 --ns-walkersize 50 \
       --outdir my_star          # -> results/nested_sampling/my_star/
```

---

## Installation

**Requirements.** An NVIDIA GPU with CUDA, Python 3 (developed with 3.12), and the
packages in [`requirements.txt`](requirements.txt) (PyTorch, NumPy, SciPy,
emcee, UltraNest, h5py, astropy, matplotlib, corner, tqdm). The code also runs
on a CPU (`--device cpu`), but far too slowly for a real fit.

```bash
git clone https://github.com/API-AXIOM/Statistical-Inference-for-Massive-Stars-MSc-Project.git
cd Statistical-Inference-for-Massive-Stars-MSc-Project
python3 -m venv venv && source venv/bin/activate
python3 -m pip install -r requirements.txt
mkdir -p logs                      # where SLURM jobs write their logs
```

`scripts/setup_env.sh` does the same on a cluster with the EESSI software
stack; `env.sh` then switches the environment on (see
[Running on a SLURM cluster](#running-on-a-slurm-cluster)).

**Emulators.** The fit needs the trained FASTWIND emulators of Vasilis
Kolympiris ([FASTWIND_emulator](https://github.com/API-AXIOM/FASTWIND_emulator)).
The code that loads them is included here, but the trained network files are
not: they are available from their author on request. Put them in
`emulators/` as described in [`emulators/README.md`](emulators/README.md), or
point `EMULATOR_ROOT` at a folder laid out the same way.

## What it fits

A parameter vector of 15 values, sampled with uniform priors over the box in
[`src/gpu_config.py`](src/gpu_config.py):

| | parameter |
|---|---|
| stellar | `teff`, `logg`, `radius` |
| wind | `mdot` (as log10), `beta`, `vinf` |
| abundances | `yhe`, `C`, `N`, `O` |
| clumping | `fic`, `fvel`, `fclump` |
| broadening | `vsini`, `vmacro` |

The first 13 are the emulators' inputs; `vsini` and `vmacro` are applied
afterwards by the broadening.

The likelihood has two terms:

* **Spectrum.** Each line-emulator window is predicted, broadened for rotation,
  macroturbulence and the instrument, and compared with the data. Gaussian,
  with sigma = 1/SNR, or with each pixel's own uncertainty from the data file
  when `--use-errors` is given.
* **Photometry.** The continuum emulator is integrated through a filter and
  compared with an observed magnitude. Gaussian, with a configurable
  uncertainty (default 0.1 mag). Switch it off with `--no-magnitude`.

Each line is fitted over a fixed wavelength window, listed in
`LINE_WINDOW_OVERRIDES` (`src/gpu_config.py`). The windows do not overlap, so
every observed wavelength belongs to at most one of them; neighbouring
windows may share an edge, and the fit refuses windows that overlap. Each
window is broadened on its own, so its edges should lie in the continuum: a
line wing that crosses an edge is cut off there. Check the windows against
your spectrum before fitting; `inspect_spectrum.py` lists them.

## Layout

```
├── env.sh                  # switch the environment on; source this first
├── run_sampler_gpu.py      # the fitting program: MCMC or nested sampling
├── inspect_spectrum.py     # check how an observation will be read
├── requirements.txt
├── README.md               # this overview
├── GETTING_STARTED.md      # step-by-step guide for a first run
├── config/
│   ├── instruments.json    # wavelength grids and resolving powers
│   └── priors.json         # LMC baselines, wind relations, mass-loss recipes
├── scripts/
│   ├── setup_env.sh        # build the environment (once)
│   ├── submit_sampler.slurm # submit a fit to SLURM
│   ├── interactive_gpu.sh  # interactive shell on a GPU node
│   └── replot.py           # redraw a finished run's figures
├── src/
│   ├── gpu_posterior.py    # the log-posterior; the piece samplers call
│   ├── samplers.py         # nested sampling, and the emcee proposal moves
│   ├── gpu_broadening.py   # batched rotation, macroturbulence and instrumental broadening
│   ├── gpu_emulator.py     # batched emulator evaluation
│   ├── gpu_photometry.py   # batched synthetic magnitudes
│   ├── gpu_config.py       # the parameter vector, its bounds, the line windows
│   ├── priors.py           # flat and LMC-informed priors
│   ├── physics.py          # derived quantities and mass-loss recipes
│   ├── map_estimate.py     # starting-point search
│   ├── version.py          # what produced a set of results
│   └── data_io.py, plots.py, emulators.py
└── emulators/
    ├── line_emulators_13par/   # line emulator loader; trained files go in models/
    ├── fluxcont/               # continuum emulator loader; trained file goes in model/
    └── filter_transmissions/   # filter curves and zero points for the photometry
```

## How it is fast

Four things, none of which changes the answer.

**The emulator's wavelength half is computed once.** A DeepONet predicts
`flux = sum_l trunk(lambda)[l] * branch(theta)[l]`. The trunk runs once per
wavelength point and the branch once per parameter vector, so the trunk is the
expensive half, and during a fit the wavelengths never change. It is
evaluated at setup and stored as a basis matrix, leaving one small network and
one matrix multiply per prediction.

**Broadening happens in the Fourier domain.** Convolution becomes
multiplication there, the three kernels compose in one pass, and, which is
what makes batching possible, the cost stops depending on kernel width. A
batch of parameter vectors with different `vsini` and `vmacro` can therefore
be broadened together, sharing one transform length. See the module docstring
in [`src/gpu_broadening.py`](src/gpu_broadening.py) for the derivation.

**Everything constant is built once.** Filter curves, zero points, instrument
profiles, wavelength cuts and transform plans are all set up before sampling
starts.

**The whole evaluation is captured as a CUDA graph.** With ~37 line windows
the model issues several hundred small GPU instructions per call, and the
instruction overhead dominates: evaluated eagerly, 512 parameter vectors cost
barely more than one. Since the batch shape is fixed, the sequence is recorded
once and replayed. The recorded version is compared against the unrecorded one
before use, and any mismatch or capture failure falls back automatically;
`--no-cuda-graph` is also possible, but changes speed.

Measured on an RTX 6000 Ada, 37 line windows, dense grid, milliseconds per
batch:

| batch | eager | with graph |
|------:|------:|-----------:|
|   1   |  23.8 |     4.3    |
|  32   |  24.8 |     6.9    |
| 128   |  25.1 |    10.9    |
| 512   |  28.3 |    26.3    |

So **prefer a wide ensemble**: 128 parameter vectors cost 2.5 times what one
does, not 128 times, and 512 parameter vectors give sixteen times the samples
of 32 for almost 4 times the cost per step.

TF32 arithmetic, which is a reduced-precision mode called TF32. It's faster
but only accurate to roughly 3 significant figures, which is why it is off
by default, since it would put ~1e-3 relative error into the emulator output;
`--tf32` enables it.

## Reading an observation

`--data` accepts `.fits`, `.npz`, and plain-text files (`.txt`, `.dat`,
`.ascii`, `.norm`).

A FITS file must hold the spectrum as a binary table with wavelength and
flux columns. The extension and the column names are found automatically from the usual
spellings (`WAVE`, `WAVELENGTH`, `LAMBDA`, `FLUX`, `NORM_FLUX`, ...), and can
be set explicitly with `--fits-extension`, `--wavelength-column` and
`--flux-column`. Wavelengths are converted to Angstrom using the file's unit
keyword. If there is no unit keyword, or it isn't recognised, the unit is
guessed from the size of the values: a median below 10 is taken as microns,
10–1000 as nm, and anything larger as Angstrom. The guess is printed when the
file is loaded. An error column is read if one is found. The star's observed
magnitude is always given on the command line with `--magnitude`.

An `.npz` must hold `wavelength` and `flux`; `error` is optional.

A plain-text file holds whitespace-separated columns `wavelength flux [error]`,
with wavelengths in Angstrom; lines starting with `#` are ignored.

Either way the flux must be **continuum normalised**, since it is compared
against a normalised model. Non-finite points are dropped and the spectrum is
sorted into ascending wavelength order.

The model does not include a radial-velocity shift, so the spectrum must
already be corrected to the star's rest frame.

Check a new file before fitting it:

```bash
python3 inspect_spectrum.py my_star.fits --plot check.png
```

That prints the file's structure, what would be read from it, whether the flux
looks normalised, an SNR implied by the point-to-point scatter, which line
windows the wavelength range covers, and how the data are sampled inside each
of them. The implied SNR assumes a stretch of continuum: on a spectrum cut
down to its line windows the line depths count as noise and the number is
meaningless.

If the file carries an uncertainty column, `--use-errors` weights each pixel by
it instead of by a single `--snr` (`--error-column` names the column in a FITS
table if it is not found automatically).

## Instrument setups

`config/instruments.json` maps a name to `[lambda_min, lambda_max, sampling]`
segments and a resolving power. `dense` covers every line window at 0.033 Å.

`--instrument` accepts more than one name for a spectrum stitched from
several arms or gratings with different resolving powers, for example
`--instrument Martins2017_stars_uv Martins2017_stars_opt` for a full
spectrum, where the resolution in the UV is 1500 and 8000 in the optical.
Each named instrument contributes its own wavelength range and resolving power;
every line window is assigned the resolving power of whichever range contains
it, and a line whose own window straddles two ranges raises an error.

With `--data` the wavelengths come from your file and only the resolving
power(s) come from here; override a single-instrument fit's resolving power
with `--resolution`. With `--mock` the segments define the grid to simulate.

### Wavelength and velocity grids

Spectra arrive on one of two kinds of grid, and the code handles both:

* **constant `d(lambda)`**: evenly spaced in wavelength;
* **constant `d(lambda)/lambda`**: evenly spaced in *velocity*, equivalently
  in `ln(lambda)`.

Which one you have is detected automatically from the data and reported at
setup. It matters: rotation, macroturbulence and a fixed resolving power are
all *velocity* broadenings, so on a velocity grid the kernels are identical at
every wavelength and the treatment is exact. On a wavelength grid they vary
slowly across a window and a single reference wavelength is used instead,
which is a good approximation over one line window and a poor one over a wide
range.

For `--mock`, `config/instruments.json` says which to build through the
`"sampling"` key: `"wavelength"` (step in Å) or `"velocity"` (step in km/s).

A line is fitted only if the observed spectrum covers its whole window with
evenly spaced points. A line whose window contains a gap or a change of
sampling, or that the spectrum covers only in part, is left out, and the fit
says which lines and why. `inspect_spectrum.py` applies the same check and
lists these lines before you fit.

## Photometry

Filter files live in `emulators/filter_transmissions/` (or pass another
folder with `--filter-dir`). Each is a whitespace-separated `wavelength transmission`
table, alongside a `zero_points.dat` with one row per filter and columns for
the Vega, AB and ST systems. The transmission curves come from the
[SVO Filter Profile Service](http://svo2.cab.inta-csic.es/theory/fps/); the
curves and zero points are taken from Kiwi-GA (Brands et al. 2022, A&A 663,
A36). Available:

| filter | file | coverage |
|---|---|---|
| `SPHERE_Ks` | `SPHERE_IRDIS_B_Ks.dat`  | 4,000–27,000 Å |
| `HST_555w`  | `HST_ACS_HRC.F555W.dat`  | 4,447–6,442 Å |
| `2MASS_Ks`  | `2MASS_Ks.dat`           | 19,000–23,990 Å |
| `VISTA_Ks`  | `Paranal_VISTA.Ks.dat`   | 19,080–24,140 Å |
| `Johnson_V` | `GCPD_Johnson.V.dat`     | 4,718–7,682 Å |

It is also possible to add other filters, make sure to also edit
`zero_points.dat` when doing so. The photometric term costs under
2% of an evaluation.

### Extinction

The model SED is unreddened, so `--extinction` adds a fixed A_lambda for the
chosen band to the model magnitude, making the model star fainter, before
comparison. It is a number you supply.

If you have photometry in several bands, comparing the observed colours with
the star's intrinsic colours gives E(B-V), and an extinction law converts that
to A_lambda for whichever band you are fitting.

## Inference methods

Two are available. Both write the same `results.npz`, `summary.txt` and
figures, so the plots and any later analysis do not care which produced them.

| method | `--sampler` | main settings (default) | starts from |
|---|---|---|---|
| ensemble MCMC (`emcee`) | `emcee` (default) | `--nwalkers` (512), `--nsteps` (20,000), `--move` (`stretch`) | a MAP estimate |
| nested sampling (UltraNest) | `nested` | `--n-live` (600), `--ns-steps` (8), `--ns-walkersize` (50) | the whole prior |

Both run vectorised, so the GPU is handed a batch of parameter vectors
rather than one at a time. Without that, either method would throw away the
speed advantage entirely.

**MCMC** first searches for the best-fitting parameters, given by 10 optimiser
runs, each started from the best point of its own 20,000-point random scan
of the prior (--map-prescan). You can change the scan size with --map-prescan N
or turn it off with --map-prescan 0. MCMC then starts its walkers around
them, and lets them explore the posterior for `--nsteps` steps. Run MCMC long
enough: the program reports the autocorrelation time and warns when one of
the chains is shorter than about 50 autocorrelationtimes, which is what MCMC
needs to confidently constrain the posterior.

Proposal moves. --move sets how each walker proposes its next position. The
default, `stretch`, proposes a point on the line through the walker and another
randomly chosen walker. It works well when the posterior is roughly round, but
can mix slowly when parameters are strongly correlated, as stellar parameters
often are (temperature trades off against radius, and the wind parameters against
each other). `de` (differential evolution) proposes a step along the difference
between two other walkers, so the ensemble learns the shape of such correlations.
`de-snooker` mixes 80% differential-evolution steps with 20% DE-snooker steps, a
variant of the same idea; this is the combination emcee's documentation suggests
for correlated problems. All results so far have used stretch, which is the only
move that has been tested properly. The other two are available.

**Nested sampling** keeps a set of live points (`--n-live`) and repeatedly
replaces the worst of them with a new point of higher likelihood, so the set
contracts onto the posterior. Besides posterior samples it returns the
evidence, log Z. In 15 dimensions the new points are found with a slice
sampler: starting from an existing live point, it searches along a random
direction for a point above the current likelihood threshold, and repeats
this `--ns-steps` times so that the new point is independent of where it
started. Fewer than 8 steps gave posteriors that were too narrow on test
problems, which is why 8 is the default. A population of `--ns-walkersize`
such searches runs together so that each round is one GPU batch; this sets the
batch size, not the statistics. Each likelihood call replays a captured CUDA
graph (see "How it is fast"), which keeps the many small batches cheap.

A nested run stops once at most 1% of the posterior mass is estimated to lie
beyond the live points. The tolerances on the evidence (dlogz = 10) and on
the information gain (dKL = 2) are set loose, because the program is used
for posteriors rather than model comparison. While it runs it prints a
progress line every 30 seconds; the most useful part is `logL live A..B`, the
likelihood of the worst and best live points. When the gap closes to roughly
10–20, the live points have reached the main body of the posterior.

`--ns-method randomwalk` replaces the slice steps with random-walk steps. It is
experimental and not recommended: on a full 37-line fit it either stopped with
an error or settled on the wrong part of parameter space.

Nested sampling needs the likelihood and the prior separately, with the prior
entering through a unit-hypercube transform. `GPUPosterior` provides
`log_likelihood`, `prior_transform` and `log_prob` for this reason. Priors
carrying *relations* between parameters cannot be inverted into a transform,
so those terms are added to the likelihood instead: the same posterior,
factorised differently.


## Priors

Flat by default: uniform over the box in `gpu_config.py`. `--prior
informative` adds what is known of LMC stars. The emulators were trained at
LMC metallicity (Z = 0.5 Z☉), so every star is treated as an LMC star.
Note that these informative priors have **not** been tested and properly
checked, I unfortunately did not have the time to do so.

| source | affects | note |
|---|---|---|
| LMC values (`lmc` block of `config/priors.json`) | C, N, O, yhe, and the wind through relations | measured from H II regions and other stars, independent of the spectrum being fitted |

Abundances are applied **one-sided** from the LMC baseline, because CNO
processing only raises nitrogen and lowers carbon and oxygen.

`--no-relations` drops the physical relations, which put distributions on
quantities derived from several parameters at once: the mass implied by log g
and radius, the ratio of terminal to escape velocity, the mass-loss rate a
recipe predicts. These forbid unphysical *combinations* that no per-parameter
range can exclude.

The mass-loss recipe, its metallicity exponent and its normalisation are set in
`config/priors.json` rather than hardcoded, because which to use is a physics
choice; the notes there explain the options.

## Simulated spectra

`--mock` builds a spectrum from known parameters (`--true-params`, 15 values
in the order of the table above, `mdot` as log10) with the same forward model,
adds noise at `--snr`, and fits it. Combined with `--data`, the data file
supplies only the wavelength grid. The simulated spectrum is saved as
`mock_observation.npz`, and the run prints the log-posterior at the true
parameters next to the value a perfect fit would reach.

## Outputs

Both samplers write:

| file | contents |
|---|---|
| `summary.txt` | the 16/50/84 percentile table, and how the fit was made |
| `results.npz` | posterior samples (`flat_samples`), percentiles, best fit, timings, settings; `sampler` records which method made it |
| `corner.png`, `posterior_predictive.png`, `full_spectrum.png` | diagnostics |
| `mock_observation.npz` | the simulated spectrum, for `--mock` runs |

MCMC (`--sampler emcee`) also writes:

| file | contents |
|---|---|
| `chain.h5` | the raw chain, every walker at every step; `--resume` continues from it |
| `chains.png`, `log_posterior.png` | walker traces and log-posterior against step number |
| `map_estimate.npz` | the starting-point estimate, reused on `--resume` |

Nested sampling (`--sampler nested`) also writes `ultranest/`, UltraNest's
checkpoints and logs, from which `--resume` continues. Its `results.npz`
additionally holds `log_evidence` and `log_evidence_error` (log Z),
`effective_sample_size`, `n_evaluations` (the number of likelihood calls), and
the weighted samples (`weighted_points`, `weighted_weights`) from which
`flat_samples` were drawn.

`results.npz` also carries `pipeline_version`: a hash of the code and
configuration files that determine a fit (`source_fingerprint`), a hash of the
emulator checkpoints (`emulator_fingerprint`), the git commit (marked `-dirty`
if there were uncommitted changes), the versions of the main packages, and
when and where the run happened. Two results with the same pair of
fingerprints came from identical code, configuration and emulators.

Long runs are safe to interrupt. Rerun the same command with `--resume`; for
MCMC, also raise `--nsteps` to the total chain length you want. A nested run
continues from its checkpoint until it converges, so it needs `--resume` alone.
`python3 scripts/replot.py results/mcmc/my_star` (or
`results/nested_sampling/my_star`) redraws the figures of a finished run
without re-sampling.

## Using the model from your own code

`GPUPosterior` is independent of any particular sampler. Build it once, then
call it however you like:

```python
import sys; sys.path.insert(0, "src")          # from the repository folder
from emulators import load_emulators
from gpu_posterior import GPUPosterior

lines, sed = load_emulators()
posterior = GPUPosterior(
    observed_wavelength=wavelength, observed_flux=flux,
    observed_magnitude=13.4, spectrum_snr=100, resolution=6700,
    line_emulators=lines, sed_emulator=sed)

posterior(theta_batch)                  # (n, 15) -> (n,) log-posterior
posterior.log_likelihood(theta_batch)   # (n, 15) -> (n,) log-likelihood
posterior.prior_transform(unit_cube)    # unit hypercube -> parameters
posterior.merged_model(theta_tensor)    # model spectrum at covered wavelengths
```

`log_likelihood` and `prior_transform` together are the interface nested
samplers such as UltraNest expect. Any sampler that can evaluate a batch
of points at once gets the full speed benefit; one that insists on a
single point at a time still works, but at roughly the cost of a batch.

## Running on a SLURM cluster

`scripts/submit_sampler.slurm` submits one fit; everything after the script name
is passed to `run_sampler_gpu.py`. Submit it from the repository folder. Its
`#SBATCH` lines (partition, GPU type, CPUs, memory, time) are set for the
cluster the code was developed on (HIPSTER, UvA); change them to match yours.
`scripts/interactive_gpu.sh [hours]` opens a shell on a GPU node with the
environment loaded.

* One GPU per fit. The work is latency bound, so a second GPU would not speed
  up a single chain; fit several stars as several jobs.
* Many clusters require each GPU job to take a minimum number of CPUs per GPU
  and reject smaller requests at submission. If yours does, raise
  `--cpus-per-task` to match.
* Asking for more memory than the CPUs' share can make SLURM raise the CPU
  count to match, after which `srun` stops with `cpus-per-task set by two
  different environment variables`. Pass the raised CPU count as well.
* `env.sh` loads Python from the EESSI software stack where it is available
  and then activates `venv/`; elsewhere it only activates `venv/` and sets the
  paths. Adapt its module block if your cluster provides Python differently.

## Known limitations

* **Lines on unevenly sampled or partly covered data are left out.**
  Broadening needs evenly spaced flux across the whole line window, so a line
  whose window spans a gap, the join between two spectrograph arms, a change
  of sampling, or the end of the spectrum is not fitted. `inspect_spectrum.py`
  lists such lines. To include one, adjust its window in
  `LINE_WINDOW_OVERRIDES` (`src/gpu_config.py`) so it lies within evenly
  sampled data, or resample the spectrum onto an even grid.
* **Metallicity.** The emulators were trained at LMC metallicity, so every star
  is fitted as an LMC star.
* **Informative priors.** The informative priors and relations have not been
  tested and have not been extensively checked against the literature.
* **Emulator FASTWIND discrepancies.** Vasilis' emulator and FASTWIND don't
always agree. For more details, read his thesis.

## Credits

The FASTWIND line and continuum emulators are by Vasilis Kolympiris
([FASTWIND_emulator](https://github.com/API-AXIOM/FASTWIND_emulator)). If you
use them, please cite his MSc thesis, *Exploring the Composition and Evolution
of Massive Stars: Predicting stellar spectra of O-type stars using neural
networks*.

## License

The code is released under the MIT License. The FASTWIND line and continuum
emulators are by Vasilis Kolympiris
([FASTWIND_emulator](https://github.com/API-AXIOM/FASTWIND_emulator); please
cite his MSc thesis, *Exploring the Composition and Evolution of Massive Stars:
Predicting stellar spectra of O-type stars using neural networks*). FASTWIND
itself and any FASTWIND input or output datasets remain subject to their own
access and licensing conditions.
