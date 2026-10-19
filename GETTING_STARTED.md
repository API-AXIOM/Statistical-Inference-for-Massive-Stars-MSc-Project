# Getting started

A step-by-step guide to running a fit on a GPU cluster that uses the SLURM
queueing system. If a word is unfamiliar, check the glossary at the bottom.

**Contents**

1. Open a terminal on the cluster
2. Get the code and the emulators
3. Turn the environment on
4. Understand where the work happens
5. Get your spectrum onto the cluster
6. Look at the spectrum before fitting it
7. Run a fit
8. Read the results
9. Resuming an interrupted or extended run
10. Choosing settings
11. When something goes wrong

---

## 1. Open a terminal on the cluster

Everything here happens in a **terminal**, a window where you type commands
instead of clicking. In VS Code connected to the cluster, `Terminal -> New
Terminal`. You will see a prompt ending in `$`. You type a command, press
Enter, it runs, and you get the prompt back.

Three commands are enough to move around:

```bash
cd ~/some_folder               # go to a folder ("change directory")
ls                             # list what is in the current folder
cat README.md                  # print a file to the screen
```

## 2. Get the code and the emulators

Download the code into your home folder:

```bash
cd ~
git clone https://github.com/API-AXIOM/Statistical-Inference-for-Massive-Stars-MSc-Project.git
cd Statistical-Inference-for-Massive-Stars-MSc-Project
```

This folder is called **the repository folder** below. Every command in this
guide is typed from inside it.

The fit also needs the trained FASTWIND emulators of Vasilis Kolympiris
([FASTWIND_emulator](https://github.com/API-AXIOM/FASTWIND_emulator)), which
are not part of this repository. Ask him for the files, then put them where
[`emulators/README.md`](emulators/README.md) says. Check they are in place:

```bash
ls emulators/line_emulators_13par/models | wc -l    # should print 37
ls emulators/fluxcont/model                          # should list emulator_FLUXCONT.pth
```

## 3. Turn the environment on

A cluster usually does not have PyTorch, emcee and so on installed for
everyone, so the code keeps its own private copy of Python's packages in a
folder called `venv`. Build it once:

```bash
bash scripts/setup_env.sh
```

On a cluster with the EESSI software stack this loads Python first; elsewhere,
make sure `python3` is version 3 with `venv` available before running it.

Then, **once in every new terminal**, before anything else:

```bash
source env.sh
```

`source` means "run these commands in my current terminal and keep the
effects". If you forget, you will get `ModuleNotFoundError: No module named
'torch'`. Do not switch on a different environment afterwards (for example the
one your Jupyter notebooks use): it replaces this one and may lack packages the
fitting program needs.

To check it worked:

```bash
python3 -c "import torch; print(torch.__version__)"
```

You should see a version number.

## 4. Understand where the work happens

You are logged in to the **login node**, a shared machine for editing files
and submitting work. It usually has no GPU, and you should not run fits on it:
it is shared with everyone else on the cluster.

The actual computing happens on **compute nodes**, which you reach through
**SLURM**, the queueing system. You describe what you need, SLURM finds you a
free machine, runs your job, and saves the output to a file.

There are two ways to ask.

### A submitted job (use this for real runs)

```bash
sbatch scripts/submit_sampler.slurm --mock --nwalkers 512 --nsteps 20000 --outdir my_first_run
```

`sbatch` means "submit this to the queue". It prints something like
`Submitted batch job {your_job_number}` and returns immediately. The job runs
later, on its own, whether or not you stay logged in. Anything you put after
the script name is passed straight through to the fitting program.

The `#SBATCH` lines at the top of `scripts/submit_sampler.slurm` say what the job
asks for: a partition (a named group of machines), a GPU type, CPUs, memory
and a time limit. They are set for the cluster this code was developed on, so
check them against your cluster's documentation and change them if needed.

Check on a job:

```bash
squeue -u $USER        # your jobs. PD = pending (waiting), R = running
tail -f logs/sampler-{your_job_number}.out    # watch the output as it appears
```

`tail -f` follows a file as it grows. Press `Ctrl-C` to stop watching (this
does **not** stop the job). To actually cancel a job: `scancel {your_job_number}`.

### An interactive session (use this for poking around)

```bash
bash scripts/interactive_gpu.sh 2
```

This asks for a GPU for 2 hours and drops you into a terminal *on the compute
node*, with the environment already loaded. Its partition and GPU type are set
in the script, like those of the submission script. Now you can run things
directly and see the output immediately:

```bash
python3 run_sampler_gpu.py --mock --nwalkers 512 --nsteps 500 --outdir test
```

Type `exit` when you are done, so the GPU goes back to the pool.

You may have to wait for a slot; jobs are queued. `squeue -u $USER --start`
estimates when yours will begin.

## 5. Get your spectrum onto the cluster

Files on your own computer are not visible to the cluster; they have to be
copied across first. Two ways:

**From VS Code** (easiest if you are already connected). In the Explorer panel
on the left, open the folder you want, then drag the file into it. Or
right-click the folder and choose *Upload*.

**From a terminal on your own computer** (not the cluster terminal, a local
one):

```bash
scp ~/Desktop/my_star.fits <your-username>@<cluster-address>:~/data/
```

`scp` means "secure copy". It takes a local file and a destination of the form
`user@machine:/path/`. The `~` on your side is your own home folder; the one
after the `:` is your home folder on the cluster. For a whole folder, add `-r`.

Make the destination folder first, from the cluster terminal, and check the
file arrived afterwards:

```bash
mkdir -p ~/data
ls -lh ~/data
```

## 6. Look at the spectrum before fitting it

Different instruments write FITS files in different ways, so check how yours
will be read before spending GPU time on it:

```bash
python3 inspect_spectrum.py ~/data/my_star.fits --plot ~/data/check.png
```

This prints the file's internal structure, what the fitting program would read
from it, and which spectral lines your wavelength range actually covers. Check
four things:

* **wavelength range** looks right, in Angstrom;
* **sampling** is recognised as either wavelength or velocity for each line
  window, and "regular to ... of a pixel" is a small number. A window marked
  `IRREGULAR` falls on unevenly spaced data, and the fit will leave that line
  out; so will a line the spectrum covers only in part. The "line windows
  covered" list shows which lines will be fitted and why any are left out;
* **looks normalised** is `True`: the fit compares against a model whose
  continuum is 1, so the data must be normalised the same way;
* **line windows covered** is not empty, and each window listed there (in
  Angstrom) starts and ends in the continuum of your spectrum rather than
  inside a line. Each window is fitted on its own, so a line wing that crosses
  a window edge is cut off. The windows are set in `LINE_WINDOW_OVERRIDES` in
  `src/gpu_config.py`; they must not overlap.

It also prints an implied signal-to-noise from the point-to-point scatter,
which is a reasonable starting value for `--snr`.

If it cannot find the spectrum automatically, the structure printed at the top
tells you what to pass instead:

```bash
python3 inspect_spectrum.py my.fits --fits-extension 1 --wavelength-column LAMBDA --flux-column NORM_FLUX
```

The same three options, with the same names, work on `run_sampler_gpu.py`.

## 7. Run a fit

Every fit is made by the same program, `run_sampler_gpu.py`. What changes is
which **data** you give it (your own spectrum, or a simulated one) and which
**sampler** maps out the posterior:

| you want to... | sampler | section | time on one NVIDIA L4 GPU, 37 lines |
|---|---|---|---|
| fit your own spectrum | MCMC (`emcee`) | 7.1 | ~25 min |
| fit your own spectrum and get the evidence (log Z) | nested sampling | 7.2 | ~35 min |
| fit a simulated spectrum with known parameters | either | 7.3 | as above |

Times are for all 37 line windows; fewer lines are faster. They include a few
minutes of setup. The program prints its own estimate before it starts.

All the commands in this section submit a job with `sbatch` (section 4). To run
one in an interactive session instead, replace `sbatch scripts/submit_sampler.slurm`
with `python3 run_sampler_gpu.py` and keep everything after it.

**Paths.** `--outdir` is just a name for the run. `--outdir my_star` puts an
MCMC run in `results/mcmc/my_star/` and a nested-sampling run in
`results/nested_sampling/my_star/`, inside the repository folder, whichever
folder you start the program from. Your data lives elsewhere, so give its full
path (`~/data/...`).

### What the program reads

The program reads `.fits` directly. It also accepts plain-text files (`.txt`,
`.dat`, `.ascii`, `.norm`) with columns `wavelength flux [error]`, and `.npz`,
NumPy's own format, which is convenient if you are preparing the spectrum in a
notebook:

```python
import numpy as np
np.savez("my_star.npz",
         wavelength=wavelength_angstrom,   # 1-D array, Angstrom
         flux=normalised_flux)             # 1-D array, continuum normalised
```

### 7.1 Your own spectrum, with MCMC

MCMC (`emcee`) is the default sampler. It first searches for the best-fitting
parameters, starts its walkers around them, and lets them explore the
posterior for a fixed number of steps.

```bash
sbatch scripts/submit_sampler.slurm --data ~/data/my_star.fits --instrument dense --snr 100 --magnitude 12.87 --filter-name VISTA_Ks --distance 49.59e3 --nwalkers 512 --nsteps 20000 --outdir my_star
```

The log goes to `logs/sampler-{your_job_number}.out`.

What the main options mean (they work the same way in every fit below):

| option | meaning |
|---|---|
| `--data` | your `.fits`, `.npz` or plain-text file (use `--mock` instead to fit a simulated spectrum, section 7.3) |
| `--instrument` | which entry in `config/instruments.json` describes your data; list several for a spectrum stitched from several arms |
| `--snr` | signal-to-noise; sets the assumed flux uncertainty as 1/SNR |
| `--use-errors` | use the file's own per-pixel uncertainties instead of `--snr` |
| `--magnitude`, `--filter-name`, `--distance` | the photometric constraint (distance in pc) |
| `--no-magnitude` | fit the spectrum only |
| `--lines` | fit only these line windows, e.g. `--lines HGAMMA HDELTA HEII4200` (names as `inspect_spectrum.py` lists them) |
| `--prior` | `flat` (the default, uniform within the parameter bounds) or `informative` (LMC abundances, helium and wind relations) |
| `--sampler` | `emcee` (MCMC, the default) or `nested` (nested sampling, section 7.2) |
| `--nwalkers`, `--nsteps` | the size of the MCMC run (MCMC only; section 10) |
| `--outdir` | the run's name; results go in `results/mcmc/<name>/` or `results/nested_sampling/<name>/` |

`--instrument` only sets the resolving power and, for `--mock`, the wavelength
grid to simulate. With `--data` your file supplies the wavelengths, but the
resolving power still comes from here; override it with `--resolution 8500`
if your entry is not right. If your spectrum is stitched from more than one
arm or grating (different resolving powers in different wavelength ranges),
list every instrument that applies, e.g. `--instrument Martins2017_stars_uv
Martins2017_stars_opt`: each contributes its own range and resolving power,
and any line whose window would straddle two of them raises an error. To
describe your own instrument, add an entry to `config/instruments.json`.

See every option with:

```bash
python3 run_sampler_gpu.py --help
```

### 7.2 Your own spectrum, with nested sampling

**Nested sampling** starts from the whole prior rather than from a best-fit
estimate, and as well as the posterior it gives the **evidence** (log Z), the
number used to compare two models of the same data. Add `--sampler nested`,
drop `--nwalkers` and `--nsteps` (MCMC only), and add the three
nested-sampling settings:

```bash
sbatch scripts/submit_sampler.slurm --sampler nested --n-live 600 --ns-steps 8 --ns-walkersize 50 --data ~/data/my_star.fits --instrument dense --snr 100 --magnitude 12.87 --filter-name VISTA_Ks --distance 49.59e3 --outdir my_star
```

**Always pass `--n-live --ns-steps --ns-walkersize`.** UltraNest's
own defaults (400 live points, 400 walkers) are not the recommended settings;
section 10 explains what each one does.

There is no chain length to choose: the run stops by itself once it has
converged. While it runs, the log prints a line every 30 seconds. Watch the
`logL live A..B` part: when the gap between the two numbers has shrunk to
roughly 10–20, the run is close to finishing.

It writes `summary.txt`, `results.npz`, `corner.png`,
`posterior_predictive.png` and `full_spectrum.png`, just as MCMC does, plus a
folder `ultranest/` holding its checkpoints. There is no `chain.h5`,
`chains.png` or `log_posterior.png`, since nested sampling has no walkers
moving over time.

### 7.3 A simulated spectrum

The program can also make a simulated spectrum from the emulator, add noise at
the given `--snr`, and fit it, which shows how well the parameters can be
recovered for a given setup. Replace `--data` with `--mock`:

```bash
sbatch scripts/submit_sampler.slurm --mock --instrument dense --snr 100 --nwalkers 512 --nsteps 20000 --outdir my_first_run
```

`--instrument` now also sets the wavelength grid to simulate (`dense` is an
evenly sampled grid covering every line). The true parameters are a built-in
default set; choose your own with `--true-params` followed by the 15 values in
the order `teff logg radius mdot yhe C N O beta vinf fic fvel fclump vsini
vmacro` (`mdot` as log10). `--seed` sets the random numbers, including the
noise, so a different seed gives a different noisy spectrum. The simulated
spectrum is saved in the output folder as `mock_observation.npz`, and
`summary.txt` lists the truth next to the fitted values. For nested sampling,
make the same changes as in 7.2.

`--mock` can be combined with `--data`. The data file then supplies only the
wavelengths; its flux and magnitude are replaced by a simulated spectrum made
from `--true-params`, with the exact sampling, gaps and line coverage of the
real observation.

## 8. Read the results

Everything lands in the run's folder, `results/mcmc/<name>/` for MCMC or
`results/nested_sampling/<name>/` for nested sampling:

| file | what it is |
|---|---|
| `summary.txt` | the 16th/50th/84th percentile table, and how the fit was made |
| `corner.png` | the posterior, all parameters against each other |
| `posterior_predictive.png` | your spectrum with the fitted model over it |
| `full_spectrum.png` | the same across the whole wavelength range, as one high-resolution image |
| `results.npz` | the posterior samples and everything else except the raw chain, for loading back into a notebook |
| `chains.png` | *(MCMC only)* one panel per parameter, showing the walkers over time |
| `log_posterior.png` | *(MCMC only)* fit quality against step number |
| `chain.h5` | *(MCMC only)* the raw chain (every walker at every step), used for resuming and for the chain plots |
| `ultranest/` | *(nested sampling only)* checkpoints, used for resuming |

Load the results back into a notebook like this:

```python
import json
import numpy as np
r = np.load("results/mcmc/my_star/results.npz", allow_pickle=False)
print(list(r))                                      # what is in there
samples = r["flat_samples"]                         # (n_samples, 15) posterior samples
names = json.loads(str(r["parameter_names"]))       # the 15 parameter names

# MCMC only: the full history of every walker is in chain.h5, not in results.npz
import emcee
backend = emcee.backends.HDFBackend("results/mcmc/my_star/chain.h5", read_only=True)
chain = backend.get_chain()                         # (n_steps, n_walkers, 15)

# Nested sampling only
r = np.load("results/nested_sampling/my_star/results.npz", allow_pickle=False)
print(r["log_evidence"], "+-", r["log_evidence_error"], " ESS", r["effective_sample_size"])
```

`flat_samples` is what you want for anything statistical: each row is one
plausible set of parameters, and the spread across rows is your uncertainty.
Use it for histograms, medians, percentiles and corner plots. For a nested
run, `flat_samples` are drawn from the weighted points; `effective_sample_size`
is the number of independent samples they are worth.

**Two things to check before believing an MCMC fit.** The program prints
warnings for both.

1. *Convergence.* `chains.png` should look like flat, fuzzy bands, not like
   trends still drifting. If you see `WARNING chain may not be converged`,
   increase `--nsteps` (section 9).
2. *Sample count.* Fewer than about 1000 independent samples means the
   percentiles are noisy. Again: more steps.

**For a nested-sampling fit**, look at the `convergence:` line near the end of
`summary.txt`. It gives log Z with its uncertainty, the effective sample size
(a few thousand with the settings in 7.2), and the number of likelihood calls.
The run decides for itself when it has converged, so there is no chain to
inspect. If the summary says the run stopped at a call cap, its posterior
cannot be trusted.

## 9. Resuming an interrupted or extended run

The MCMC chain is written to `chain.h5` after every step. If a job reaches its
time limit, is interrupted, or you want more samples, continue it by rerunning
the original command with the same `--outdir`, data, instrument, likelihood,
photometry, prior, emulator and walker settings. Add `--resume` and increase
`--nsteps`.

`--nsteps` is the desired total chain length, not the number of extra steps.
For example, to extend a 10,000-step run to 20,000 steps:

```bash
sbatch scripts/submit_sampler.slurm --mock --nwalkers 512 --nsteps 20000 --outdir my_first_run --resume
```

This runs the remaining 10,000 steps. A node failure can lose at most the
current unfinished step, not the run.

To request more wall time up front, edit `#SBATCH --time=12:00:00` at the top
of `scripts/submit_sampler.slurm`.

A nested-sampling run resumes the same way, from its `ultranest/` folder, but
needs only `--resume` added to the original command; there is no `--nsteps` to
raise. Keep every other option the same, including the nested-sampling
settings.

## 10. Choosing settings

### MCMC

**Walkers.** The GPU evaluates walkers in batches, so a larger ensemble costs
far less than its size suggests. Use 512 walkers, the default. `emcee` requires
an even number and at least twice the number of fitted parameters: 30 for this
15-parameter model.

**Steps.** Start with 10,000–20,000 steps. Inspect `chains.png`, the reported
autocorrelation time, and any convergence warning before treating percentile
intervals as final. If the walkers still drift, or the chain is shorter than
roughly 50 autocorrelation times, extend the same run with `--resume` and a
larger `--nsteps` (section 9). The autocorrelation time estimated from a short
chain tends to be too small, so leave some headroom.

**Runtime.** Runtime depends on the GPU, the wavelength coverage, the number
of line windows and the walker count. The program prints an estimate for its
configuration before sampling and reports the measured milliseconds per step
while it runs. For a dense-grid simulated spectrum with all 37 lines on an
NVIDIA L4, 512 walkers take about 0.10 seconds per step: about 17 minutes for
10,000 steps or 35 minutes for 20,000, plus setup.

### Nested sampling

Nested sampling keeps a set of **live points** spread over the region of
parameter space still in play. Each iteration it discards the worst one and
replaces it with a new point of higher likelihood, found by moving a copy of
an existing live point a few **slice steps** at random. The three settings:

| option | what it is | use |
|---|---|---|
| `--n-live` | number of live points. More give a larger effective sample size and fits that agree better when repeated, at the cost of more iterations | 600 |
| `--ns-steps` | slice steps per new live point. Too few, and the new point stays too close to the one it was copied from, so the posterior comes out too narrow | 8 |
| `--ns-walkersize` | how many points are moved side by side in one GPU call. Sets the batch size, not the statistics | 50 |

These were chosen by fitting the same simulated 37-line spectrum several times
with each candidate setting. With 400 live points, log Z came out noticeably
off; more slice steps, more live points or more walkers made fits slower
without making them more consistent. Repeat fits of one spectrum scatter by
about 0.1 of a posterior half-width and by about 1 in log Z, so treat a single
run's log Z as uncertain by about ±1, whatever error it quotes.

**Stopping.** A run stops once less than 1% of the posterior is estimated to
lie outside what it has explored. There is nothing to set.

### Which GPU

One GPU per fit: a second GPU does not make a single fit faster, so fit
several stars as several jobs. To ask for a different GPU, partition or CPU
count for one job, pass SLURM options *before* the script name, for example:

```bash
sbatch --partition=<partition> --gres=gpu:<gpu type>:1 --cpus-per-task=32 \
  scripts/submit_sampler.slurm --mock --nwalkers 512 --nsteps 20000 --outdir my_first_run
```

Many clusters require each GPU job to take a minimum number of CPUs per GPU
and reject a smaller request at submission; your cluster's documentation says
how many.

---

## 11. When something goes wrong

| message | what happened |
|---|---|
| `ModuleNotFoundError: No module named 'torch'` | you did not `source env.sh` in this terminal (section 3) |
| `ModuleNotFoundError: No module named 'ultranest'` | a different Python environment is switched on; run `source env.sh` again (section 3) |
| `FileNotFoundError` mentioning `emulators/` | the emulator files are missing or in the wrong place (section 2) |
| `FileNotFoundError: ... No such file or directory` | a path in your command does not exist from where the program runs; use full `~/...` paths for your data (section 7) |
| `submit from the repository folder` | `sbatch` was run from another folder; `cd` into the repository folder first |
| `Requested CPU/GPU ratio ... required ratio` | the cluster wants more CPUs per GPU; raise `--cpus-per-task` (section 10) |
| `CUDA ... no NVIDIA driver` | running without a GPU: on the login node, or in a job that did not ask for one |
| `filter directory ... does not exist` | photometry files not found; pass `--filter-dir` or use `--no-magnitude` |
| `no line window overlaps the observed wavelength grid` | your wavelengths do not reach any modelled line |
| job disappears from `squeue` with no results | it probably hit the time limit; rerun with `--resume` (section 9) |
| `--nwalkers must be even` / `at least 30` | emcee splits the ensemble in half |

To see what a failed job said, look in `logs/`. The newest file is the one you
want:

```bash
ls -lt logs/ | head
cat logs/sampler-{jobnumber}.out
```

---

## Glossary

**terminal / shell**: a window where you type commands. The `$` is the
prompt, waiting for input.

**`~` (home folder)**: your own folder on the cluster, `/home/<your username>`.
The terminal replaces `~` with it.

**full path / relative path**: a full path says where a file is from the top
(`~/data/star.fits`, `/home/...`). A relative path (`results/star.npz`) is read
starting from the folder you are currently in, so it only works from the right
folder.

**repository folder**: the folder `git clone` made, holding this code
(section 2).

**login node**: the shared machine you land on when you connect. For editing
and submitting only; never run computations here.

**compute node**: a machine that does the actual work. You reach one through
SLURM, never directly.

**SLURM**: the cluster's queueing system. You describe what you need, it
finds you a machine when one is free. `sbatch` submits, `squeue` lists,
`scancel` cancels.

**job**: one submitted piece of work, identified by a number.

**partition**: a named group of machines, usually differing in which GPU they
hold.

**virtual environment (venv)**: a private folder holding its own Python and
packages, so a project does not depend on what is installed system-wide.
`source env.sh` switches it on.

**GPU**: a processor with thousands of simple cores that all perform the same
operation at once. Excellent for evaluating many parameter sets
simultaneously, wasted on one.

**batch**: a stack of parameter sets evaluated together in one go. The reason
more walkers are nearly free.

**walkers**: the parallel chains of an ensemble MCMC run, which move through
parameter space together.

**live points**: the set of points nested sampling keeps spread over the part
of parameter space still in play (section 10).

**evidence (log Z)**: how well a model explains the data overall, averaged
over its prior. Only nested sampling computes it; used to compare two models
of the same data.

**HDF5 (`.h5`)**: a file format for large scientific arrays that can be
appended to while a program runs. Used here so a fit can be resumed.

**npz**: NumPy's format for saving several named arrays in one file. Read
with `np.load`.

**log file**: everything a program prints. For a submitted job this goes to
a file in `logs/` rather than your screen.
