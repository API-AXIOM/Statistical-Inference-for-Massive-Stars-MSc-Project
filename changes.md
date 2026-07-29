# Changes

## Bug fixes

### `KIWI-GA/example_input/line_list.txt` + `KIWI-GA/input/automated2/line_list.txt`
Commented out the `NIV3480` line (wavelength range 3476–3489 Å). The NN model only covers 3915–7170 Å, so this line produced an empty `spect_wl` array inside `NN_wrapper_Hhe_split.broaden_fwline`, causing:
```
IndexError: index 0 is out of bounds for axis 0 with size 0
```
The fix was applied to both the template (`example_input/`) and the already-copied `automated2` input directory.

---

### `KIWI-GA/population_Hhe_split.py` and `KIWI-GA/population.py`
`Gamma_Edd_check` hardcoded `yhe = 0.08` regardless of the model being evaluated. When `yhe` is a free parameter its actual sampled value was never used in the Eddington limit check, and the print output always showed `yhe = 0.08`.

**Fix:** read `yhe` from the model parameter vector when `'yhe'` is in `param_names`; fall back to `0.08` only when it is fixed.

```python
# Before
yhe = 0.08

# After
if 'yhe' in param_names:
    yhe = float(model[param_names.index('yhe')])
else:
    yhe = 0.08
```

---

### `ga_notebook_tools.py`
`compute_obs_flux` and `flux_to_magnitude` used module-level variables `_KS_WAVE`, `_KS_TRANS`, and `zpflux` that were never initialised in the module — they only existed in the notebook's setup cell. This caused:
```
NameError: name '_KS_WAVE' is not defined
```
**Fix:** added Ks filter transmission loading at module level using `ppp.filter_path`.

---

### `Comparisons_with_KIWI-GA.ipynb` — cell 8
`run_ga_coverage_test` defaulted to the GA's 5-param `PARAM_NAMES` (`teff, logg, mdot, yhe, vrot` — no `radius`), producing a 5-element `theta_k`. Both `simulate_model_spectrum` and `simulate_kband_magnitude` unpack a 6-element theta expecting `radius` at index 2, causing:
```
ValueError: too many values to unpack (expected 6)
```
**Fix:** pass `param_names=["teff", "logg", "radius", "mdot", "yhe", "vrot"]` and a matching `prior` (including `radius: (6, 21)`) explicitly to `run_ga_coverage_test`. The GA coverage check skips `radius` automatically since it is not in the GA output.

---

## Code quality / cleanup (`MCMC_with_calc_kband.ipynb`)

### Cell 2 — hardcoded paths
Replaced three hardcoded path strings with `ppp.*` equivalents for consistency with `Comparisons_with_KIWI-GA.ipynb`:

| Before | After |
|---|---|
| `filterdir = "filter_transmissions/"` | `filterdir = ppp.filter_path` |
| `keras.saving.load_model('NN_model.keras')` | `keras.saving.load_model(ppp.keras_model_path)` |
| `open('normalisation.json')` | `open(ppp.norm_path)` |

### Cell 3 — new broadening functions

| Issue | Fix |
|---|---|
| `cc = 2.99792458e10` redefined locally inside `planck_wavelength`, shadowing the module-level constant | Deleted the local assignment; function now uses module-level `cc` |
| Dead `fftconvolve(flux_uniform, kernel, mode="same")` call in `vspace2` — result immediately overwritten by the padded version two lines below | Deleted the dead call |
| `return_wave_space` parameter name in `broaden_velocity_space` inconsistent with `output_wl` used everywhere else in the codebase | Renamed to `output_wl` throughout (signature, docstring, call site in `simulate_model_spectrum`) |
| `interp1d(..., kind='quadratic')` in `resample_to_vel` and `resample_to_wave` — rebuilds interpolation objects on every MCMC step; no accuracy gain over linear on a 0.2 Å grid | Replaced with `np.interp` |
| Verbose NumPy-style docstrings on 7 new functions, inconsistent with the one-line style used elsewhere | Trimmed to single-line docstrings |
| Inline comments restating what the adjacent code already says | Removed (~15 comments across the new functions) |
