# Emulators

The fits use the FASTWIND line and continuum emulators of Vasilis Kolympiris
([FASTWIND_emulator](https://github.com/API-AXIOM/FASTWIND_emulator); please
cite his MSc thesis, *Exploring the Composition and Evolution of Massive Stars:
Predicting stellar spectra of O-type stars using neural networks*). This folder
holds the modules that load them, `line_emulators_13par/line_emulator.py` and
`fluxcont/fluxcont_emulator.py`, and the filter curves used for the photometry.

The trained network files are not part of this repository. They are available
from the emulators' author on request. Put them here:

```
emulators/
├── line_emulators_13par/
│   ├── line_emulator.py
│   └── models/
│       ├── emulator_OUT.CIII4070_VTV010.zst.pth
│       ├── ...                                  # one file per line window, 37 in all
│       └── emulator_OUT.SiIV1400_VTV010.zst.pth
├── fluxcont/
│   ├── fluxcont_emulator.py
│   └── model/
│       └── emulator_FLUXCONT.pth
└── filter_transmissions/                        # included
```

The 37 line windows are CIII4070, CIII5696, CIIINIIIcombined, CIV1169b,
CIV1550, CIV5801, HALPHAHEII6527, HBETA, HDELTA, HEI4026, HEI4387, HEI4471,
HEI4922, HEI5875, HEI7065, HEII1640, HEII4200, HEII4541, HEII4686, HEII5411,
HEII6406, HEII6683, HEPS, HGAMMA, NIII1750, NIV1718, NIV3480, NIV4058,
NIV6380, NV1240, NV4603, OIII5592, OIV1340, OIV1371, PV1118, SiIII4552 and
SiIV1400. A fit can use any subset of them (`--lines`), but every file in
`models/` is loaded at start-up.

To check that the files are in place, run the two loading modules directly:

```bash
python emulators/line_emulators_13par/line_emulator.py
python emulators/fluxcont/fluxcont_emulator.py
```

The first loads all 37 line emulators, lists their names and predicts the
Hα window for an example star; the second predicts the continuum for the
same star. Each prints a short summary, or an error if a file is missing.

To keep the files somewhere else, lay that folder out the same way and point
`EMULATOR_ROOT` at it before running, for example
`export EMULATOR_ROOT=/data/fastwind_emulators`. The filter curves in
`filter_transmissions/` move with it.

## What the emulators predict

Each line emulator predicts the continuum-normalised FASTWIND profile of one
line window, and the continuum emulator the spectral energy distribution
between 1000 and 99,840 Å, both as a function of 13 parameters in this order:

    teff, logg, radius, mdot, yhe, C, N, O, beta, vinf, fic, fvel, fclump

A fit does not use each line emulator's full wavelength range, but the
narrower window set for that line in `LINE_WINDOW_OVERRIDES`
(`src/gpu_config.py`). Hβ, for example, is predicted over 4770–4965 Å but
fitted over 4845.6–4877.5 Å.

The loading modules' own `predict` methods take `mdot` in linear units
(M☉/yr). The fitting code samples log10 `mdot` and passes it to the networks
directly, through the log-space normalisation stored in each checkpoint
(`src/gpu_emulator.py`).

## Filter curves

`filter_transmissions/` holds one `wavelength transmission` table per filter
and `zero_points.dat` with each filter's Vega, AB and ST zero points. The
curves and zero points are taken from Kiwi-GA (Brands et al. 2022, A&A 663,
A36). Four of the curves originally come from the SVO Filter Profile Service
(http://svo2.cab.inta-csic.es/theory/fps/); the SPHERE/IRDIS Ks curve comes
from the SPHERE instrument team. The filters available to `--filter-name` are
listed in the main README.
