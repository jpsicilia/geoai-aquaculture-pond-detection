# Data

The challenge dataset is released by FAO/ITU under a **CC-BY-SA 4.0** licence and
is **not redistributed in this repository**.

To reproduce the results, download the three files from the challenge page and
place them in the repository root (or the directory from which you run
`pipeline.py`):

- `Train.csv` — labelled training pixels (12 months × 12 bands + `label`)
- `Test.csv` — unlabelled test pixels (different acquisition period)
- `SampleSubmission.csv` — submission template

Challenge page:
https://zindi.world/competitions/geoai-aquaculture-pond-identification-challenge/data

## Dataset at a glance
- Each row = one 10 m × 10 m ground patch.
- 144 feature columns: 12 spectral/radar bands × 12 months.
  - Optical (Sentinel-2): blue, green, red, re1, re2, re3, nir, nira, swir1, swir2
  - SAR (Sentinel-1): VV, VH
- `-9999` marks a month with no observation.
- Train: full coverage. Test: ~60% missing, seasonally structured — the source
  of the domain shift this project addresses.
