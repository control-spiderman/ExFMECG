# Public ECG example

E07303 is a 10-second, 500-Hz, 12-lead ECG from the Georgia 12-lead ECG
Challenge Database distributed with the PhysioNet/Computing in Cardiology
Challenge 2020, version 1.0.2. The public header reports age 81 years, female sex
and SNOMED CT diagnosis 164889003 (atrial fibrillation).

`E07303.mat` and `E07303.hea` are unmodified source files. The upstream dataset
is licensed under the Creative Commons Attribution 4.0 International license:
https://physionet.org/content/challenge-2020/view-license/1.0.2/

`E07303.npy` is derived from the source waveform by converting the header gain
of 1,000 ADC units per mV and resampling the 5,000 source samples to the 1,024
model input points. Recreate it from the source files with:

```bash
python examples/prepare_georgia_example.py
```

Please cite:

Perez Alday, E. A. et al. Classification of 12-lead ECGs: The
PhysioNet/Computing in Cardiology Challenge 2020 (version 1.0.2). PhysioNet
(2022). https://doi.org/10.13026/dvyd-kd57
