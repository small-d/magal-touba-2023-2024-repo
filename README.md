# Magal Touba 2023–2024 — Data Engineering Pipeline

Reproducible data engineering pipeline for the exploitation of hospital records
collected at the Cheikh Ahmadoul Khadim National Hospital Center (CHAK, Touba,
Senegal) during the 2023 and 2024 editions of the Grand Magal.

Companion repository for:

> D. Diallo, A. D. Gueye, M. L. Ba, I. Ba, M. Diagne, "A Formalized and
> Reproducible Data Engineering Pipeline for the Exploitation of Hospital
> Records during Mass Gatherings: The Case of the Grand Magal of Touba
> (2023–2024)," CNRIA 2026 (Early-Stage Research track).

## Repository structure

```
.
├── PIPELINE_DE_DATA_ENGINEERING.py   # Full pipeline: τ (pivot/typing), ν (harmonization),
│                                      # δ (imputation), ρ (deduplication) + quality report
├── Analyse_Flux_Patients.py          # Exploratory analysis + figures on D_clean
├── data/
│   ├── D_clean.csv                   # Final cleaned, patient-level dataset (N = 1,904)
│   ├── mapping_icd11_snomed.csv      # Preliminary ICD-11 / SNOMED CT mapping
│   ├── table_harmonisation_semantique.csv  # Full semantic harmonization rule table
│   ├── table1_volumetrie.csv         # Row-to-patient volumetric audit trail (raw pipeline output)
│   ├── table2_qualite_donnees.csv    # Data quality metrics (Weiskopf & Weng)
│   ├── table3_sensibilite_imputation.csv   # Imputation sensitivity analysis
│   └── table8_synthese_tests_statistiques.csv  # Summary of comparative statistical tests
├── figures/
│   ├── fig1_flux_pipeline.png        # Fig. 1 of the manuscript
│   └── fig2_admissions_journalieres.png  # Fig. 2 of the manuscript
├── requirements.txt
└── README.md
```

**Note:** raw source files (`Magal_2023.xlsx`, `Magal_2024.xlsx`) are not included in this
public repository, as they contain unaggregated hospital records. They are available from
the corresponding author upon reasonable request, subject to CHAK data-sharing agreements.

## Reproducing the results

```bash
pip install -r requirements.txt
python PIPELINE_DE_DATA_ENGINEERING.py   # regenerates D_clean.csv and all table*.csv files
python Analyse_Flux_Patients.py          # regenerates the exploratory figures
```

`PIPELINE_DE_DATA_ENGINEERING.py` uses a fixed random seed (`RANDOM_STATE = 42`) and prints,
as console/log output, every patient count reported in Section IV of the manuscript
(raw EAV rows → valid EAV rows → patients after pivot → patients after deduplication),
for full auditability.

## Pipeline formalization

The pipeline is defined as Φ = ρ ∘ δ ∘ ν ∘ τ:

- **τ** — type conversion and EAV-to-patient pivoting
- **ν** — text normalization and semantic harmonization
- **δ** — missing-value imputation
- **ρ** — deduplication (phonetic + text-similarity matching, λ = 0.85)

## Citation

If you use this pipeline or dataset, please cite the CNRIA 2026 paper above
(full citation to be updated with final proceedings details upon acceptance).
See `CITATION.cff` for machine-readable citation metadata.

## License

Code: MIT License (see `LICENSE`). Data: released under CC-BY 4.0 unless otherwise noted —
see data-sharing terms in the paper's Methods section (III-B, III-G).

## Contact

Demba Diallo — demba.diallo@uadb.edu.sn
