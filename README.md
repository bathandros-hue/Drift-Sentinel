# Drift Sentinel

Explainable quality checks comparing baseline and current tabular data. Reports feature-level Population Stability Index (PSI), Jensen–Shannon divergence, missingness changes, unseen categories, numeric parse failures, and alert reasons.

## Quickstart

Python 3.10+, no runtime dependencies:

    python -m unittest discover -s tests -v
    python -m driftsentinel.cli --baseline examples/baseline.csv --current examples/current.csv

Optional --fail-on-drift exits with status 2 when any feature is flagged.

## Methodology

- Infer numeric vs categorical types from valid baseline observations, never from the shifted current dataset.
- Numeric baseline quantile cut points plus tail boundaries remain fixed for both datasets.
- Constant-valued baselines get a narrow equality band to reveal changes.
- Categorical features include an explicit new-category bucket.
- Empty/NA/NULL values are tracked independently of the distribution.
- PSI and JS divergence are computed over smoothed empirical bin proportions.
- Report contains raw histogram counts, bin labels, and reasons for all flags.

The PSI=0.2 and missingness=0.1 defaults are practical **heuristics**, not validated universal statistical significance thresholds.

## Limitations

Univariate drift only; no covariance or causal detection, multiple-testing adjustment, temporal drift modeling, feature importance, or model-performance monitoring. Small samples may produce volatile PSI, especially for rare new categories; review individual histograms before acting.

## GitHub topics

data-drift, mlops, monitoring, data-quality, explainable-ai, python

## License

MIT; see LICENSE.
