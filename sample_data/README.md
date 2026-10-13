# Dataset policy

No vendor A-share dataset ships in this public repository. Demo data are generated using `python run_auto_alpha.py --demo ...` or `python -m tools.generate_sample_data` and contain only randomized synthetic prices. Both local demo and licensed data CSVs are ignored by `.gitignore`.

For real data, check point-in-time availability, corporate-action adjustment, symbol universe consistency, units for `volume`/`amount`, licensing and authority to publish. Do **not** upload Tushare/downloaded market data without permission.
