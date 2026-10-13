# Publication checklist (do not skip)

- [ ] **Revoke/rotate** the Tushare API key previously embedded in `data/tushare_loader.py` in the ORIGINAL ZIP. Treat it as compromised even if the clean ZIP no longer contains the key.
- [ ] Confirm rights to publish internship-created code, research methods, formulas and figures. Private company IP may require written permission.
- [ ] Never upload the original 50 MB ZIP, cached `.pyc` files, full market CSVs or full `artifacts/` results.
- [ ] Use the **clean ZIP** only. Review output from `git status --short` before making your first commit.
- [ ] Verify `python -m pytest -q` and `python run_auto_alpha.py --demo --quick --mode random` locally on your intended environment.
- [ ] Keep `.env`, local market datasets and any API credentials out of git history (the `.gitignore` and scans reduce risk but do not replace review).
- [ ] Understand that the archived Sharpe/return statistics cannot be marketed as independent OOS or live P&L.
- [ ] Confirm `README.md` renders on GitHub and GitHub Actions passes.
- [ ] Add a reuse license only if you have legal authority to do so. Currently no license is granted by default.
