# Upload to GitHub from VS Code Terminal (PowerShell)

## Repository details

- **Repository name**: `auto-alpha-research`
- **Description**: `Automated alpha factor discovery with LSTM/Transformer generators, REINFORCE, IC/ICIR screening, and walk-forward validation.`
- **Suggested Topics**: `quantitative-finance`, `alpha-factors`, `machine-learning`, `pytorch`, `reinforcement-learning`, `time-series`, `factor-investing`, `algorithmic-trading`
- Visibility: **Private first**, switch to Public **after** checking code/data publication rights and revoking the old credential.
- When creating the empty remote repository, **do not** pre-create README, `.gitignore` or license; they are already in the local package.

## PowerShell commands

In VS Code use `File → Open Folder` and select the **extracted clean** `auto-alpha-research/` directory:

```powershell
git init
git branch -M main
git add .
git status --short
# IMPORTANT: inspect staged list and ensure NO secrets or market CSVs
git commit -m "Initial release: automated alpha research framework"
git remote add origin https://github.com/YOUR_USERNAME/auto-alpha-research.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username. Authenticate in the browser/credential manager when prompted.

If the local directory is **already** a Git repository: skip `git init` and inspect `git remote -v` before adding `origin`; when origin already exists use `git remote set-url origin <URL>` only if you intend to change it. Do not use `git push --force`.

For later updates:

```powershell
git add .
git status --short
git commit -m "Improve research methodology and tests"
git push
```

Check that `README.md` is at the repository root, the Actions workflow runs, and `artifacts/` / `sample_data/*.csv` are absent from the remote files.
