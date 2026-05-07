# DagsHub + MLflow Setup Guide

This project already supports optional DagsHub tracking in `.github/workflows/mlops.yml`.
You only need to create a DagsHub repo and add 3 GitHub secrets.

## 1) Create DagsHub Repo

1. Log in to `https://dagshub.com` with your GitHub account.
2. Click `New Repository`.
3. Create a repo (example: `Day21-Track2-CI-CD-for-AI-Systems`).
4. Open the repo and click the `MLflow` tab.
5. Copy your MLflow Tracking URI from that page.

Example format:

```text
https://dagshub.com/<username>/<repo>.mlflow
```

## 2) Create DagsHub Access Token

1. In DagsHub, go to `Settings` -> `Access Tokens`.
2. Create a token with write access.
3. Copy token value once (you cannot view it again).

## 3) Add GitHub Secrets

In GitHub repo -> `Settings` -> `Secrets and variables` -> `Actions`, add:

- `DAGSHUB_MLFLOW_TRACKING_URI`: your URI from MLflow tab
- `DAGSHUB_USERNAME`: your DagsHub username
- `DAGSHUB_TOKEN`: your DagsHub access token

Do not add quotes or extra spaces.

## 4) Verify in Workflow

Trigger pipeline manually from Actions tab or push a change.

In the `Train` job, check step `Configure MLflow tracking`:

- If secrets are set: log should print `Using DagsHub MLflow tracking.`
- If not set: it falls back to local `sqlite:///mlflow.db`.

## 5) Verify Runs in DagsHub

After a successful `Train` job:

1. Open DagsHub repo -> `MLflow`.
2. Confirm new run appears with params/metrics (`accuracy`, `f1_score`, class metrics).
3. Open run artifacts and verify `model` artifact exists.

## Troubleshooting

- Empty DagsHub page: check `DAGSHUB_*` secrets spelling.
- Authentication error: regenerate token and update `DAGSHUB_TOKEN`.
- URI error: ensure it ends with `.mlflow`.
