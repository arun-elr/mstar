# Remote media fetch cap, CPU frontend check

Source heads: upstream main `2c8ffde8f7` and PR #1551 `505acbf6f2`.
Python 3.12.15 on a 32 CPU host. No GPU or model forward ran.

`bench.py` sends 48 remote image URLs to the Bagel chat adapter through one
FastAPI test server. The allowlisted local HTTP server holds each accepted
fetch until a timer releases it after two seconds. Once 32 URLs reach the
upstream on main, or eight on the PR, the script sends one plain text chat
request. Each arm ran three times in a fresh server context. All 48 remote
requests returned 400; on the PR, 40 were refused at the fetch cap and eight
reached the upstream. The plain text request returned 200 on every run.

Run from each pinned source checkout with its path in `PYTHONPATH`:

```sh
MSTAR_ALLOW_REMOTE=1 MSTAR_REMOTE_ALLOWED_HOSTS=127.0.0.1 \
  PYTHONPATH=<checkout> <python-with-test-dependencies> bench.py --arm main --runs 3
```

Use `--arm pr` for the PR checkout. `main.json` and `pr.json` are the raw
outputs. `chart.py` plots them as `remote_fetch_latency.png`.

This isolates frontend contention under a synthetic slow HTTP server. It is
not an H200 serving or model performance measurement.
