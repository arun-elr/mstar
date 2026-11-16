# #774 control socket restart probe

This uses separate Linux processes and the Rust ZeroMQ communicator. It
checks that the first worker can reach its conductor, terminates that
conductor with SIGTERM while the worker stays alive, starts a new conductor
and worker on the same base IPC prefix, then queues 20 messages from the old
worker and one from the new worker.

The first launch sets `MSTAR_DEPLOYMENT_TOKEN=first-launch`; the second sets
`second-launch`. On the PR, each token selects a subdirectory under the same
base prefix. The API server mints a fresh token when one is not supplied.
An explicit launcher token must change on each restart. This probe covers
restarts with distinct tokens.

Run it from each checkout with its own Python package on the import path:

```bash
PYTHONPATH="$PWD" python /path/to/restart_probe.py
```

The shared virtual environment has an editable M* install. `PYTHONPATH` is
required when the script lives outside the checkout; without it, both runs
can silently import the same editable M* install. The probe checks that the
loaded package is inside the current checkout and records its path and Git
head. Both runs used Rust extension version 0.1.1 from the same shared virtual
environment. The PR diff has no Rust source or Cargo changes.

| Checkout | Old worker before restart | Old messages at restarted conductor within 2 s | New worker message within 2 s |
| --- | ---: | ---: | ---: |
| main `2c8ffde8f7fd321c29aad3d1aad3e8cc8e8dc175` | 1/1 | 20/20 | 1/1 |
| PR `66f8d6092d5e70dbc5465ab520bb7272dafee6ea` | 1/1 | 0/20 | 1/1 |

`main.json` and `pr.json` are the raw outputs from clean checkouts. The probe
exercises the control mesh, not a full model server, tensor sends, or GPU
inference. The endpoint fields show that the old worker keeps aiming at the
first conductor's path; only on the PR does that path differ from the restarted
conductor's path. The existing PR regression tests cover the socket isolation
in process; this run also checks that an old worker survives the conductor exit.
