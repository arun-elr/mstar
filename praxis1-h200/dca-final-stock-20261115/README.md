# Final-head Praxis DROID stock-client check

M* source: `dca4991c1e3f5a859c76183ed4990f571ac1279b`. Stock Positronic source: `62b6fbd54e1786c938ec3e457581ca902bb3b2aa` (0.4.6). One H200, same pinned DROID checkpoint and source-matched Rust extension/server. The run used synthetic observations and ended with the VM stopped.

Both Python and Rust frontends passed cold, warm, reset and close through the stock Positronic v2 client. Each returned 21, 24, then 21 actuator-command rows with `robot_command` and `target_grip`. Each also returned a finite 21-row cold openpi action. These are single-session functional checks, not latency benchmarks.

The `model-cache-decision.json` file records why the prior complete 39-file SHA pass could be reused. `source-match.log` records the current Rust extension/server check. Per-frontend raw client logs and served configs are included. The paired 56-scene and session-latency results were measured at earlier source `241e8cedb175adc2b0117a4cbde8522d795b4e03` in `../241e-current-paired-20261115/`; the later merge did not change Praxis runtime files.
