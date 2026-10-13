"""Run bench_wan_decoder.py from the M* copy at $MSTAR, with M*-variant VAEs loaded from /vo/snap (offline)."""
import os
import sys
from pathlib import Path
sys.path.insert(0, os.environ["MSTAR"] + "/benchmark/wan_vae_decoder")
import common, bench_wan_decoder as B
common.hf_snapshot = lambda repo: Path("/vo/snap") / ("wan21" if repo == common.WAN21_REPO else "wan22")
B.load_vae.__globals__["hf_snapshot"] = common.hf_snapshot
if __name__ == "__main__":
    B.main()
