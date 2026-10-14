# Contact sheets for issue #434 (Cosmos3-Edge kv windowed drift)

Media for https://github.com/mstar-project/mstar/issues/434, not code.

- `contact_sheet.png`: the shipped kv and chained modes against a non-windowed 241-frame clip, with the model card prompt and with the robot-arm prompt from #425's stream client.
- `contact_sheet_options.png`: the same against the config-only option, chained with 61-frame windows and a 32-frame (or 16-frame) overlap.

Every tile is a raw decoded frame of a 241-frame rollout on an H200: upstream main `2beeece8`, `configs/cosmos3_edge.yaml`, seed 0, 832x480, 20 steps, i2v from the model card's example image. Cosmos3-Edge output is under the OpenMDW-1.1 license.
