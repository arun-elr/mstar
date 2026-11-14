# Praxis DROID lift_cube trace on one H200

This chart shows one corrected `lift_cube` simulation episode at scene seed 0 and sampling seed 0. The trace set `policy.prompt = env.prompt`, matching the official simulation runner. An earlier exploratory trace omitted that assignment and is not used here.

The pinned Praxis reference succeeded at tick 108. M* `5902054c0f` failed after 165 ticks. The M* cube rose above the 5 cm threshold at ticks 163 to 165 and reached 6.56 cm, but the task requires four consecutive ticks above threshold and it had three. The first observation fingerprints matched. The first composed command batch differed, with a maximum absolute pose element difference of 0.02133, relative pose L2 of 1.56%, and maximum gripper difference of 0.00933. Cube-height traces first separated at tick 71.

The chart was generated from the corrected raw traces, which are retained separately because they include command payloads and local runtime metadata. Both runs used the same task prompt, scene and sampling seeds, and 15 Hz physics. M* reported Triton attention. The reference simulator's effective attention backend was not logged, so this is an observed closed-loop comparison, not strict backend-matched numerical parity. The M* trace used the Python frontend; the earlier seven-task guard used Rust. This chart does not identify the numerical cause of the missed task.

M* source: `5902054c0f9fa057a80ae5a7e15e67f19d4bcde2`. Reference source: `68f4d7a254b4bf41951d2e97f44b07f6cac2bf69`. Hardware: one NVIDIA H200. The serving gate remains off.
