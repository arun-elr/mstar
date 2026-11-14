"""Cold-only simulation adapter for M*'s openpi DROID bring-up mode.

This benchmark adapter translates the sim pose into openpi's FR3 hand state,
then composes the returned relative actions with the pinned reference pose
conversion. It is not the stateful Positronic serving path.
"""

from __future__ import annotations

import time

import numpy as np
from websockets.sync.client import connect

from praxis import single_arm
from praxis.serve import wire
from praxis.serve.sim_client import RemoteOutput

FIRST_ROWS = 21
SEED_STRIDE = 1000


def openpi_observation(
    history: object, prompt: str, seed: int
) -> tuple[dict, np.ndarray]:
    """Return one openpi observation and the same pose in the model frame."""
    latest = history.latest
    pose8_sim = np.concatenate((latest.ee_pose, [latest.grip])).astype(np.float32)
    pose8_model = single_arm.pose8_to_model(pose8_sim, single_arm.TURNED_FLANGE)
    model_rot = single_arm.quat_xyzw_to_matrix(pose8_model[[4, 5, 6, 3]])
    hand_xyz, hand_rot = single_arm.pose_to_rig(
        pose8_model[:3].astype(np.float64), model_rot, single_arm.FR3_HAND
    )
    hand_xyzw = single_arm.matrix_to_quat_xyzw(hand_rot)
    hand_pose = np.concatenate((hand_xyz, hand_xyzw[[3, 0, 1, 2]])).astype(np.float32)
    return {
        "observation/exterior_image_1_left": np.asarray(
            latest.exterior, dtype=np.uint8
        ),
        "observation/wrist_image_left": np.asarray(latest.wrist, dtype=np.uint8),
        "observation/cartesian_position": hand_pose,
        "observation/gripper_position": np.asarray([latest.grip], dtype=np.float32),
        "prompt": prompt,
        "seed": seed,
    }, pose8_model


class MStarPolicy:
    """One root WebSocket per episode, always 21 cold relative rows per call."""

    name = "mstar_openpi_cold"
    decode_video = False

    def __init__(self, url: str, prompt: str = "", timeout: float = 300.0) -> None:
        if not url.startswith(("ws://", "wss://")):
            raise ValueError("M* openpi URL must use ws:// or wss://")
        self.url = url
        self.prompt = prompt
        self.timeout = timeout
        self._socket = None
        self.meta: dict = {}
        self.calls = 0
        self.scene_seed = 0
        self.sampler_seed = 0

    def reset(self, env: object, seed: int) -> None:
        self.close()
        self.calls = 0
        self.scene_seed = int(seed)
        try:
            self._socket = connect(
                self.url, open_timeout=self.timeout, max_size=None, compression=None
            )
            raw = self._socket.recv(timeout=self.timeout)
            if not isinstance(raw, bytes):
                raise ValueError("M* openpi did not send binary metadata")
            meta = wire.deserialise(raw)
            if not isinstance(meta, dict):
                raise ValueError("M* openpi did not send metadata")
            self.meta = meta
        except Exception:
            self.close()
            raise

    def act(self, history: object) -> RemoteOutput:
        if self._socket is None:
            raise RuntimeError("call reset before act")
        seed = self.sampler_seed + SEED_STRIDE * self.scene_seed + self.calls
        observation, pose8_model = openpi_observation(history, self.prompt, seed)
        start = time.monotonic()
        try:
            self._socket.send(wire.serialise(observation))
            raw = self._socket.recv(timeout=self.timeout)
            if not isinstance(raw, bytes):
                raise RuntimeError(f"M* openpi call failed: {raw}")
            answer = wire.deserialise(raw)
            if not isinstance(answer, dict):
                raise ValueError("M* openpi returned a non-map answer")
            relative = answer.get("actions")
            if not isinstance(relative, np.ndarray) or relative.shape != (
                FIRST_ROWS,
                10,
            ):
                raise ValueError("M* openpi must return a 21x10 relative action array")
            if not np.isfinite(relative).all():
                raise ValueError("M* openpi returned non-finite action values")
            commands = single_arm.rel10_to_wire(
                relative,
                pose8_model,
                frame=single_arm.TURNED_FLANGE,
                mode=single_arm.control_mode_payload("droid_impedance"),
            )
        except Exception:
            self.close()
            raise
        self.calls += 1
        return RemoteOutput(
            commands,
            info={"step_s": time.monotonic() - start, "depth": 0, "seed": seed},
        )

    def close(self) -> None:
        socket = self._socket
        self._socket = None
        if socket is not None:
            socket.close()
