"""Strict ONNX contract. No hardware, camera or simulator dependencies."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np

TERM_DIMS = {"base_ang_vel": 3, "projected_gravity": 3, "velocity_commands": 3,
             "joint_pos": 12, "joint_vel": 12, "actions": 12,
             "gait_parameters": 7, "gait_clock": 4}
TERRAIN_TERM_DIMS = {**TERM_DIMS, "height_scan": 8}


def validate_height_contract(contract: dict) -> None:
    """Require the spatial meaning exported by training, not merely 320 floats."""
    expected = {"frame": "leveled_body", "units": "m", "reference": "body_origin",
                "invalid_policy": "stop"}
    if not isinstance(contract, dict) or any(contract.get(k) != v for k, v in expected.items()):
        raise ValueError("unsupported/missing height_scan_contract")
    xy = np.asarray(contract.get("sample_xy_m"), dtype=float)
    if (xy.shape != (8, 2) or not np.isfinite(xy).all()
            or len(np.unique(xy, axis=0)) != 8):
        raise ValueError("height_scan_contract requires eight distinct sample_xy_m")


def load_manifest(path: str | Path) -> dict:
    path = Path(path).resolve()
    m = json.loads(path.read_text(encoding="utf-8"))
    if m.get("schema_version") != 1 or m.get("history_layout") != "term_major_oldest_first":
        raise ValueError("unsupported policy schema/history layout")
    names = [t["name"] for t in m["terms"]]
    dimensions = TERRAIN_TERM_DIMS if names == list(TERRAIN_TERM_DIMS) else TERM_DIMS
    if names != list(dimensions):
        raise ValueError("unsupported observation terms/order")
    expected_size = 5 * sum(dimensions.values())
    if m["history_length"] != 5 or m["observation_size"] != expected_size or m["action_size"] != 12:
        raise ValueError("Robot12 requires 5-frame 280/320 -> 12; 288 is not supported")
    if "height_scan" in dimensions:
        validate_height_contract(m.get("height_scan_contract"))
    elif "height_scan_contract" in m:
        raise ValueError("blind policy cannot declare a height_scan contract")
    if len(m["joint_names"]) != 12 or len(set(m["joint_names"])) != 12:
        raise ValueError("expected 12 distinct motor joints")
    for t in m["terms"]:
        if t["size"] != dimensions[t["name"]]:
            raise ValueError("incorrect observation term dimension")
        scale = np.asarray(t["scale"], dtype=float)
        if scale.shape not in ((), (t["size"],)) or not np.isfinite(scale).all():
            raise ValueError("invalid observation scale")
        if t["clip"] is not None:
            clip = np.asarray(t["clip"], dtype=float)
            if clip.shape != (2,) or not np.isfinite(clip).all() or clip[0] > clip[1]:
                raise ValueError("invalid observation clipping")
    for field in ("default_joint_pos", "default_joint_vel"):
        a = np.asarray(m[field], dtype=float)
        if a.shape != (12,) or not np.isfinite(a).all():
            raise ValueError(f"invalid {field}")
    if not np.isfinite(m["control_hz"]) or m["control_hz"] <= 0:
        raise ValueError("invalid policy frequency")
    if not np.isclose(m["control_hz"], 50) or m.get("history_reset") != "repeat_first_sample":
        raise ValueError("unsupported control rate or history reset")
    scale = np.asarray(m["action_scale"], dtype=float)
    if scale.shape != () or not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid action scale")
    for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
        bounds = np.asarray(m["velocity_ranges"][key], dtype=float)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] > bounds[1]:
            raise ValueError("invalid command envelope")
    m["model_path"] = str((path.parent / m["file"]).resolve())
    digest = hashlib.sha256(Path(m["model_path"]).read_bytes()).hexdigest()
    if digest != m["sha256"]:
        raise ValueError("ONNX checksum differs from manifest; import the policy again")
    return m


class OnnxPolicy:
    def __init__(self, manifest: dict):
        import onnxruntime as ort

        self.observation_size = manifest["observation_size"]
        ort.disable_telemetry_events()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(manifest["model_path"], sess_options=options,
                                            providers=["CPUExecutionProvider"])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or len(outputs) != 1 or inputs[0].shape != [1, self.observation_size]
                or outputs[0].shape != [1, 12] or inputs[0].type != "tensor(float)"
                or outputs[0].type != "tensor(float)"):
            raise ValueError(f"ONNX tensors must be float32 [1,{self.observation_size}] -> [1,12]")
        self.input_name, self.output_name = inputs[0].name, outputs[0].name

    def infer(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        if observation.shape != (self.observation_size,) or not np.isfinite(observation).all():
            raise ValueError("invalid policy observation")
        out = self.session.run([self.output_name], {self.input_name: observation[None]})[0]
        if out.shape != (1, 12) or not np.isfinite(out).all():
            raise ValueError("invalid policy action")
        return out[0].copy()

    def benchmark(self, count: int = 1000) -> dict:
        if count <= 0:
            raise ValueError("benchmark count must be positive")
        obs = np.zeros(self.observation_size, dtype=np.float32)
        for _ in range(20):
            self.infer(obs)
        elapsed = []
        for _ in range(count):
            start = time.perf_counter()
            self.infer(obs)
            elapsed.append((time.perf_counter() - start) * 1000)
        return {"samples": count, "mean_ms": float(np.mean(elapsed)),
                "p95_ms": float(np.percentile(elapsed, 95)), "max_ms": max(elapsed),
                "providers": self.session.get_providers()}

