"""Import the already-exported Robot12 actor and audit its saved training contract.

Usage: python scripts/import_policy.py --run ../robot-lab/logs/rsl_rl/robot12/RUN
Does not unpickle checkpoints, import Isaac Lab, or execute YAML Python tags.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np
import onnx
import yaml

from dog3588.policy import TERM_DIMS, TERRAIN_TERM_DIMS, OnnxPolicy, validate_height_contract, load_manifest


class DataOnlyLoader(yaml.SafeLoader):
    pass


def python_tag_as_data(loader, _tag, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    return loader.construct_scalar(node)


DataOnlyLoader.add_multi_constructor("tag:yaml.org,2002:python/", python_tag_as_data)


def resolve_defaults(patterns, names):
    result = []
    for name in names:
        matches = [value for pattern, value in patterns.items() if re.fullmatch(pattern, name)]
        if len(matches) != 1:
            raise ValueError(f"default for {name} is ambiguous or absent")
        result.append(float(matches[0]))
    if not np.isfinite(result).all():
        raise ValueError("nonfinite joint defaults")
    return result


def build_manifest(env, model_path, height_contract=None):
    group = env["observations"]["policy"]
    if (group["history_length"] != 5 or not group["flatten_history_dim"]
            or not group["concatenate_terms"] or group["concatenate_dim"] != -1):
        raise ValueError("unsupported actor history/concatenation settings")
    metadata_keys = {"concatenate_terms", "concatenate_dim", "enable_corruption", "history_length", "flatten_history_dim"}
    names = [k for k, v in group.items() if k not in metadata_keys and v is not None]
    dimensions = TERRAIN_TERM_DIMS if height_contract is not None else TERM_DIMS
    if names != list(dimensions):
        raise ValueError(f"unsupported actor terms/order: {names}")
    if height_contract is not None:
        validate_height_contract(height_contract.get("height_scan_contract"))
    funcs = {"base_ang_vel": "base_ang_vel", "projected_gravity": "projected_gravity",
             "velocity_commands": "generated_commands", "joint_pos": "joint_pos_rel",
             "joint_vel": "joint_vel_rel", "actions": "last_action",
             "gait_parameters": "generated_commands", "gait_clock": "gait_clock"}
    terms = []
    for name in names:
        term = group[name]
        module = "robot_lab.tasks.velocity.mdp.gait" if name == "gait_clock" else "isaaclab.envs.mdp.observations"
        expected_function = (height_contract["observation_function"] if name == "height_scan"
                             else f"{module}:{funcs[name]}")
        if term["func"] != expected_function or term.get("modifiers"):
            raise ValueError(f"unsupported transform/function for {name}")
        if name == "height_scan" and term["params"] != height_contract["observation_params"]:
            raise ValueError("height_scan parameters differ from exported spatial contract")
        if name in ("velocity_commands", "gait_parameters"):
            expected = "base_velocity" if name == "velocity_commands" else "gait"
            if term["params"] != {"command_name": expected}:
                raise ValueError(f"unexpected command source: {name}")
        scale = term["scale"] if term["scale"] is not None else 1.0
        a = np.asarray(scale, dtype=float)
        if a.shape not in ((), (dimensions[name],)) or not np.isfinite(a).all():
            raise ValueError(f"unsupported scale for {name}")
        terms.append({"name": name, "size": dimensions[name], "scale": scale, "clip": term["clip"]})
    actions = env["actions"]
    if list(actions) != ["joint_pos"]:
        raise ValueError("expected one joint-position action term")
    action = actions["joint_pos"]
    if (not action["preserve_order"] or not action["use_default_offset"] or action["clip"] is not None
            or action["class_type"] != "isaaclab.envs.mdp.actions.joint_actions:JointPositionAction"):
        raise ValueError("unsupported action contract")
    joint_names = action["joint_names"]
    if len(joint_names) != 12:
        raise ValueError("expected twelve motor joints")
    for name in ("joint_pos", "joint_vel"):
        asset = group[name]["params"]["asset_cfg"]
        if not asset["preserve_order"] or asset["joint_names"] != joint_names:
            raise ValueError("actor joint observation/action orders differ")
    action_scale = action["scale"]
    if not isinstance(action_scale, (int, float)) or not np.isfinite(action_scale) or action_scale <= 0:
        raise ValueError("unsupported action scale")
    gait = env["commands"]["gait"]
    gait_fields = ("frequency_hz", "stance_fraction", "clearance_m", "phase_offsets", "yaw_threshold_rad_s")
    dt = env["sim"]["dt"] * env["decimation"]
    if not np.isclose(dt, 0.02):
        raise ValueError("this deployment profile is for a 50 Hz policy")
    manifest = {"schema_version": 1, "file": "policy.onnx", "sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "robot": "Robot12", "control_hz": 1 / dt,
                "history_layout": "term_major_oldest_first", "history_length": 5,
                "history_reset": "repeat_first_sample", "observation_size": 5 * sum(dimensions.values()), "action_size": 12,
                "terms": terms, "joint_names": joint_names, "action_scale": action_scale,
                "default_joint_pos": resolve_defaults(env["scene"]["robot"]["init_state"]["joint_pos"], joint_names),
                "default_joint_vel": resolve_defaults(env["scene"]["robot"]["init_state"]["joint_vel"], joint_names),
                "gait": {k: gait[k] for k in gait_fields},
                "velocity_ranges": {k: env["commands"]["base_velocity"]["ranges"][k]
                                    for k in ("lin_vel_x", "lin_vel_y", "ang_vel_z")},
                "normalization": "exported_graph_only", "deployment_noise": False}
    if height_contract is not None:
        manifest["height_scan_contract"] = height_contract["height_scan_contract"]
    OnnxPolicy({**manifest, "model_path": str(model_path)})
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("policies/robot12"))
    parser.add_argument("--height-contract", type=Path,
                        help="training-exported spatial contract and exact height observation function/params")
    args = parser.parse_args()
    run = args.run.resolve()
    source = run / "exported/policy.onnx"
    config_path = run / "params/env.yaml"
    env = yaml.load(config_path.read_text(encoding="utf-8"), Loader=DataOnlyLoader)
    agent_path = run / "params/agent.yaml"
    agent = yaml.load(agent_path.read_text(encoding="utf-8"), Loader=DataOnlyLoader)
    if agent.get("clip_actions") is not None:
        raise ValueError("runner action clipping is not supported by this adapter")
    repo = run.parents[3]
    try:
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    provenance = {"run": run.name, "repository_commit_at_import": commit,
                            "source_graph_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                            "env_yaml_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                            "checkpoint": "unknown: exported ONNX contains no checkpoint metadata",
                            "terrain_type": env["scene"]["terrain"]["terrain_type"],
                            "actuator": "MotorPDActuator: simulated torque PD; real LX824 uses internal position servo"}
    # Validate in a staging directory first: an incompatible import must not
    # replace the currently working model. Stop the controller before updating.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="dog3588-import-", dir=args.output.parent) as stage_name:
        stage = Path(stage_name)
        # Recent PyTorch exports keep weights in policy.onnx.data. Embed them.
        model = onnx.load(str(source), load_external_data=True)
        onnx.external_data_helper.convert_model_from_external_data(model)
        onnx.checker.check_model(model)
        onnx.save_model(model, str(stage / "policy.onnx"), save_as_external_data=False)
        height_contract = json.loads(args.height_contract.read_text(encoding="utf-8")) if args.height_contract else None
        manifest = build_manifest(env, stage / "policy.onnx", height_contract)
        manifest["training"] = provenance
        shutil.copy2(config_path, stage / "training_env.yaml")
        shutil.copy2(agent_path, stage / "training_agent.yaml")
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        load_manifest(stage / "manifest.json")
        args.output.mkdir(parents=True, exist_ok=True)
        for filename in ("policy.onnx", "training_env.yaml", "training_agent.yaml", "manifest.json"):
            (stage / filename).replace(args.output / filename)
    print(json.dumps({"output": str(args.output), "sha256": manifest["sha256"],
                      "input": [1, manifest["observation_size"]], "output_shape": [1, 12]}, ensure_ascii=False))


if __name__ == "__main__":
    main()

