import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from dog3588.policy import OnnxPolicy, load_manifest


def test_real_export_inference_and_no_external_weights(manifest):
    import onnx

    graph = onnx.load(manifest["model_path"], load_external_data=False)
    assert not any(x.external_data for x in graph.graph.initializer)
    policy = OnnxPolicy(manifest)
    action = policy.infer(np.zeros(280, dtype=np.float32))
    assert action.shape == (12,) and np.isfinite(action).all()
    with pytest.raises(ValueError): policy.infer(np.zeros(56))


def test_changed_model_fails_hash_check(manifest, cfg, tmp_path):
    source_manifest = Path(cfg["policy_manifest"])
    m = json.loads(source_manifest.read_text(encoding="utf-8"))
    shutil.copyfile(manifest["model_path"], tmp_path / "policy.onnx")
    with (tmp_path / "policy.onnx").open("ab") as file:
        file.write(b"corrupt")
    (tmp_path / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        load_manifest(tmp_path / "manifest.json")

