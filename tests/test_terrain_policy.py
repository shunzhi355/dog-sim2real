import copy
import hashlib
import json

import numpy as np
import pytest

from dog3588.observation import ObservationBuilder
from dog3588.policy import OnnxPolicy, load_manifest
from dog3588.types import ImuState, JointState


def terrain_manifest(manifest):
    m = copy.deepcopy(manifest)
    m['observation_size'] = 320
    m['terms'].append({'name': 'height_scan', 'size': 8, 'scale': 2., 'clip': [-.5, .5]})
    m['height_scan_contract'] = {
        'frame': 'leveled_body', 'units': 'm', 'reference': 'body_origin',
        'invalid_policy': 'stop', 'sample_xy_m': [[.1 * (i // 2 + 1), .05 * (-1 if i % 2 else 1)] for i in range(8)]}
    return m


def test_terrain_history_and_blind_rejection(manifest):
    m = terrain_manifest(manifest)
    b = ObservationBuilder(m)
    joints = JointState(np.zeros(12), np.zeros(12), np.ones(12))
    imu = ImuState(np.zeros(3), np.array([0, 0, -1]), 1.)
    gait = np.array([1.5, .575, .02, 0, .5, .5, 0])
    with pytest.raises(ValueError, match='requires valid'):
        b.build(joints, imu, np.zeros(3), gait)
    obs = b.build(joints, imu, np.zeros(3), gait, np.full(8, -.2))
    assert obs.shape == (320,)
    np.testing.assert_allclose(obs[-40:], -.4)
    obs = b.build(joints, imu, np.zeros(3), gait, np.ones(8))
    np.testing.assert_allclose(obs[-40:-8], -.4)
    np.testing.assert_allclose(obs[-8:], 1.)  # clip before scale, same as proprioception
    with pytest.raises(ValueError, match='does not accept'):
        ObservationBuilder(manifest).build(joints, imu, np.zeros(3), gait, np.zeros(8))


def test_actual_320_graph_and_contract_validation(manifest, tmp_path):
    import onnx
    from onnx import helper, TensorProto, numpy_helper

    m = terrain_manifest(manifest)
    graph = helper.make_graph(
        [helper.make_node('MatMul', ['obs', 'weight'], ['actions'])], 'synthetic-contract-test',
        [helper.make_tensor_value_info('obs', TensorProto.FLOAT, [1, 320])],
        [helper.make_tensor_value_info('actions', TensorProto.FLOAT, [1, 12])],
        [numpy_helper.from_array(np.zeros((320, 12), dtype=np.float32), 'weight')])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], ir_version=9)
    graph_path = tmp_path / 'policy.onnx'
    onnx.save(model, graph_path)
    m.update(file='policy.onnx', sha256=hashlib.sha256(graph_path.read_bytes()).hexdigest())
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(m))
    policy = OnnxPolicy(load_manifest(path))
    np.testing.assert_array_equal(policy.infer(np.zeros(320)), np.zeros(12))
    with pytest.raises(ValueError):
        policy.infer(np.zeros(280))
    m['height_scan_contract']['reference'] = 'unknown_ground'
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match='height_scan_contract'):
        load_manifest(path)
    m['observation_size'] = 288
    path.write_text(json.dumps(m))
    with pytest.raises(ValueError, match='288'):
        load_manifest(path)
