"""Run an exported ONNX without devices or an inferred hardware contract.

The optional mock loop borrows preprocessing from an explicitly named config.
It proves only software execution, not that the new training contract matches.
No checkpoint is unpickled, no default model is replaced, and no hardware option
is exposed. External ONNX weights are embedded in a temporary file for checking.
"""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np
import onnx
import onnxruntime as ort

from dog3588.config import load_config
from dog3588.policy import OnnxPolicy, load_manifest
from dog3588.runtime import run


def session(path):
    ort.disable_telemetry_events()
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])


def check_export(model_path, *, count=1000, mock_config=None, seconds=5., log_path=None):
    model_path = Path(model_path).resolve()
    if count < 1 or not np.isfinite(seconds) or seconds <= 0:
        raise ValueError('count and seconds must be positive and finite')
    source = session(model_path)
    inputs, outputs = source.get_inputs(), source.get_outputs()
    if (len(inputs) != 1 or len(outputs) != 1 or inputs[0].shape not in ([1, 280], [1, 320])
            or outputs[0].shape != [1, 12] or inputs[0].type != 'tensor(float)'
            or outputs[0].type != 'tensor(float)'):
        raise ValueError('expected float32 Robot12 [1,280/320] -> [1,12] ONNX')
    size = inputs[0].shape[1]
    # Temporary embedding leaves the user's ONNX, .data and .pt untouched.
    with tempfile.TemporaryDirectory(prefix='dog3588-export-check-') as directory:
        directory = Path(directory)
        packed = directory / 'policy.onnx'
        model = onnx.load(str(model_path), load_external_data=True)
        onnx.external_data_helper.convert_model_from_external_data(model)
        onnx.checker.check_model(model)
        onnx.save_model(model, str(packed), save_as_external_data=False)
        policy = OnnxPolicy({'model_path': str(packed), 'observation_size': size})
        maximum_error = 0.
        rng = np.random.default_rng(4497)
        for _ in range(100):
            observation = rng.uniform(-1., 1., size=size).astype(np.float32)
            original = source.run([outputs[0].name], {inputs[0].name: observation[None]})[0][0]
            actual = policy.infer(observation)
            np.testing.assert_allclose(actual, original, rtol=1e-5, atol=1e-6)
            maximum_error = max(maximum_error, float(np.max(np.abs(actual - original))))
        result = {'mode': 'offline_export_check', 'source': str(model_path),
                  'source_graph_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
                  'packed_sha256': hashlib.sha256(packed.read_bytes()).hexdigest(),
                  'input': [1, size], 'output': [1, 12],
                  'contract_verified': False, 'comparison_samples': 100,
                  'packing_max_abs_error': maximum_error,
                  'zero_observation_action': policy.infer(np.zeros(size)).tolist(),
                  'benchmark': policy.benchmark(count)}
        print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
        if mock_config is not None:
            cfg = load_config(mock_config)
            reference = load_manifest(cfg['policy_manifest'])
            if reference['observation_size'] != size:
                raise ValueError('mock reference observation dimension differs from the new model')
            # These are explicitly ASSUMED preprocessing values for an offline
            # demonstration. Do not create a deployable policy from this copy.
            manifest = dict(reference, file='policy.onnx', sha256=result['packed_sha256'])
            manifest.pop('model_path', None)
            manifest['training'] = {'contract_verified': False,
                                    'note': 'temporary assumed reference for a hardware-free mock only',
                                    'reference_manifest': cfg['policy_manifest'],
                                    'source_export': str(model_path)}
            path = directory / 'manifest.json'
            path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
            cfg['policy_manifest'] = str(path)
            # Port zero prevents the mock from consuming real controller commands.
            cfg['command']['port'] = 0
            print(json.dumps({'event': 'mock_contract_assumption',
                              'reference': str(Path(mock_config).resolve()),
                              'note': 'Software smoke only: training semantics are not verified.'}), flush=True)
            result['mock'] = run(cfg, hardware=False, arm=False, autonomous=False,
                                 seconds=seconds, log_path=log_path)
            print(json.dumps(result['mock'], indent=2, allow_nan=False), flush=True)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--mock-config', type=Path,
                        help='explicitly assume this preprocessing for a no-device smoke run only')
    parser.add_argument('--seconds', type=float, default=5.)
    parser.add_argument('--log', type=Path)
    args = parser.parse_args()
    check_export(args.model, count=args.count, mock_config=args.mock_config,
                 seconds=args.seconds, log_path=args.log)


if __name__ == '__main__':
    main()
