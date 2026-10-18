"""Strict, deterministic replay of recorded or explicitly synthetic mission facts."""

from dataclasses import asdict
import json
from pathlib import Path

from .mission import MissionFSM, MissionInput, MissionState


def replay_mission(cfg, path, *, mission_id='replay', output=None, require_complete=False):
    fsm = MissionFSM(cfg['mission'], mission_id=mission_id)
    previous = None
    transitions, requests, count = [], [], 0
    stream = Path(output).open('w', encoding='utf-8') if output else None
    try:
        with Path(path).open(encoding='utf-8') as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    values = json.loads(line)
                    sample = MissionInput(**values)
                    if previous is not None and sample.now <= previous:
                        raise ValueError('replay timestamps must strictly increase')
                    previous = sample.now
                    result = fsm.step(sample)
                except (ValueError, TypeError) as exc:
                    raise ValueError(f'{path}:{line_number}: {exc}') from exc
                count += 1
                if not transitions or transitions[-1]['state'] != result.state.value:
                    transitions.append({'t': sample.now, 'state': result.state.value, 'reason': result.reason})
                if result.delivery_request:
                    requests.append(asdict(result.delivery_request))
                if stream:
                    stream.write(json.dumps({'t': sample.now, **asdict(result)}, allow_nan=False) + '\n')
    finally:
        if stream:
            stream.close()
    if not count:
        raise ValueError('mission replay is empty')
    result = {'mode': 'offline_perception_replay', 'samples': count, 'state': fsm.state.value,
              'reason': fsm.reason, 'obstacles_completed': fsm.obstacles_completed,
              'delivery_completed': fsm.delivery_completed, 'delivery_requests': requests,
              'transitions': transitions}
    if require_complete and fsm.state != MissionState.DONE:
        raise ValueError(f"replay did not complete: {fsm.state.value}: {fsm.reason}")
    return result
