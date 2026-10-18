"""Write an explicitly synthetic full-round replay. No devices or real success claims."""

import argparse
import json
from pathlib import Path

from dog3588.config import load_config
from dog3588.mission import MissionFSM, MissionInput


def generate(output, colour='yellow'):
    # Route configuration is part of this fixture, never an assumption about the venue.
    cfg = load_config(Path(__file__).resolve().parents[1] / 'config/robot-deploy.yaml')
    cfg['mission'].update(parcel_colour=colour, route_map={colour: {'red+green': 'left'}},
                          return_route_map={colour: {'green': 'right'}})
    fsm = MissionFSM(cfg['mission'], mission_id='demo')
    rows, now = [], 0.

    def emit(count=1, **extra):
        nonlocal now
        for _ in range(count):
            now = round(now + .1, 6)
            row = dict(now=now, vision_stamp=now, tof_stamp=now, tof_valid=True,
                       line=dict(found=True, error=0., heading=0., confidence=1.))
            row.update(extra)
            rows.append(row)
            out = fsm.step(MissionInput(**row))
            if out.state.value == 'fault':
                raise RuntimeError(out.reason)
        return out

    emit(start=True)
    emit(2, colours=['blue'])
    emit(2, events=['obstacle_near'])
    emit(2, events=['obstacle_crossed'])
    emit(31, stable=True)
    emit(2, colours=['red', 'green'])
    emit(2, events=['turn_complete'])
    emit(2)
    emit(2, colours=[colour])
    out = emit(2, events=['docked'])
    request_id = out.delivery_request.request_id
    emit(delivery_ack_id=request_id, delivery_ack_status='completed',
         delivery_ack_stamp=round(now + .1, 6))
    emit(2, events=['delivery_confirmed'])
    emit(2, events=['return_aligned'])
    emit(2)
    emit(2, colours=['green'])
    emit(2, events=['turn_complete'])
    emit(2)
    emit(2, events=['home_visible'])
    out = emit(2, events=['home_confirmed'])
    if out.state.value != 'done':
        raise RuntimeError('synthetic demo failed to complete')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'mission.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    import yaml
    (output / 'config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    return output / 'config.yaml', output / 'mission.jsonl'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('tmp/mission-demo'))
    parser.add_argument('--colour', choices=('yellow', 'brown'), default='yellow')
    args = parser.parse_args()
    config, replay = generate(args.output, args.colour)
    print(f'python -m dog3588 --config "{config}" mission-replay "{replay}" --mission-id demo --require-complete')
