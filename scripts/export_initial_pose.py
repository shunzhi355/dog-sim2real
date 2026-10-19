"""Export the URDF assembled zero pose for manual calibration, never motor control.

Requires numpy and matplotlib. All dimensions come from the supplied URDF.
Only zero-rotation joint origins are supported; incompatible models are rejected.
"""

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def geometry(path):
    robot = ET.parse(path).getroot()
    origins = {'base_link': np.zeros(3)}
    pending = list(robot.findall('joint'))
    while pending:
        ready = [j for j in pending if j.find('parent').get('link') in origins]
        if not ready:
            raise ValueError('Disconnected URDF')
        for joint in ready:
            origin = joint.find('origin')
            if not np.allclose(np.fromstring(origin.get('rpy', '0 0 0'), sep=' '), 0, atol=1e-12):
                raise ValueError('Expected zero joint-origin rotations')
            origins[joint.find('child').get('link')] = (
                origins[joint.find('parent').get('link')]
                + np.fromstring(origin.get('xyz', '0 0 0'), sep=' '))
            pending.remove(joint)
    result = {}
    for leg in ('fl', 'fr', 'rl', 'rr'):
        points = {label: origins[f'{leg}_{body}'] * 1000 for label, body in {
            'H': 'mount', 'P': 'upper_leg', 'D': 'input_crank', 'A': 'coupler',
            'K': 'lower_leg', 'R': 'drive_rod'}.items()}
        for label, suffix in (('B', 'input_loop'), ('F', 'foot_loop')):
            loop = robot.find(f"loop_joint[@name='{leg}_{suffix}']")
            ends = [(origins[e.get('link')] + np.fromstring(e.get('xyz'), sep=' ')) * 1000 for e in loop]
            if not np.allclose(ends[0], ends[1], atol=1e-7):
                raise ValueError('Open linkage at zero')
            points[label] = ends[0]
        foot = robot.find(f"link[@name='{leg}_lower_leg']/collision[@name='{leg}_lower_leg_foot_pad_collision']")
        points['C'] = points['K'] + np.fromstring(foot.find('origin').get('xyz'), sep=' ') * 1000
        radius = float(foot.find('geometry/sphere').get('radius')) * 1000
        upper = points['K'] - points['P']
        lower = points['C'] - points['K']
        crank = points['A'] - points['D']
        result[leg] = {
            'points_body_mm': {key: value.tolist() for key, value in points.items()},
            'foot_radius_mm': radius,
            'upper_back_down_deg': float(np.degrees(np.arctan2(-upper[2], -upper[0]))),
            'lower_forward_down_deg': float(np.degrees(np.arctan2(-lower[2], lower[0]))),
            'input_forward_up_deg': float(np.degrees(np.arctan2(crank[2], crank[0]))),
        }
    return result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--urdf', type=Path, required=True)
    cli.add_argument('--output', type=Path, default=Path('docs/assets/initial-pose'))
    args = cli.parse_args()
    legs = geometry(args.urdf)
    bottom = [v['points_body_mm']['C'][2] - v['foot_radius_mm'] for v in legs.values()]
    if not np.allclose(bottom, bottom[0], atol=1e-5):
        raise ValueError('Feet do not share a ground plane at zero')
    height = -min(bottom)
    report = {'definition': 'URDF assembled q=0; calibration reference, not learned equilibrium',
              'urdf_sha256': hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
              'body_axes': 'X forward, Y left, Z up', 'base_origin_height_mm': height,
              'target_joint_pos_rad': [0.] * 12, 'raw_zero_positions': [None] * 12,
              'direction_signs': [None] * 12, 'hardware_verified': False, 'legs': legs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    plt.rcParams['font.family'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.4), layout='constrained')
    fig.set_layout_engine('constrained', rect=(0, .06, 1, .94))
    fig.suptitle('Robot12 标定初始姿态：模型 12 轴 q = 0 rad\n尺寸来自 URDF 装配零位；不是舵机 raw=0 或 raw=500', fontsize=17)
    for ax, leg, title in zip(axes[:2], ('fl', 'rl'), ('左前腿侧视（右前镜像）', '左后腿侧视（右后镜像）')):
        v = legs[leg]
        pts = {k: np.array(p)[[0, 2]] for k, p in v['points_body_mm'].items()}
        origin_x = pts['P'][0]
        pts = {k: p - [origin_x, 0] + [0, height] for k, p in pts.items()}
        for a, b in (('P', 'D'), ('A', 'B'), ('B', 'P'), ('P', 'R'), ('R', 'F'), ('F', 'K')):
            ax.plot(*np.array([pts[a], pts[b]]).T, color='#9aa6b2', lw=2)
        for a, b, color, label in (('P','K','#136caa','大腿 P→K'), ('K','C','#dd761a','小腿 K→足垫中心 C'), ('D','A','#8e44ad','输入曲柄 D→A')):
            ax.plot(*np.array([pts[a], pts[b]]).T, color=color, lw=4, label=label)
        offsets = {'P':(-14,3),'D':(3,-5),'A':(4,0),'B':(-6,8),'K':(-13,-5),'R':(-13,5),'F':(-13,2),'C':(7,6)}
        for key, offset in offsets.items():
            p = pts[key]
            ax.plot(*p, 'o', color='#243747', ms=4)
            ax.annotate(key, p, xytext=offset, textcoords='offset points', fontsize=11)
        ax.add_patch(Circle(pts['C'], v['foot_radius_mm'], facecolor='#c6e8f8', edgecolor='#136caa', alpha=.65))
        ax.axhline(0, color='#243747', lw=1)
        ax.set_title(title + f"\n大腿后下 {v['upper_back_down_deg']:.2f}°；曲柄前上 {v['input_forward_up_deg']:.2f}°", fontsize=12)
        ax.annotate('前方 +X →', (-15, 35), fontsize=12)
        ax.text(.03,.04, f"P 到地面 {pts['P'][1]:.2f} mm\nP→K 长度 {np.linalg.norm(pts['K']-pts['P']):.2f} mm\n小腿轴线前下 {v['lower_forward_down_deg']:.2f}°", transform=ax.transAxes, fontsize=10, bbox=dict(facecolor='white', edgecolor='none', alpha=.95))
        ax.set(xlim=(-125,80), ylim=(-30,210), xlabel='相对大腿根轴 P 的前后距离 / mm', ylabel='距足垫底部平面的高度 / mm')
        ax.set_aspect('equal'); ax.grid(alpha=.2)
    axes[0].legend(loc='upper left', fontsize=8, framealpha=.9)
    ax = axes[2]
    for leg, v in legs.items():
        x,y,z = v['points_body_mm']['C']
        # Top view: +X upwards, +Y leftwards.
        ax.add_patch(Circle((-y,x),v['foot_radius_mm'], facecolor='#c6e8f8',edgecolor='#136caa'))
        ax.text(-y,x,leg.upper(),ha='center',va='center',fontsize=11)
        h = v['points_body_mm']['H']
        p = v['points_body_mm']['P']
        ax.plot([-h[1],-p[1],-y],[h[0],p[0],x],color='#9aa6b2',lw=2,zorder=0)
    ax.plot(0,0,'+',ms=12,color='#243747');ax.annotate('机身原点', (0,0),xytext=(5,7),textcoords='offset points',fontsize=10)
    ax.annotate('前方 +X', (0,135),ha='center',fontsize=12)
    front=legs['fl']['points_body_mm']['C'];rear=legs['rl']['points_body_mm']['C']
    ax.set_title(f'俯视：足垫中心位置\n前宽 {2*front[1]:.2f} / 后宽 {2*rear[1]:.2f} mm\n前后间距 {front[0]-rear[0]:.2f} mm',fontsize=12)
    ax.set(xlim=(-130,130),ylim=(-150,155),xlabel='左 ← 横向距离 / mm → 右',ylabel='后 ← 前后距离 / mm → 前')
    ax.set_aspect('equal');ax.grid(alpha=.2)
    fig.text(.5,.015,f'机身原点离地 {height:.3f} mm（不是机身底板高度）；模型几何，不是实机测量。',ha='center',fontsize=11)
    fig.savefig(args.output.with_suffix('.png'),dpi=160)
    svg = args.output.with_suffix('.svg')
    fig.savefig(svg)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
