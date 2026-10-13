"""Report physical operation changes across hosts without changing saved traces."""
import _bootstrap
import argparse
from co_common import *
from run_all import saved_groups,output_root
from whole_pipeline import load_layout


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('object',nargs='?',default='B');args=parser.parse_args()
    count=0
    for group in saved_groups(args.object):
        out=output_root(args.object)/group['id']/'step4/step4.2'
        if not (out/'process.json').exists():continue
        try:
            report=I.check_report(out/'data/report.json')
            assert report['force_exit_work_passed']
        except (OSError,RuntimeError,AssertionError):continue
        with np.load(out/'layout.npz') as z:native=z['native_world']
        mesh=state(args.object,group['poses'][0])[2];center=mesh.center_mass
        process=json.loads((out/'process.json').read_text());previous=None;rows=[];inputs=[out/'process.json',out/'layout.npz',out/'data/report.json']
        for step in process:
            path=out/step['layout'];inputs.append(path);layout=load_layout(path)
            changes=[]
            if previous is not None:
                for k,name in enumerate(group['poses']):
                    old_world=native[previous.hosts[k]]@previous.placements[k]
                    new_world=native[layout.hosts[k]]@layout.placements[k]
                    move=(new_world[:3,:3]@center+new_world[:3,3])-(old_world[:3,:3]@center+old_world[:3,3])
                    old_exit=previous.placements[k,:3,:3].T@previous.directions[k]
                    new_exit=layout.placements[k,:3,:3].T@layout.directions[k]
                    angle=float(np.degrees(np.arccos(np.clip(old_exit@new_exit,-1,1))))
                    rehosted=previous.hosts[k]!=layout.hosts[k]
                    if np.linalg.norm(move)>1e-12 or angle>1e-5 or rehosted:
                        changes.append(dict(pose=name,rehosted=bool(rehosted),
                            old_host=group['poses'][previous.hosts[k]],host=group['poses'][layout.hosts[k]],
                            world_body_center_displacement_mm=(1000*move).tolist(),
                            world_body_center_distance_mm=float(1000*np.linalg.norm(move)),
                            object_relative_exit_angle_degrees=angle))
            rows.append(dict(index=step['index'],phase=step['phase'],changes=changes));previous=layout
        save(out/'data/operation_metrics.json',dict(poses=group['poses'],steps=rows,
            explanation='Across Juxtapose hosts, raw fixture translation/direction differences in process.json include a coordinate change. These metrics compare physical world body centers and object-relative exit angles.',
            geometry_and_original_traces_changed=False,provenance=provenance(inputs,[Path(__file__)])))
        page=out/'README.md';marker='<!-- physical-operation-metrics -->'
        if page.exists():
            text=page.read_text().split(marker)[0].rstrip()
            lines=[marker,'','换 host 时，上表的方向／平移是支撑坐标参数差，包含坐标系变更。'
                '物体实际变化应读下面的世界中心距离和物体自身退出角；原日志和布局保持不变。','',
                '| 步 | pose | 世界物体中心移动mm | 物体自身退出角变化° |','|---|---|---:|---:|']
            for step in rows:
                for change in step['changes']:
                    lines.append(f"| {step['index']} | {change['pose']} | {change['world_body_center_distance_mm']:.3f} | {change['object_relative_exit_angle_degrees']:.3f} |")
            page.write_text(text+'\n\n'+'\n'.join(lines)+'\n')
        count+=1
    print('PHYSICAL OPERATION METRICS',count,'cases',flush=True)


if __name__=='__main__':main()
