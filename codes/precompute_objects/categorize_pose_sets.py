"""Partition saved groups into legal common, legal noncommon, and illegal files."""
import hashlib
import json
import shutil
from pathlib import Path
from codes.precompute_objects.common_directions import ROOT,write,classify
import numpy as np


def run():
    archive=ROOT/'codes/precompute_objects/data/pose_sets_before_categories_20261006'
    archive.mkdir(parents=True,exist_ok=True)
    summary=[]
    for source in sorted((ROOT/'objects').glob('*/pose_sets.json')):
        folder=source.parent;data=json.loads(source.read_text())
        if data.get('category')=='legal_with_common_direction':
            counts={k:json.loads((folder/f).read_text())['set_count'] for k,f in {
                'legal_with_common_direction':'pose_sets.json','legal_without_common_direction':'no_common_direction_pose_sets.json','illegal':'illegal_pose_sets.json'}.items()}
            summary.append(dict(object=folder.name,already_partitioned=True,counts=counts));continue
        original_sets=data['sets'];canonical_ids=[g['id'] for g in original_sets]
        normals={r['pose_id']:np.asarray(r['T_world_mesh'])[:3,:3].T@np.array([0.,0.,1.])
                 for r in json.loads((folder/'poses.json').read_text())['poses']}
        categories={key:[] for key in ['legal_with_common_direction','legal_without_common_direction','illegal']}
        saved=archive/folder.name;saved.mkdir(exist_ok=True)
        source_hashes={}
        for name in ['pose_sets.json','no_common_direction_pose_sets.json','illegal_pose_sets.json','common_direction_audit.json']:
            path=folder/name
            if path.exists():
                if (saved/name).exists():raise RuntimeError(f'Preserve archive: {saved/name}')
                shutil.copyfile(path,saved/name)
                source_hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        groups=[]
        for name in ['pose_sets.json','no_common_direction_pose_sets.json','illegal_pose_sets.json']:
            path=folder/name
            if path.exists():
                for g in json.loads(path.read_text())['sets']:
                    groups.append((name,g))
        seen=set();matrix=np.asarray(data['directed_violating_counts'])
        for name,g in groups:
            key=tuple(sorted(g['poses']))
            if key in seen:raise RuntimeError(f'Duplicate group: {folder.name}/{key}')
            seen.add(key)
            indices=[int(p.split('_')[1])-1 for p in g['poses']]
            legal=not np.any(matrix[np.ix_(indices,indices)])
            direction=classify(np.array([normals[p] for p in g['poses']]))
            category='illegal' if not legal else ('legal_with_common_direction' if direction['common_direction_exists'] else 'legal_without_common_direction')
            categories[category].append(dict(g,passed=bool(legal),floor_demands_compatible=bool(legal),
                 total_directed_violating_samples=int(matrix[np.ix_(indices,indices)].sum()),common_direction=direction,
                 canonical_precomputed_set=name=='pose_sets.json',original_collection=name))
        names={'legal_with_common_direction':'pose_sets.json','legal_without_common_direction':'no_common_direction_pose_sets.json','illegal':'illegal_pose_sets.json'}
        for category,name in names.items():
            metadata={k:v for k,v in data.items() if k not in ['sets','set_count','schema','passed']} if category=='legal_with_common_direction' else {}
            rows=categories[category]
            write(folder/name,dict(metadata,schema='cadgrasp_categorized_pose_sets_v1',object=folder.name,
                category=category,passed=category!='illegal',set_count=len(rows),sets=rows,
                pose_manifest_sha256=data['pose_manifest_sha256'],canonical_set_ids=canonical_ids if category=='legal_with_common_direction' else [],
                total_legal_set_count=sum(len(categories[k]) for k in ['legal_with_common_direction','legal_without_common_direction']),
                partition_source_sha256=source_hashes,
                scope='legal means saved full-load native-floor compatibility; common direction means nonzero native-floor hemisphere intersection; no full fixture acceptance'))
        summary.append(dict(object=folder.name,counts={k:len(v) for k,v in categories.items()}))
    write(ROOT/'objects/pose_set_categories.json',dict(schema='cadgrasp_pose_set_categories_v1',objects=summary,
          files=dict(legal_with_common_direction='pose_sets.json',legal_without_common_direction='no_common_direction_pose_sets.json',illegal='illegal_pose_sets.json'),
          archive=str(archive.relative_to(ROOT))))
    return summary

if __name__=='__main__':
    print(json.dumps(run(),indent=2))
