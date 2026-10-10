import hashlib,json,subprocess,time
from pathlib import Path
root=Path('slides/pose_set_search/output/B_work_cone_20261008');records=[];pauses=0;began=time.monotonic()
for folder in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith('pose')):
 while not (folder/'videos.json').exists():time.sleep(3)
 metadata=json.loads((folder/'videos.json').read_text())
 for kind in ['process','result']:
  d=metadata[kind];movie=folder/d['file']
  assert d['onscreen_text'] is False and d['view_count']==1
  assert d['method_order']==['whole','incremental'] and d['camera']['fixed']
  assert d['sha256']==hashlib.sha256(movie.read_bytes()).hexdigest()
  if kind=='process':assert d['process_animation_uses_saved_endpoint_poses']
  run=subprocess.run(['ffmpeg','-v','error','-i',str(movie),'-f','null','-'],capture_output=True,text=True)
  assert run.returncode==0 and not run.stderr,run.stderr
  if kind=='result':
   runs=[];count=0;last=None
   for frame in d['timeline']:
    if frame.get('phase')=='seated':
     assert frame['work_area_visible'] and frame['forbidden_access_region_visible'] and frame['possible_force_arrows_visible']
     key=(frame['method'],frame['pose'])
     if last==key:count+=1
     else:
      if count:runs.append(count)
      last=key;count=1
    elif count:runs.append(count);count=0;last=None
   if count:runs.append(count)
   assert all(n==24 for n in runs)
   expected=sum(json.loads((folder/m/'search_report.json').read_text())['pose_count'] for m in ['whole','incremental'])
   assert len(runs)==expected,(folder,len(runs),expected)
   pauses+=len(runs)
  records.append(dict(case=folder.name,kind=kind,decoded=True,file=d['file'],frames=d['frame_count'],sha256=d['sha256']))
  print('DECODED',folder.name,kind,flush=True)
 for method in ['whole','incremental']:
  directory=folder/method;rows=json.loads((directory/'process.json').read_text());geometry=json.loads((directory/'mesh_geometry.json').read_text())
  assert len(rows)==len(geometry['steps'])
  for row,geo in zip(rows,geometry['steps']):
   assert row['index']==geo['index'] and (directory/geo['mesh']).exists()
   if 'layout_sha256' in geo:assert geo['layout_sha256']==hashlib.sha256((directory/row['layout']).read_bytes()).hexdigest()
  render=json.loads((directory/'render.json').read_text())
  assert render['forbidden_access_regions_drawn']
  assert all(work['half_angle_deg']==30 and work['total_opening_deg']==60 for work in render['work_access_definition'])
assert len(records)==20
(root/'media_checks.json').write_text(json.dumps(dict(complete=True,decoded_movie_count=len(records),seated_pose_pauses=pauses,pause_frames=24,fps=24,view_count=1,forbidden_regions_visible=True,pressure_acceptance=False,records=records,seconds=time.monotonic()-began),indent=2)+'\n')
print('MEDIA CHECKS COMPLETE',len(records),'movies',pauses,'one-second pauses',flush=True)
