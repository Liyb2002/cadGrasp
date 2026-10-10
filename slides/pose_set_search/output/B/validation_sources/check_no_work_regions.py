import json,hashlib,subprocess,sys,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,'slides/pose_set_search/code')
from render_mesh_media import WorkingSurface,C,np
from case_sets import CASES,case_directory
root=Path('slides/pose_set_search/output/B');archive=root/'_history/before_separate_work_access_20261008'
hash_file=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protected=json.loads((archive/'unchanged_artifacts.json').read_text())
for name,expected in protected.items():assert hash_file(root/name)==expected,name
production=json.loads((root/'protected_coopt_sources.json').read_text())
for name,expected in production.items():assert hash_file(Path(name))==expected,name
records=[];pauses=0;unique_surfaces={}
for case in CASES:
 folder=case_directory(root,case);media=json.loads((folder/'videos.json').read_text())
 assert media['complete'] and media['geometry_preserved'] and media['view_count']==1
 for kind in ['process','result']:
  metadata=media[kind]
  assert not metadata['forbidden_access_regions_drawn'] and not metadata['onscreen_text']
  assert metadata['camera']['fixed'] and metadata['view_count']==1 and metadata['fps']==24
  assert metadata['method_order']==['whole','incremental']
  assert metadata['frame_count']==len(metadata['timeline'])
  assert all(not row.get('forbidden_access_region_visible',False) for row in metadata['timeline'])
  movie=folder/metadata['file'];assert hash_file(movie)==metadata['sha256']
  records.append(dict(case=folder.name,kind=kind,file=movie.name,frames=metadata['frame_count'],sha256=metadata['sha256']))
  if kind=='result':
   unique_surfaces.update(metadata['working_surfaces'])
   runs=[];count=0;last=None
   for row in metadata['timeline']:
    if row.get('phase')=='seated':
     assert row['work_area_visible'] and row['possible_force_arrows_visible'] and row['force_arrow_count']==28
     key=(row['method'],row['pose'])
     if last==key:count+=1
     else:
      if count:runs.append(count)
      count=1;last=key
    elif count:runs.append(count);count=0;last=None
   if count:runs.append(count)
   assert len(runs)==2*len(CASES[case]) and all(n==24 for n in runs)
   pauses+=len(runs)
 for method in ['whole','incremental']:
  figure=json.loads((folder/method/'render.json').read_text())
  assert not figure['forbidden_access_regions_drawn'] and not figure['process_png_onscreen_text']
  assert figure['source_mesh_sha256']==hash_file(folder/method/'support.obj')
body=C.state('B','pose_1')[2]
for pose,surface in unique_surfaces.items():
 work=WorkingSurface(body,pose)
 assert all(len(layer)==2 for layer in work.layers(np.eye(4),arrows=True))
 assert surface['face_ids']==work.ids.tolist() and surface['arrow_count']==28
 assert not surface['forbidden_access_region_drawn']
 faces=np.asarray(surface['sample_face_ids']);bary=np.asarray(surface['sample_barycentric'])
 assert np.isin(faces,work.ids).all()
 assert np.allclose(np.einsum('av,avc->ac',bary,body.triangles[faces]),surface['sample_points_object_m'],atol=1e-12,rtol=0)
 assert np.allclose(-body.face_normals[faces],surface['force_directions_object'],atol=1e-12,rtol=0)
def decode(record):
 result=subprocess.run(['ffmpeg','-v','error','-threads','2','-i',str(root/record['case']/record['file']),'-f','null','-'],capture_output=True,text=True)
 assert result.returncode==0 and not result.stderr,result.stderr
 return dict(record,decoded=True)
with ThreadPoolExecutor(max_workers=2) as pool:records=list(pool.map(decode,records))
figure=json.loads((root/'work_access_isometric.json').read_text())
assert hash_file(root/'work_access_isometric.png')==figure['sha256'] and figure['view_count']==8
assert figure['display_region']['source_cone_definition']['half_angle_deg']==30
assert figure['display_region']['subset_of_search_exclusion'] and figure['geometry_and_search_unchanged']
assert figure['display_region']['maximum_cap_radial_error_m']<.00003
assert hash_file(root/Path(figure['source_layout']).parent/'support.obj')==figure['support_sha256']
checks=dict(complete=True,decoded_movie_count=len(records),seated_pose_pauses=pauses,pause_frames=24,fps=24,view_count=1,
 forbidden_regions_visible=False,forbidden_regions_in_separate_figure=True,forbidden_region_figure='work_access_isometric.png',
 pressure_acceptance=False,records=records,immutable_artifact_count=len(protected),immutable_artifacts_unchanged=True,
 production_sources_unchanged=True,working_surfaces_and_inward_force_samples_preserved=True)
(root/'media_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
print('VERIFIED',len(records),'decoded movies;',pauses,'one-second force pauses;',len(protected),'unchanged artifacts;',len(production),'unchanged production sources;',len(unique_surfaces),'original work surfaces')
