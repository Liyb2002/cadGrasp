import json,hashlib,subprocess
from pathlib import Path
root=Path('slides/pose_set_search/output/B_work_cone_20261008');path=root/'media_checks.json';checks=json.loads(path.read_text());refreshed=[]
for record in checks['records']:
 folder=root/record['case'];metadata=json.loads((folder/'videos.json').read_text())[record['kind']];movie=folder/metadata['file']
 actual=hashlib.sha256(movie.read_bytes()).hexdigest();assert actual==metadata['sha256']
 if actual!=record['sha256']:
  result=subprocess.run(['ffmpeg','-v','error','-i',str(movie),'-f','null','-'],capture_output=True,text=True)
  assert result.returncode==0 and not result.stderr,result.stderr
  record.update(sha256=actual,frames=metadata['frame_count'],redecoded_after_feasible_volume_polish=True)
  refreshed.append(str(movie.relative_to(root)))
  if record['kind']=='result':
   runs=[];count=0;last=None
   for row in metadata['timeline']:
    if row.get('phase')=='seated':
     assert row['work_area_visible'] and row['forbidden_access_region_visible'] and row['possible_force_arrows_visible']
     key=(row['method'],row['pose'])
     if last==key:count+=1
     else:
      if count:runs.append(count)
      count=1;last=key
    elif count:runs.append(count);count=0;last=None
   if count:runs.append(count)
   assert len(runs)==20 and all(n==24 for n in runs)
checks['refreshed_after_final_polish']=refreshed
path.write_text(json.dumps(checks,indent=2)+'\n')
print('ALL20 CURRENT VIDEO HASHES MATCH; redecoded changed movies:',refreshed,flush=True)
