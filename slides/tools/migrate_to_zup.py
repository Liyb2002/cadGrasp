"""One-time inverse migration of a local reflected Y-up dataset to native Z-up.

Run from a checkout whose source has already been converted. The manifest blocks
reapplication. Existing mesh and pose files are changed in place; solver-basis
fixtures are excluded. Historical reports are marked as transported, not rerun.
Use the baseline entry points afterwards to produce current-code certificates.
"""
def main():
    from pathlib import Path
    import sys,json,numpy as np,hashlib,struct,re
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    import coordinate_transport as C,coordinate_data as D
    D.POLAR.update({'point','points','supports','patch_min_rho_at','upper','camera_position','camera_target','direction_vector'})
    ROOT=Path(__file__).resolve().parents[2]
    manifest=ROOT/'slides/tools/coordinate_migration.json'
    prior = json.loads(manifest.read_text()) if manifest.exists() else {}
    if prior.get('coordinate_system') == 'z_up_xy_floor':
        raise SystemExit('Already migrated to Z-up')
    if prior.get('coordinate_system') != 'y_up_xz_floor':
        raise SystemExit('Expected a recorded reflected Y-up dataset; native Z-up inputs must not be converted')
    hashes={};counts={}
    def sha(data):return hashlib.sha256(data).hexdigest()
    def save(p,old,new):
     if old==new:return
     p.write_bytes(new);hashes[sha(old)]=sha(new);counts[p.suffix]=counts.get(p.suffix,0)+1
    # Preserve STL triangle and face order exactly; OBJ group/material records remain intact.
    for folder in ('objects','slides'):
     for p in (ROOT/folder).rglob('*'):
      if p.suffix.lower()=='.stl':
       old=p.read_bytes(); n=struct.unpack_from('<I',old,80)[0]
       if len(old)!=84+50*n:
        lines=old.decode().splitlines(True);vertices=[]
        for i,line in enumerate(lines):
         bits=line.split()
         if bits and bits[0]=='vertex':bits[2],bits[3]=bits[3],bits[2];lines[i]=' '.join(bits)+'\n';vertices.append(i)
         elif bits[:2]==['facet','normal']:bits[3],bits[4]=bits[4],bits[3];lines[i]=' '.join(bits)+'\n'
         elif bits and bits[0]=='endloop':
          if len(vertices)!=3:raise ValueError(p)
          a,b=vertices[1:];lines[a],lines[b]=lines[b],lines[a];vertices=[]
        save(p,old,''.join(lines).encode());continue
       dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attr','<u2')]);a=np.frombuffer(old[84:],dtype).copy()
       a['normal']=C.polar(a['normal']);a['vertices']=C.triangles(a['vertices']);save(p,old,old[:84]+a.tobytes())
      elif p.suffix.lower()=='.obj':
       old=p.read_bytes();lines=[]
       for line in old.decode().splitlines(True):
        bits=line.split()
        if bits and bits[0] in ('v','vn') and len(bits)>=4:bits[2],bits[3]=bits[3],bits[2];line=' '.join(bits)+'\n'
        elif bits and bits[0]=='f':line=' '.join([bits[0],bits[1],*bits[:1:-1]])+'\n'
        lines.append(line)
       save(p,old,''.join(lines).encode())
    # Strings include stored paths; only known world fields are transported.
    def strings(x):
     if isinstance(x,str):return x.replace('slides/obj_supp/baseline_algo','slides/baseline_algo').replace('obj_supp/baseline_algo','baseline_algo').replace('y_up_xz_floor','z_up_xy_floor').replace('Y-up','Z-up').replace('\\hat{y}','\\hat{z}')
     if isinstance(x,list):return [strings(a) for a in x]
     if isinstance(x,dict):return {k:strings(v) for k,v in x.items()}
     return x
    for folder in ('objects','slides'):
     for p in (ROOT/folder).rglob('*.json'):
      old=p.read_bytes()
      try:data=json.loads(old)
      except ValueError:continue
      result=strings(D.record(data))
      if isinstance(result,dict) and (folder=='objects' or 'output' in p.parts or 'setup' in p.parts):
       result['coordinate_system']='z_up_xy_floor'
       if 'output' in p.parts:result['coordinate_migration']={'method':'P=(x,z,y); axial=-P; oriented triangles reverse winding','algorithms_reexecuted':False}
      new=(json.dumps(result,ensure_ascii=False,separators=(',',':'),allow_nan=True)+'\n').encode();save(p,old,new)
    for folder in ('objects','slides'):
     for p in (ROOT/folder).rglob('*.npz'):
      if 'fixtures' in p.parts:continue
      old=p.read_bytes()
      with np.load(p) as z: a=D.arrays({k:z[k] for k in z.files})
      for k,v in a.items():
       if v.dtype.kind=='U':a[k]=np.array([strings(t) for t in v.flat]).reshape(v.shape)
      np.savez_compressed(p,**a);hashes[sha(old)]=sha(p.read_bytes());counts['.npz']=counts.get('.npz',0)+1
    # Native MuJoCo planes, inertias, free-joint keyframes and gravity use Z-up.
    from scene import write_scene
    for p in (ROOT/'objects').glob('*/scene.xml'):
     old=p.read_bytes();write_scene(p.parent.name)
     hashes[sha(old)]=sha(p.read_bytes());counts['.xml']=counts.get('.xml',0)+1
    # Rewrite existing artifact hash links topologically. Code fingerprints are left
    # intact: converted historical certificates are not claimed to be new executions.
    for iteration in range(30):
     changed=0
     for folder in ('objects','slides'):
      for p in (ROOT/folder).rglob('*.json'):
       old=p.read_bytes();text=old.decode()
       def replace(m):
        v=m[0];seen=set()
        while v in hashes and v not in seen:seen.add(v);v=hashes[v]
        return v
       new=re.sub(r'(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])',replace,text).encode()
       if new!=old:save(p,old,new);changed+=1
      for p in (ROOT/folder).rglob('*.npz'):
       with np.load(p) as z:a={k:z[k] for k in z.files}
       update=False
       for k,v in a.items():
        if v.dtype.kind=='U':
         out=np.array([replace(re.match('.+',str(x))) if re.fullmatch('[0-9a-f]{64}',str(x)) else str(x) for x in v.flat]).reshape(v.shape)
         if not np.array_equal(out,v):a[k]=out;update=True
       if update:
        old=p.read_bytes();np.savez_compressed(p,**a);hashes[sha(old)]=sha(p.read_bytes());changed+=1
     print('hash pass',iteration,changed,flush=True)
     if not changed:break
    else:raise RuntimeError('Hash references did not settle')
    manifest.write_text(json.dumps({'coordinate_system':'z_up_xy_floor','counts':counts,'mapping':'positions/forces: (x,z,y); torques/axes: (-x,-z,-y); rigid transforms P T P; faces reversed','rerun_cases':[]},indent=2)+'\n')
    print(counts)


if __name__ == "__main__":
    main()
