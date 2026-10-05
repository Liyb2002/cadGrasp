"""Co-optimize independent floor-legal straight exits; recover sampled equilibrium."""
from co_common import *
from scipy.optimize import linprog
import argparse,time,contextlib,traceback


def fibonacci(count):
    k=np.arange(count);z=1-2*(k+.5)/count;theta=k*np.pi*(3-np.sqrt(5))
    return np.c_[np.sqrt(1-z*z)*np.cos(theta),np.sqrt(1-z*z)*np.sin(theta),z]

def project_common(common,normals,lift=.01):
    directions=[]
    for normal in normals:
        d=common-min(common@normal,0.)*normal+lift*normal
        if np.linalg.norm(d)<1e-10:
            axis=np.eye(3)[np.argmin(np.abs(normal))];d=axis-(axis@normal)*normal+lift*normal
        directions.append(d/np.linalg.norm(d))
    return np.asarray(directions)

class Search:
    def __init__(self,name,group):
        self.name=name;self.group=group;self.base=HERE/'output'/name/group['id'];self.out=self.base/'step4/step4.2';(self.out/'data').mkdir(parents=True,exist_ok=True)
        self.states=[state(name,p)[:2] for p in group['poses']];self.mesh=state(name,group['poses'][0])[2]
        self.normals=np.array([T[:3,:3].T@np.array([0.,0.,1.]) for task,T in self.states])
        self.seed_path=self.base/'step3/step3.3/support_with_rings.obj';self.seed_mesh=trimesh.load(self.seed_path,force='mesh',process=False);self.seed=S.solid(self.seed_mesh)
        z=np.load(self.base/'step3/step3.2/data/contacts.npz');self.tri=z['triangles_mesh_m'];self.src=z['source_faces'];self.allowed=z['allowed_faces'];self.face_normals=self.mesh.face_normals[self.src]
        self.initial=json.loads((self.base/'step4/step4.1/data/report.json').read_text());self.length=self.initial['initialization']['full_length_m']
        self.indices=[]
        for p in group['poses']:
            mask=np.load(self.base/f'step4/step4.1/data/{p}.npz')['force_mask'];bad=np.flatnonzero(~mask);self.indices.append([int(bad[0])] if len(bad) else [0])
        self.seen=set();self.evaluations=0;self.full_classifications=0;self.exact_evaluations=0;self.best=None;self.best_common=None;self.trace=[]

    def classify(self,triangles,sources):
        masks=[];infos=[];supplies=[]
        for task,T in self.states:
            full=supply(task,T,triangles,sources);mask,info=J.classify(full,task.targets);masks.append(mask);infos.append(info);supplies.append(full)
        self.full_classifications+=1;return masks,infos,supplies

    def candidate(self,directions,common=None,label='proposal',proxy=True):
        directions=np.asarray(directions,float);directions/=np.linalg.norm(directions,axis=1)[:,None]
        if np.any(np.sum(directions*self.normals,axis=1)<-1e-12):return None
        key=tuple(np.round(directions.flatten(),9))
        if key in self.seen:return None
        self.seen.add(key);self.evaluations+=1
        if proxy:
            # This ignores nonlocal shadowing. It only screens out proposals;
            # no proxy result is ever accepted as actual remaining material.
            keep=np.max(self.face_normals@directions.T,axis=1)<=1e-9
            t,src=self.tri[keep],self.src[keep]
            supplies=[supply(task,T,t,src) for task,T in self.states]
            for k,((task,T),full) in enumerate(zip(self.states,supplies)):
                for index in self.indices[k]:
                    if C.W.solve(full,U.target(task.targets[index])) is None:return None
            masks=[]
            for k,((task,T),full) in enumerate(zip(self.states,supplies)):
                mask,info=J.classify(full,task.targets);self.full_classifications+=1
                if not mask.all():
                    bad=int(np.flatnonzero(~mask)[0]);self.indices[k]=list(dict.fromkeys(self.indices[k]+[bad]))[-8:];return None
                masks.append(mask)
        self.exact_evaluations+=1;best_geometry=None
        for fan in [8,2,16]:
            try:
                sweeps=[S.solid(S.swept_solid(self.mesh,self.length*d,fan_in=fan)) for d in directions]
                cut=union(sweeps);remaining=self.seed-cut;removed=self.seed-remaining
                overlap=max(material_volume(remaining^part) for part in sweeps)
                partition=abs(material_volume(self.seed)-material_volume(remaining)-material_volume(removed))
                error=max(overlap,partition)
                if best_geometry is None or error<best_geometry[0]:best_geometry=(error,fan,sweeps,cut,remaining,removed,overlap,partition)
                if error<1e-10:break
            except (RuntimeError,ValueError):continue
        if best_geometry is None:return None
        error,fan,sweeps,cut,remaining,removed,overlap,partition=best_geometry
        if error>=1e-10:
            self.trace.append(dict(proposal=self.evaluations,label=label,status='geometry_unresolved',error_m3=error));return None
        remaining_mesh=S.unpack(remaining)
        tri,src=contact_boundary(self.mesh,remaining_mesh,self.allowed) if len(remaining_mesh.faces) else (np.empty((0,3,3)),np.empty(0,int))
        masks,infos,supplies=self.classify(tri,src);counts=np.array([m.sum() for m in masks]);sizes=np.array([len(m) for m in masks]);fractions=counts/sizes
        score=(int(sum(m.all() for m in masks)),float(fractions.min()),float(fractions.sum()))
        result=dict(directions=directions.copy(),fan=fan,sweeps=sweeps,cut=cut,remaining=remaining,removed=removed,triangles=tri,sources=src,masks=masks,infos=infos,supplies=supplies,overlap=overlap,partition=partition,counts=counts.tolist(),score=score,common=None if common is None else common.copy())
        row=dict(proposal=self.evaluations,label=label,actual_covered_counts=counts.tolist(),score=score,remaining_volume_cm3=material_volume(remaining)*1e6)
        self.trace.append(row)
        if self.best is None or score>self.best['score']:
            self.best=result;self.best_common=common;print('RECOVERY',self.group['id'],counts.tolist(),'proposal',self.evaluations,flush=True)
        for k,mask in enumerate(masks):
            if not mask.all():self.indices[k]=list(dict.fromkeys(self.indices[k]+[int(np.flatnonzero(~mask)[0])]))[-8:]
        return result if all(m.all() for m in masks) else None

    def search(self):
        initial=np.array([r['direction_fixture'] for r in self.initial['state_results']]);result=self.candidate(initial,label='native +z initialization',proxy=False)
        if result is not None:return result
        lp=linprog([0,0,0,-1],A_ub=np.c_[-self.normals,np.ones(len(self.normals))],b_ub=np.zeros(len(self.normals)),bounds=[(-1,1)]*3+[(None,None)],method='highs')
        if lp.success and np.linalg.norm(lp.x[:3])>1e-8:
            common=lp.x[:3]/np.linalg.norm(lp.x[:3]);result=self.candidate(project_common(common,self.normals),common,label='common floor cone')
            if result is not None:return result
        for count in [160,512,2048]:
            for common in fibonacci(count):
                result=self.candidate(project_common(common,self.normals),common,label=f'shared tendency {count}')
                if result is not None:return result
            if self.best_common is not None:
                center=self.best_common.copy();axis=np.eye(3)[np.argmin(abs(center))];u=np.cross(center,axis);u/=np.linalg.norm(u);v=np.cross(center,u)
                for angle in [20,10,5,2,.5]:
                    for theta in np.linspace(0,2*np.pi,32,endpoint=False):
                        a=np.cos(np.deg2rad(angle))*center+np.sin(np.deg2rad(angle))*(np.cos(theta)*u+np.sin(theta)*v)
                        for lift in [.0001,.003,.01,.03]:
                            result=self.candidate(project_common(a,self.normals,lift),a,label='common local refinement')
                            if result is not None:return result
            save(self.out/'data/search_checkpoint.json',dict(complete=False,proposals=self.evaluations,exact_evaluations=self.exact_evaluations,best_counts=None if self.best is None else self.best['counts'],trace=self.trace))
        # Allow each exit to deviate independently around the best real result.
        if self.best is not None:
            for sweep in range(8):
                before=self.best['score']
                for k in range(len(self.states)):
                    center=self.best['directions'][k].copy();axis=np.eye(3)[np.argmin(abs(center))];u=np.cross(center,axis);u/=np.linalg.norm(u);v=np.cross(center,u)
                    for angle in [20,10,5,2,.5]:
                        for theta in np.linspace(0,2*np.pi,16,endpoint=False):
                            d=np.cos(np.deg2rad(angle))*center+np.sin(np.deg2rad(angle))*(np.cos(theta)*u+np.sin(theta)*v);directions=self.best['directions'].copy();directions[k]=d
                            result=self.candidate(directions,label=f'independent pose {k} refinement')
                            if result is not None:return result
                if self.best['score']==before:break
        return None

def render(search,result,restored,out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.patches import Patch
    n=len(search.states);cols=min(n,3);nr=(n+cols-1)//cols;fig=plt.figure(figsize=(6*cols,6*nr))
    remaining=S.unpack(result['remaining']);restored_mesh=S.unpack(restored);old=result['remaining']-restored;old_mesh=S.unpack(old)
    displays=[S.swept_solid(search.mesh,.10*d,fan_in=result['fan']) for d in result['directions']]
    radius=max(np.ptp(np.vstack([transform_points(m.vertices,T) for m in [search.mesh,remaining,displays[i]] if len(m.vertices)]),axis=0).max() for i,(task,T) in enumerate(search.states))*1000*.55
    for i,(task,T) in enumerate(search.states):
        ax=fig.add_subplot(nr,cols,i+1,projection='3d');vertices=[]
        for m,color,alpha in [(displays[i],'#36bbd0',.10),(old_mesh,'#999999',.6),(search.mesh,'#79a9d7',.24),(restored_mesh,'#49b66a',.9)]:
            if not len(m.vertices):continue
            world=m.copy();world.apply_transform(T);vertices.append(world.vertices*1000);ax.add_collection3d(Poly3DCollection(world.triangles*1000,facecolor=color,edgecolor='none',alpha=alpha))
        vv=np.vstack(vertices);center=(vv.min(0)+vv.max(0))/2
        floor=np.array([[center[0]-radius,center[1]-radius,0],[center[0]+radius,center[1]-radius,0],[center[0]+radius,center[1]+radius,0],[center[0]-radius,center[1]+radius,0]])
        ax.add_collection3d(Poly3DCollection([floor],facecolor='#eeeeee',edgecolor='#bbbbbb',alpha=.15))
        native=T[:3,:3]@result['directions'][i];start=transform_points(search.mesh.vertices.mean(0)[None,:],T)[0]*1000;ax.quiver(*start,*native,length=100,color='#009cae',arrow_length_ratio=.13,linewidth=2)
        for axis,x in zip('xyz',center):getattr(ax,'set_'+axis+'lim')(x-radius,x+radius)
        ax.set_box_aspect((1,1,1));ax.view_init(22,-55);ax.set_axis_off();ax.set_title(search.group['poses'][i].replace('_',' ').title()+' — recovered equilibrium\n32768 / 32768 original demands',fontsize=11)
    fig.suptitle('Step4.2: co-optimized exits; green material restored\nAll original force / torque demands pass; connectivity is not required',fontsize=14)
    fig.legend(handles=[Patch(color=c,label=l) for c,l in [('#79a9d7','Object'),('#36bbd0','First 100 mm exit sweep'),('#49b66a','Restored material'),('#999999','Retained material')]],loc='lower center',ncol=4)
    fig.subplots_adjust(left=0,right=1,bottom=.05,top=.80 if nr==1 else .89,wspace=0,hspace=.08);fig.savefig(out/'overview.png',dpi=150,bbox_inches='tight');plt.close(fig)

def finish(search,result,began):
    out=search.out;initial_mesh=trimesh.load(search.base/'step4/step4.1/remaining_support.obj',force='mesh',process=False);restored=result['remaining']-S.solid(initial_mesh)
    for filename,solid in [('remaining_support.obj',result['remaining']),('removed_support.obj',result['removed']),('restored_support.obj',restored)]:D.export_exact_obj(S.unpack(solid),out/filename)
    np.savez_compressed(out/'data/remaining_contacts.npz',triangles_mesh_m=result['triangles'],source_faces=result['sources']);rows=[];inputs=[search.seed_path,search.base/'step3/step3.2/data/contacts.npz',search.base/'step3/step3.3/data/report.json',search.base/'step4/step4.1/data/report.json',search.base/'step4/step4.1/remaining_support.obj'];artifacts={}
    for k,(pose,(task,T)) in enumerate(zip(search.group['poses'],search.states)):
        d=result['directions'][k];native=T[:3,:3]@d;minimum=float(min(task.domain.mesh.vertices[:,2].min(),task.domain.mesh.vertices[:,2].min()+search.length*native[2]));separated=bool(((search.mesh.vertices+search.length*d)@d).min()>(search.seed_mesh.vertices@d).max()+1e-9)
        assert minimum>=-1e-9 and separated and result['masks'][k].all()
        row=dict(pose=pose,force_passed=True,force_covered=int(result['masks'][k].sum()),load_count=len(result['masks'][k]),direction_fixture=d.tolist(),direction_world=native.tolist(),path_fixture_m=[[0.,0.,0.],(d*search.length).tolist()],minimum_object_world_z_along_path_m=minimum,endpoint_completely_separated=separated,classifier=result['infos'][k]);rows.append(row);inputs+=task.inputs
        np.savez_compressed(out/'data'/f'{pose}.npz',mask=result['masks'][k],supply_7d=result['supplies'][k],T_fixture_to_world=T,direction_world=native);D.export_exact_obj(S.unpack(result['sweeps'][k]),out/'data'/f'{pose}_sweep.obj')
        for f in [f'{pose}.npz',f'{pose}_sweep.obj']:artifacts[f]=I.sha256(out/'data'/f)
    render(search,result,restored,out)
    save(out/'data/search.json',dict(complete=True,proposal_count=search.evaluations,exact_evaluations=search.exact_evaluations,trace=search.trace))
    report=dict(complete=True,passed=True,status='force_and_exit_pass',stage='step4.2',pose_set=search.group['id'],state_results=rows,full_fixture_accepted=False,connectivity_required=False,ground_coverage_required=False,ground_rebuilt=False,original_loads_reused=True,load_subsampling=False,search_method='Deterministic shared-direction tendencies projected into each pose floor-legal hemisphere, local common and independent direction refinement; actual cut acceptance',path_kind='Independent straight exits',full_length_m=search.length,display_length_m=.10,proposal_count=search.evaluations,exact_evaluations=search.exact_evaluations,remaining_contact_triangle_count=len(result['triangles']),remaining_component_count=len(result['remaining'].decompose()),initial_remaining_volume_cm3=abs(initial_mesh.volume)*1e6,remaining_volume_cm3=material_volume(result['remaining'])*1e6,restored_volume_cm3=material_volume(restored)*1e6,removed_volume_cm3=material_volume(result['removed'])*1e6,maximum_remaining_sweep_overlap_m3=result['overlap'],volume_partition_error_m3=result['partition'],boolean_fan_in=result['fan'],validation_policy='One actual construction acceptance using all original force demands, continuous full object sweeps and path-floor legality; no exported-model replay. Connectivity, installed support floors, ground coverage and strength are deferred.',provenance=provenance(inputs,[HERE/'step42.py',HERE/'co_common.py',Path(S.__file__).with_name('translation_sweep.py'),Path(J.__file__),Path(C.W.__file__)]),artifacts={**artifacts,**{f'../{f}':I.sha256(out/f) for f in ['overview.png','remaining_support.obj','removed_support.obj','restored_support.obj']},'remaining_contacts.npz':I.sha256(out/'data/remaining_contacts.npz'),'search.json':I.sha256(out/'data/search.json')},seconds=time.monotonic()-began)
    save(out/'data/report.json',report);I.check_report(out/'data/report.json')
    lines=[f"# {search.group['id']}：Step4.2 恢复承载",'', '**PASS：每个 pose 的全部 32,768 个原始力／力矩需求通过。** 每条退出路径在自身地面上方，并完整离开原始支撑。', '', '绿色是相对 Step4.1 新恢复的材料，灰色是保留材料，青色显示前 100 mm 退出扫掠。切除与接受使用完整连续退出。', '', f"搜索 {search.evaluations} 个方向组合，实际构造检查 {search.exact_evaluations} 次；恢复材料 {report['restored_volume_cm3']:.2f} cm³。", '', '本阶段不要求连通，不要求保留原圆环，也未重建接地材料；不是完整共享夹具或强度接受。', '', '| Pose | 原始需求通过数 | 世界退出方向 |','|---|---:|---|']
    for row in rows:lines.append(f"| {row['pose']} | {row['force_covered']}/{row['load_count']} | {', '.join(f'{x:.3f}' for x in row['direction_world'])} |")
    (out/'README.md').write_text('\n'.join(lines)+'\n');return dict(id=search.group['id'],passed=True,proposals=search.evaluations,restored_volume_cm3=report['restored_volume_cm3'],seconds=report['seconds'])

def run(name,group):
    began=time.monotonic();search=Search(name,group)
    try:
        result=search.search()
        if result is None:
            save(search.out/'data/search.json',dict(complete=False,proposals=search.evaluations,trace=search.trace,best_counts=None if search.best is None else search.best['counts']))
            return dict(id=group['id'],passed=False,status='search_unresolved',proposals=search.evaluations,best_counts=None if search.best is None else search.best['counts'])
        row=finish(search,result,began);print('STEP4.2 PASS',row,flush=True);return row
    except Exception as error:
        traceback.print_exc();save(search.out/'data/error.json',dict(error=str(error),trace=traceback.format_exc()));return dict(id=group['id'],passed=False,error=str(error))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sets',nargs='+');parser.add_argument('--jobs',type=int,default=2);args=parser.parse_args();groups=json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets'];groups=[g for g in groups if not args.sets or g['id'] in args.sets]
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    with ProcessPoolExecutor(max_workers=args.jobs,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(run,'B',g) for g in groups];rows=[f.result() for f in futures]
    save(HERE/'output/B/data/step42_batch.json',dict(complete=all(r['passed'] for r in rows),sets=len(rows),passed_sets=sum(r['passed'] for r in rows),results=rows))
    print('STEP4.2 TOTAL',sum(r['passed'] for r in rows),'/',len(rows),flush=True)
if __name__=='__main__':main()
