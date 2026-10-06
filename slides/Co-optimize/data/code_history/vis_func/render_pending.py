"""Publish honest pending views without claiming previous acceptance."""
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'helper_func'))
import _bootstrap
from co_common import *
def render(group):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    out=HERE/'output/B'/group['id']/'step4/step4.2';p=out/'data/report.json';r=json.loads(p.read_text())
    renderer_key=next(iter(I.hashes([HERE/'vis_func/render_pending.py'])))
    I.check_hashes(r['provenance']['inputs']);I.check_hashes({k:v for k,v in r['provenance']['code'].items() if k!=renderer_key})
    for filename,digest in r.get('artifacts',{}).items():assert I.sha256(p.parent/filename)==digest
    old_renderer=r['provenance']['code'].get(renderer_key)
    if old_renderer and old_renderer!=I.sha256(HERE/'vis_func/render_pending.py'):r.setdefault('presentation_revisions',[]).append(dict(previous_renderer_sha256=old_renderer,geometry_changed=False,force_records_changed=False))
    if r.get('connectivity_required') and r['passed']:return
    withdrawn=group['id']=='pose1+4+7+12+21+27';r['passed']=False;r['connectivity_required']=True;r['status']='contact_corrected_search_unresolved' if withdrawn else 'force_exit_pass_connectivity_unresolved'
    if withdrawn:
        for row in r['state_results']:
            if row.get('force_covered') is not None:row['historical_force_covered_with_invalid_remnants']=row['force_covered']
            row['force_passed']=False;row['force_covered']=None
    support=trimesh.load(out/'remaining_support.obj',force='mesh',process=False);cols=min(3,len(group['poses']));nr=(len(group['poses'])+cols-1)//cols;fig=plt.figure(figsize=(6*cols,6*nr))
    for i,row in enumerate(r['state_results']):
        task,T,mesh=state('B',row['pose']);m=support.copy();m.apply_transform(T);ax=fig.add_subplot(nr,cols,i+1,projection='3d')
        ax.add_collection3d(Poly3DCollection(m.triangles*1000,facecolor='#929292',edgecolor='none',alpha=.85));ax.add_collection3d(Poly3DCollection(task.domain.mesh.triangles*1000,facecolor='#74add1',edgecolor='none',alpha=.25))
        v=np.vstack([m.vertices,task.domain.mesh.vertices])*1000;c=(v.min(0)+v.max(0))/2;rad=np.ptp(v,axis=0).max()*.6
        for axis,x in zip('xyz',c):getattr(ax,'set_'+axis+'lim')(x-rad,x+rad)
        ax.set_box_aspect((1,1,1));ax.view_init(22,-55);ax.set_axis_off();ax.set_title(row['pose'].replace('_',' ').title())
    title='Step4.2 unresolved: previous force acceptance withdrawn' if withdrawn else 'Step4.2 unresolved: original loads and exits pass; connectivity pending'
    fig.suptitle(title,fontsize=14);fig.subplots_adjust(left=0,right=1,bottom=0,top=.93,wspace=0,hspace=.05);fig.savefig(out/'overview.png',dpi=150,bbox_inches='tight');plt.close(fig)
    r['provenance']['code'].update(I.hashes([HERE/'vis_func/render_pending.py']));r['artifacts']['../overview.png']=I.sha256(out/'overview.png');save(p,r);I.check_report(p)
    (out/'README.md').write_text(f"# {group['id']}：Step4.2 未完成\n\n"+('旧的承载通过记录已撤销：共面残片不能计作与退出方向冲突的接触。过滤后重新搜索仍未找到完整可行结果；当前不作承载通过声明。' if withdrawn else '全部原始承载需求与退出条件通过，材料尚未连通。逐块删除检查已删除可删碎块；剩下两个互补必要块尚未连上。')+'\n\n当前模型和图片是此前搜索状态，不能当作共同优化已通过的结果。\n')
if __name__=='__main__':
    for g in json.loads((ROOT/'objects/B/pose_sets.json').read_text())['sets']:
        if g['id'] in ['pose1+6+11+13+14+17','pose1+4+7+12+21+27']:render(g)
