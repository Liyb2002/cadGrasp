"""Presentation-only regeneration; no geometry or acceptance change."""
from pathlib import Path
import numpy as np
from step3_scheculer import operation_dsl as F,contacts as I

def render_geometry(group,tasks,state,witness):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import trimesh
    from step4_connect_support import build_coupled_saddle as S,boxed_support as B
    mesh=trimesh.load(witness.parent/'shape.obj',force='mesh',process=False)
    _,owners,_=F.roots(tasks,state)
    objects=[t.domain.mesh.vertices@b+o for t,b,o in zip(tasks,state.bases,state.offsets)]
    clouds=np.vstack([mesh.vertices]+objects);center=clouds.mean(axis=0);radius=np.ptp(clouds,axis=0).max()*.55
    fig=plt.figure(figsize=(16,4));titles=['Contact regions and reaction normals','Absolute object exit directions','Connected fixture','Fixture with one object pose']
    for panel,title in enumerate(titles):
        ax=fig.add_subplot(1,4,panel+1,projection='3d');ax.set_title(title,fontsize=10)
        if panel in (0,1):
            for index,(owner,cells,contact) in enumerate(owners.values()):
                ax.add_collection3d(Poly3DCollection(contact,facecolors=plt.get_cmap('tab20')(index%20),edgecolors='none'))
        if panel in (2,3):ax.add_collection3d(Poly3DCollection(mesh.triangles,facecolors='#b4b9bd',edgecolors='none',alpha=1.))
        if panel==0:
            for k,(t,row,b,o) in enumerate(zip(tasks,state.groups,state.bases,state.offsets)):
                points=np.vstack([c['triangles_m'].reshape(-1,3) for c in row])@b+o
                normal=sum((-t.domain.mesh.face_normals[c['source_faces']]*c['triangle_areas_m2'][:,None]).sum(axis=0) for c in row)@b
                normal/=max(np.linalg.norm(normal),1e-15)
                start=points.mean(axis=0)
                ax.quiver(*start,*normal,length=radius*.3,color=plt.get_cmap('tab10')(k),arrow_length_ratio=.2)
        if panel==1:
            for k,(t,vertices,path,b) in enumerate(zip(tasks,objects,state.paths,state.bases)):
                ax.add_collection3d(Poly3DCollection(vertices[t.domain.mesh.faces],facecolors=plt.get_cmap('tab10')(k),edgecolors='none',alpha=.13))
                start=vertices.mean(axis=0);direction=np.asarray(path['initial_object_exit_world'])@b
                ax.quiver(*start,*direction,length=radius*.45,color=plt.get_cmap('tab10')(k),arrow_length_ratio=.2)
                ax.text(*start,t.pose,fontsize=7)
        if panel==3:
            ax.add_collection3d(Poly3DCollection(objects[0][tasks[0].domain.mesh.faces],facecolors='#77acd0',edgecolors='none',alpha=.32))
        if panel in (2,3):
            from shapely.geometry import MultiPoint
            from step0_pose_selection.floor_points import pressure_centers
            for t,b,o in zip(tasks,state.bases,state.offsets):
                points=pressure_centers(t.targets/t.scale,t.domain.com)[0]
                hull=MultiPoint(points).convex_hull
                if hull.geom_type=='Polygon':
                    xy=np.asarray(hull.exterior.coords);v=np.c_[xy,np.zeros(len(xy))]@b+o
                    ax.plot(*v.T,color='#b67f3d',linewidth=1.)
        ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[1]-radius,center[1]+radius),zlim=(center[2]-radius,center[2]+radius))
        ax.set_box_aspect((1,1,1));ax.set_axis_off();ax.view_init(elev=25,azim=-60)
    fig.suptitle(f'{group.name}: qualified common-cavity support; head count is not optimized',fontsize=12)
    fig.tight_layout();fig.savefig(group/'step4/data'/F.BODY_STAGE/'construction.png',dpi=150);plt.close(fig)


def publish(groups):
    import shutil,json
    from step3_scheculer.review_operation_dsl import load
    from step3_scheculer.run_dsl import saved_task
    for group in groups:
        name='dsl_cavity'
        stage=group/'step3_scheculer'/name
        F.BODY_STAGE=name+'_support'
        r=I.check_report(stage/'report.json');tasks=[saved_task(group,p) for p in r['poses']]
        state,_=load(stage/'final',tasks)
        output=group/'step4/data'/F.BODY_STAGE;root=group/'step4'
        body=I.check_report(output/'report.json')
        render_geometry(group,tasks,state,I.ROOT/r['witness'])
        body['artifacts']['construction.png']=I.sha256(output/'construction.png')
        body['presentation_only_update']=True;body['provenance']['code'].update(I.hashes([Path(__file__)]))
        I.save(output/'report.json',body)
        shutil.copy2(output/'construction.png',root/'construction_steps.png')
        public=json.loads((root/'report.json').read_text())
        public['artifacts']['construction_steps.png']=I.sha256(root/'construction_steps.png')
        public['presentation_only_update']=True;public['provenance']['code'].update(I.hashes([Path(__file__)]))
        public['provenance']['inputs']=I.hashes([output/'report.json'])
        I.save(root/'report.json',public)
        I.save(root/'data/report.json',dict(public,artifacts={'../'+k:v for k,v in public['artifacts'].items()}))
        metric=json.loads((group/'step5_evaluate/report.json').read_text())
        for name in list(metric['provenance']['inputs']):metric['provenance']['inputs'][name]=I.sha256(I.ROOT/name)
        I.save(group/'step5_evaluate/report.json',metric)
        I.check_report(group/'step5_evaluate/report.json')
