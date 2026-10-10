"""Franka FK/IK and exact triangle-solid hand collision screening.

IK uses the official Menagerie Panda joint limits; collision Booleans use
its collision meshes, rather than MuJoCo's convex hull of the object.
"""
from pathlib import Path
import numpy as np
import mujoco
import trimesh
from scipy.spatial.transform import Rotation
from co_common import S, G, material_volume

MODEL = Path(__file__).parent/'assets/franka_emika_panda/panda.xml'

class Panda:
    def __init__(self):
        self.model=mujoco.MjModel.from_xml_path(str(MODEL))
        self.data=mujoco.MjData(self.model)
        self.hand=self.model.body('hand').id
        self.lower=self.model.jnt_range[:7,0];self.upper=self.model.jnt_range[:7,1]
        self.home=np.array([0.,0.,0.,-1.57079,0.,1.57079,-.7853])
        self.geometry=[]
        for g in range(self.model.ngeom):
            if self.model.geom_group[g]!=3:continue
            typ=self.model.geom_type[g]
            if typ==mujoco.mjtGeom.mjGEOM_MESH:
                m=self.model.geom_dataid[g];v=self.model.mesh_vertadr[m];f=self.model.mesh_faceadr[m]
                mesh=trimesh.Trimesh(self.model.mesh_vert[v:v+self.model.mesh_vertnum[m]].copy(),self.model.mesh_face[f:f+self.model.mesh_facenum[m]].copy(),process=True)
            elif typ==mujoco.mjtGeom.mjGEOM_BOX:
                mesh=trimesh.creation.box(2*self.model.geom_size[g])
            else:continue
            if not mesh.is_watertight:mesh=mesh.convex_hull
            body=self.model.body(self.model.geom_bodyid[g]).name
            self.geometry.append((g,body,typ==mujoco.mjtGeom.mjGEOM_BOX,mesh))

    def ik(self,T,start=None):
        best=None
        for seed in [self.home if start is None else start,self.home+np.array([.5,.3,-.5,0,.5,0,0])]:
            q=np.clip(seed,self.lower+1e-5,self.upper-1e-5)
            for _ in range(160):
                self.data.qpos[:7]=q;mujoco.mj_forward(self.model,self.data)
                p=self.data.xpos[self.hand];R=self.data.xmat[self.hand].reshape(3,3)
                err=np.r_[T[:3,3]-p,Rotation.from_matrix(T[:3,:3]@R.T).as_rotvec()]
                if np.linalg.norm(err[:3])<.0005 and np.linalg.norm(err[3:])<.005:
                    return q.copy()
                jp=np.zeros((3,self.model.nv));jr=jp.copy()
                mujoco.mj_jacBody(self.model,self.data,jp,jr,self.hand)
                jac=np.vstack([jp[:,:7],jr[:,:7]])
                dq=jac.T@np.linalg.solve(jac@jac.T+.001*np.eye(6),err)
                q=np.clip(q+np.clip(dq,-.15,.15),self.lower+1e-5,self.upper-1e-5)
        return best

    def hand_parts(self,T,width):
        # width is the central fingertip contact-plane separation. The largest
        # pad's inner plane is 1.5mm outside each finger joint origin.
        self.data.qpos[:7]=self.home;self.data.qpos[7:9]=np.clip((width-.003)/2,0,.04)
        mujoco.mj_forward(self.model,self.data)
        H=np.eye(4);H[:3,:3]=self.data.xmat[self.hand].reshape(3,3);H[:3,3]=self.data.xpos[self.hand]
        root=T@np.linalg.inv(H)
        result=[]
        for g,body,pad,mesh in self.geometry:
            if body not in ('hand','left_finger','right_finger'):continue
            P=np.eye(4);P[:3,:3]=self.data.geom_xmat[g].reshape(3,3);P[:3,3]=self.data.geom_xpos[g]
            moved=mesh.copy();moved.apply_transform(root@P)
            result.append(dict(geom=int(g),body=body,pad=pad,mesh=moved,solid=S.solid(moved)))
        return result

    def hand_solids(self,T,width):
        return [(p['body'],p['pad'],p['solid']) for p in self.hand_parts(T,width)]

    def collision_report(self,T,width,obstacle,allow_pads=False):
        hits=[]
        for part in self.hand_parts(T,width):
            if allow_pads and part['pad']:continue
            overlap=material_volume(part['solid']^obstacle)
            if overlap>1e-11:
                hits.append(dict(geom=part['geom'],body=part['body'],pad=part['pad'],overlap_mm3=overlap*1e9))
        return hits

    def approach_report(self,T,width,obstacle):
        for distance in np.linspace(.08,0,9):
            P=T.copy();P[:3,3]-=distance*T[:3,2]
            hits=self.collision_report(P,.08,obstacle)
            if hits:return dict(passed=False,stage='approach',distance_m=float(distance),gap_m=.08,hits=hits)
        for opening in np.linspace(.08,width,7):
            hits=self.collision_report(T,opening,obstacle,allow_pads=True)
            if hits:return dict(passed=False,stage='closure',distance_m=0.,gap_m=float(opening),hits=hits)
        return dict(passed=True)

    def object_contacts(self,T,width,obstacle):
        hits=[]
        for part in self.hand_parts(T,max(.003,width-.0004)):
            if not part['pad']:continue
            overlap=material_volume(part['solid']^obstacle)
            if overlap>1e-13:hits.append(dict(body=part['body'],geom=part['geom'],overlap_mm3=overlap*1e9))
        return dict(passed={h['body'] for h in hits}=={'left_finger','right_finger'},hits=hits,closure_compression_m=.0004)

    def center_on_first_contacts(self,T,obstacle):
        """Center asymmetric local shape before symmetric jaw closure.

        Find each jaw's first intersection separately with fixed orientation;
        translate the hand along its closing axis so both contacts occur at
        the same opening. 0.4mm total compression is a numerical contact probe.
        """
        parts=[p for p in self.hand_parts(T,.08) if p['pad']]
        axis=T[:3,1];thresholds={}
        for body,sign in [('left_finger',1),('right_finger',-1)]:
            pads=[p['solid'] for p in parts if p['body']==body]
            def touched(gap):
                move=sign*(gap-.08)/2*axis/S.SCALE
                return any(material_volume(p.translate(move.tolist())^obstacle)>1e-13 for p in pads)
            previous=.08
            if touched(previous):return dict(passed=False,reason='pads_blocked_at_max_opening')
            bracket=None
            for gap in np.linspace(.08,.003,21)[1:]:
                if touched(gap):bracket=(gap,previous);break
                previous=gap
            if bracket is None:return dict(passed=False,reason='jaw_never_contacts',body=body)
            lower,upper=bracket
            for _ in range(15):
                middle=(lower+upper)/2
                if touched(middle):lower=middle
                else:upper=middle
            thresholds[body]=(lower+upper)/2
        left,right=thresholds['left_finger'],thresholds['right_finger']
        center=(left-right)/4;gap=(left+right)/2
        if gap<.003:return dict(passed=False,reason='below_minimum_central_pad_gap')
        adjusted=T.copy();adjusted[:3,3]+=center*axis
        return dict(passed=True,hand=adjusted,width=gap,first_contact_gaps=thresholds,center_shift_m=center,compression_m=.0004)

    def contact_patches(self,T,width,mesh):
        points=[];normals=[];sources=[];bodies=[]
        for part in self.hand_parts(T,width):
            if not part['pad']:continue
            box=part['mesh'];planes=np.unique(np.round(np.c_[box.face_normals,-np.einsum('ij,ij->i',box.face_normals,box.triangles[:,0])],12),axis=0)
            for index in mesh.triangles_tree.intersection(box.bounds.ravel()):
                poly=mesh.triangles[index].copy()
                for plane in planes:
                    poly=G.clip_plane(poly,plane)
                    if len(poly)<3:break
                if len(poly)<3:continue
                area=sum(np.linalg.norm(np.cross(poly[k]-poly[0],poly[k+1]-poly[0]))/2 for k in range(1,len(poly)-1))
                if area<1e-12:continue
                points.extend(poly);normals.extend([-mesh.face_normals[index]]*len(poly));sources.extend([index]*len(poly));bodies.extend([part['body']]*len(poly))
        return dict(points=np.asarray(points).reshape(-1,3),normals=np.asarray(normals).reshape(-1,3),sources=np.asarray(sources,int),bodies=bodies)

    def clear(self,T,width,obstacle,allow_pads=False):
        for body,pad,solid in self.hand_solids(T,width):
            if allow_pads and pad:continue
            if material_volume(solid^obstacle)>1e-11:return False
        return True

    def work_contact_clear(self,T,width,work_mesh):
        # Clip work triangles against actual fingertip boxes. Pair centers
        # outside work faces alone do not guarantee the whole pad is clear.
        for body,pad,solid in self.hand_solids(T,max(.003,width-.0004)):
            if not pad:continue
            box=S.unpack(solid);lower,upper=box.bounds
            normals=box.face_normals
            planes=np.unique(np.round(np.c_[normals,-np.einsum('ij,ij->i',normals,box.triangles[:,0])],12),axis=0)
            for index in work_mesh.triangles_tree.intersection(np.r_[lower,upper]):
                poly=work_mesh.triangles[index].copy()
                for plane in planes:
                    poly=G.clip_plane(poly,plane)
                    if len(poly)<3:break
                if len(poly)>=3 and sum(np.linalg.norm(np.cross(poly[k]-poly[0],poly[k+1]-poly[0])) for k in range(1,len(poly)-1))>2e-12:
                    return False
        return True

    def approach_clear(self,T,width,obstacle):
        for distance in np.linspace(.08,0,9):
            P=T.copy();P[:3,3]-=distance*T[:3,2]
            if not self.clear(P,.08,obstacle):return False
        for opening in np.linspace(.08,width,7):
            if not self.clear(T,opening,obstacle,allow_pads=True):return False
        return True

    def arm_report(self,q,width,world_obstacle):
        self.data.qpos[:7]=q;self.data.qpos[7:9]=np.clip((width-.003)/2,0,.04)
        mujoco.mj_forward(self.model,self.data)
        for contact in self.data.contact:
            if contact.dist < -.0001:
                return dict(passed=False,reason='self_collision',bodies=[self.model.body(self.model.geom_bodyid[g]).name for g in (contact.geom1,contact.geom2)],depth_m=float(-contact.dist))
        for g,body,pad,mesh in self.geometry:
            if pad:continue
            P=np.eye(4);P[:3,:3]=self.data.geom_xmat[g].reshape(3,3);P[:3,3]=self.data.geom_xpos[g]
            moved=mesh.copy();moved.apply_transform(P)
            # link0 is the bolted stationary base. Its upstream collision mesh
            # extends 32.5 micrometres below its mounting plane; treating that
            # mounting contact as a moving-link floor collision rejects all IK.
            if body!='link0' and moved.vertices[:,2].min() < -1e-6:
                return dict(passed=False,reason='floor',body=body,min_z_m=float(moved.vertices[:,2].min()))
            overlap=material_volume(S.solid(moved)^world_obstacle)
            if overlap>1e-11:return dict(passed=False,reason='object',body=body,overlap_mm3=overlap*1e9)
        return dict(passed=True)

    def arm_clear(self,q,width,world_obstacle):
        return self.arm_report(q,width,world_obstacle)['passed']

    def carry_path(self,hand,width,assembly,transforms):
        """Lift, rotate/translate at a common height, lower; sampled checks."""
        from scipy.spatial.transform import Slerp
        q=self.home.copy();records=[]
        station=np.array([.45,0.,.15])
        for index,(first,second) in enumerate(zip(transforms,transforms[1:])):
            high=max(first[2,3],second[2,3])+.25
            A=first.copy();B=second.copy();A[2,3]=B[2,3]=high
            waypoints=[first,A,B,second]
            for stage,(start,end) in enumerate(zip(waypoints,waypoints[1:])):
                rotations=Slerp([0,1],Rotation.from_matrix([start[:3,:3],end[:3,:3]]))
                for fraction in np.linspace(0,1,7):
                    T=np.eye(4);T[:3,:3]=rotations(fraction).as_matrix();T[:3,3]=(1-fraction)*start[:3,3]+fraction*end[:3,3]+station
                    q=self.ik(T@hand,q)
                    if q is None:return dict(passed=False,reason='path_ik',transition=index,stage=stage,fraction=float(fraction))
                    placed=assembly.copy();placed.apply_transform(T)
                    if not self.arm_clear(q,width,S.solid(placed)):
                        return dict(passed=False,reason='path_collision',transition=index,stage=stage,fraction=float(fraction))
                    records.append(dict(T=T.tolist(),q=q.tolist()))
        return dict(passed=True,samples=records,continuous_collision_certificate=False)

    def loading_path(self,hand,width,mesh,fixture,T,direction):
        q=self.home.copy();records=[]
        fixed=fixture.copy();world=T.copy();world[:3,3]+=[.45,0,.15]
        fixed.apply_transform(world)
        for distance in np.linspace(.20,0,13):
            moved=mesh.copy();move=np.eye(4);move[:3,3]=distance*direction
            moved.apply_transform(world@move)
            local=hand.copy();local[:3,3]+=distance*direction
            q=self.ik(world@local,q)
            if q is None:return dict(passed=False,reason='loading_ik',distance=float(distance))
            if not self.arm_clear(q,width,S.solid(moved)+S.solid(fixed)):
                return dict(passed=False,reason='loading_arm_collision',distance=float(distance))
            if distance==.20 and material_volume(S.solid(moved)^S.solid(fixed))>1e-11:
                return dict(passed=False,reason='loading_start_not_outside_fixture')
            records.append(dict(distance=float(distance),q=q.tolist()))
        return dict(passed=True,samples=records,continuous_collision_certificate=False)
