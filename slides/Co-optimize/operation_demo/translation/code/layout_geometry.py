"""Final-placement geometry; intermediate relocation occupancy is not carved."""
from pathlib import Path
import numpy as np
import co_common as R


class Layout:
    def __init__(self):
        self.base=R.HERE/'output/B/pose1+2+4+6'
        self.poses=['pose_1','pose_2','pose_4','pose_6']
        self.object_path=self.base/'step3/step3.1/registered_object.obj'
        self.wrap_path=self.base/'step3/step3.2/wrapped_support.obj'
        self.report_path=self.base/'step4/step4.1/data/report.json'
        self.obj=R.trimesh.load(self.object_path,force='mesh',process=False)
        self.wrap=R.trimesh.load(self.wrap_path,force='mesh',process=False)
        self.states=[R.state('B',p) for p in self.poses]
        self.transforms=[T for _,T,_ in self.states]
        report=R.json.loads(self.report_path.read_text())
        saved={row['pose']:row['direction_fixture'] for row in report['state_results']}
        self.directions=np.asarray([saved[p] for p in self.poses])
        self.object_solid=R.S.solid(self.obj)
        self.wrap_solid=R.S.solid(self.wrap)
        self.baseline_path=self.base/'step4/step4.1/data/initialization_final_support.obj'
        self.baseline=R.S.solid(R.trimesh.load(self.baseline_path,force='mesh',process=False))^self.wrap_solid
        self.exits=[R.S.solid(R.S.swept_solid(self.obj,.5*d,fan_in=8)) for d in self.directions]
        # Retain actual saved per-pose cut geometry at its corresponding placement.
        self.cut_paths=[self.base/f'step4/step4.1/data/{p}_own_removed.obj' for p in self.poses]
        self.cuts=[R.S.solid(R.trimesh.load(p,force='mesh',process=False)) for p in self.cut_paths]
        self.fixed_blockers=R.S.solid(R.S.unpack(R.union(self.exits[1:]+self.cuts[1:])))
        self.baseline=R.S.solid(R.S.unpack(self.baseline))
        T=self.transforms[0]
        self.distance=float(np.ptp(R.transform_points(self.obj.vertices,T)[:,0]))
        self.final_offset=T[:3,:3].T@np.array([self.distance,0.,0.])

    def construct(self,offsets):
        offsets=np.asarray(offsets)
        unique_offsets=np.unique(offsets,axis=0)
        if np.all(offsets==0):return self.baseline,self.exits
        seed=R.union([self.wrap_solid.translate(o/R.S.SCALE) for o in unique_offsets])
        exits=[s.translate(o/R.S.SCALE) for s,o in zip(self.exits,offsets)]
        cuts=[s.translate(o/R.S.SCALE) for s,o in zip(self.cuts,offsets)]
        remaining=seed-self.fixed_blockers-exits[0]-cuts[0]
        remaining=R.S.solid(R.S.unpack(remaining))
        # Explicitly carve final body occupancy; nominal sweeps can lose slivers.
        bodies=[self.object_solid.translate(o/R.S.SCALE) for o in unique_offsets]
        for body in bodies:remaining=remaining-body
        for _ in range(3):
            overlaps=[R.material_volume(remaining^body) for body in bodies]
            if max(overlaps)<=1e-10:break
            for body,value in zip(bodies,overlaps):
                if value>1e-10:remaining=remaining-body
        self.final_body_overlap_m3=max(R.material_volume(remaining^body) for body in bodies)
        return remaining,exits
