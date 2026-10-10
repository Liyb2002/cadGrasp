import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'helper_func'))
import _bootstrap
import numpy as np
from computed_packing import minimum_axis_layout, small_motion_path
from physics_guided_geometry import tangent_frames
from global_placement_sampling import select_tool_finalists
from contact_event_steps import event_candidates
from types import SimpleNamespace

class ComputedPackingTests(unittest.TestCase):
    def test_layout_separates_each_support_from_other_sweeps_in_native_floor(self):
        rng=np.random.default_rng(9)
        normals=rng.normal(size=(6,3));normals/=np.linalg.norm(normals,axis=1)[:,None]
        frames=tangent_frames(normals)
        pieces=np.tile(np.array([[-.025,-.02,-.01],[.025,.02,.01]]),(6,1,1))
        obstacles=np.tile(np.array([[-.03,-.03,-.03],[.03,.03,.3]]),(6,1,1))
        offsets,_=minimum_axis_layout(pieces,obstacles,frames,rng,trials=12)
        np.testing.assert_allclose(offsets[0],0)
        np.testing.assert_allclose(np.sum(normals*offsets,axis=1),0,atol=1e-12)
        for i in range(6):
            for j in range(6):
                if i==j:continue
                a=pieces[i]+offsets[i];b=obstacles[j]+offsets[j]
                self.assertTrue(np.any(a[0]>=b[1]+.005-1e-8) or np.any(b[0]>=a[1]+.005-1e-8))

    def test_coincident_other_exit_can_generate_outward_translation_event(self):
        search=SimpleNamespace(n=2,points=np.array([[0.,0.,0.]]),outward=np.array([[1.,0.,0.]]),
            sources=np.array([0]),rays=[np.array([[1.,0.,0.,0.,0.,0.,0.]])],normals=np.array([[0.,0.,1.],[0.,0.,1.]]))
        search.frames=tangent_frames(search.normals)
        search.refresh_targets=lambda current:[(0,0,None)]
        search.projection=lambda current,k,i:{'loss':1.,'dual':np.array([-1.,0.,0.,0.,0.,0.,0.])}
        search.boundary=SimpleNamespace(shadows=lambda direction:(SimpleNamespace(intersection=lambda box:[]),[]))
        directions=np.array([[0.,0.,1.],[1.,0.,0.]])
        rows=event_candidates(search,{},directions,np.zeros((2,3)),limit=1)
        translations=[r for r in rows if r['kind']=='translation-contact-event']
        self.assertEqual(len(translations),1)
        np.testing.assert_allclose(translations[0]['offsets'][0],0.)
        self.assertLess(translations[0]['offsets'][1,0],0.)
        self.assertGreaterEqual((translations[0]['offsets'][0]-translations[0]['offsets'][1])[0],1e-5-1e-10)

    def test_tools_receive_trials_despite_incompatible_guidance_scales(self):
        def row(kind,raw,loss):
            return ((raw,),kind,None,None,{'loss':loss,'sum_loss':loss},{'guidance_before':raw*2,'guidance_after':raw})
        direction=row('recovery-direction',1e-9,.8)
        better_direction=row('direction',100.,.2)
        translation=row('translation',1000.,.3)
        chosen=select_tool_finalists([direction,better_direction,translation],2)
        self.assertEqual([r[1] for r in chosen],['direction','translation'])
        self.assertIs(select_tool_finalists([direction,better_direction,translation],1)[0],better_direction)

    def test_motion_path_preserves_endpoints_and_increment_limits(self):
        start=np.array([[0.,0.,1.],[0.,0.,1.]])
        angle=.42;end=np.array([[0.,0.,1.],[np.sin(angle),0.,np.cos(angle)]])
        offset=np.array([[0.,0.,0.],[.071,-.025,0.]])
        path=small_motion_path(start,end,offset)
        ds=np.array([row['directions'] for row in path]);ts=np.array([row['offsets_m'] for row in path])
        np.testing.assert_allclose(ds[0],start);np.testing.assert_allclose(ds[-1],end)
        np.testing.assert_allclose(ts[-1],offset);np.testing.assert_allclose(np.linalg.norm(ds,axis=2),1,atol=1e-12)
        self.assertLessEqual(np.linalg.norm(np.diff(ts,axis=0),axis=2).max(),.001+1e-12)
        self.assertLessEqual(np.arccos(np.clip(np.sum(ds[1:]*ds[:-1],axis=2),-1,1)).max(),np.deg2rad(1)+1e-12)

if __name__=='__main__':unittest.main()
