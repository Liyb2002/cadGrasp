"""Concurrent hybrid with physics-selected exact contact tangent proposals."""
from hybrid_fast import *
from physics_guided_contact_planes import ranked_contact_planes


class ContactPlaneSearch(FastHybridSearch):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.contact_plane_trace=[]
        self.additional_code += [Path(__file__),HERE/'helper_func/physics_guided_contact_planes.py']
        self.additional_artifacts += ['contact_plane_proposals.json']

    def ordered_common(self,axes):
        covering=super().ordered_common(axes)
        if self.best is None:return covering
        weights,_=self.dual_weights(self.best)
        if weights.sum()<1e-15:return covering
        ranked,record=ranked_contact_planes(self,self.best,weights,limit=32)
        self.contact_plane_trace.append(record)
        save(self.out/'contact_plane_proposals.json',self.contact_plane_trace)
        # Retain the entire original covering bank and its bounded local stream.
        return [(float(-cost),directions,None) for cost,directions in ranked]+covering

    def finish(self,*args,**kwargs):
        save(self.out/'contact_plane_proposals.json',self.contact_plane_trace)
        self.report_extra.update(contact_plane_sampling=True,
            contact_plane_policy='up to 32 physics-selected tangent proposals per covering bank; full nominal trajectory ranking; all original covering directions retained')
        return super().finish(*args,**kwargs)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object',default='B');parser.add_argument('--set',required=True)
    parser.add_argument('--directions',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--iterations',type=int,default=12);parser.add_argument('--max-proposals',type=int,default=1200)
    args=parser.parse_args();source=ROOT/'objects'/args.object
    groups=json.loads((source/'pose_sets.json').read_text())['sets']
    if (source/'illegal_pose_sets.json').exists():
        groups += [dict(g,id='illegal/'+g['id']) for g in json.loads((source/'illegal_pose_sets.json').read_text())['sets']]
    group=next(g for g in groups if g['id']==args.set)
    search=ContactPlaneSearch(args.object,group,out=args.out,directions=args.directions,max_proposals=args.max_proposals)
    try:search.optimize(args.iterations)
    except (RuntimeError,ValueError) as error:
        save(search.out/'unresolved.json',dict(status='numerically_unresolved',passed=False,pose_set=group['id'],error=str(error)));raise

if __name__=='__main__':main()
