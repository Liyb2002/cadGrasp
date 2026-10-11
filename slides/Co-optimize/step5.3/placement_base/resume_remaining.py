"""Continue checkpointed geometry, rejecting impossible contact cores early."""
import run as R
G=R.G
original_boundary=G.K.contact_boundary
original_construction=R.construction_attempts
original_builder=G.Construction


def build_with_ground_bound(data,tasks,body,padding):
    first=R.C.np.linalg.inv(data['rows'][0]['fixture_transform'])@data['rows'][0]['object_transform']
    bound=None
    def boundary(body,local,allowed):
        nonlocal bound
        if bound is None:
            common=local.copy();common.apply_transform(first)
            bound,_=R.ground_coverage(common,data)
        if not bound['passed']:
            return R.C.np.empty((0,3,3)),R.C.np.empty(0,int)
        return original_boundary(body,local,allowed)
    G.K.contact_boundary=boundary
    try:builder=original_builder(data,tasks,body,padding=padding)
    finally:G.K.contact_boundary=original_boundary
    builder.relaxed_nominal_ground_bound=bound
    return builder


def continued(data,tasks,body,trials,out):
    previous=R.C.COOPT/'output/B'/data['case']['id']/'step5/step5.3/placement_base_resume_v1/construction_attempts.json'
    if not previous.exists():return original_construction(data,tasks,body,trials,out)
    saved=R.C.json.loads(previous.read_text());done={a['proposal'] for a in saved if a['padding_m']==.14}
    # Keep the original deterministic ranking; remove only complete pairs.
    feasible=[t for t in trials if t['seating']['passed'] and t['label']!='assembly_z_necessary']
    required=[t for t in feasible if t['label'] in ['z_xy_5mm','z_xyz_5mm']]
    ranked=sorted(feasible,key=lambda t:(-sum(p['passed'] for p in t['ground_screen']['per_pose']),
        t['seating']['maximum_displacement_coordinate_mm'],t['label']))
    selected=required+[t for t in ranked if t not in required][:2]
    remaining=[t for t in selected if t['label'] not in done]
    fresh,accepted,best=original_construction(data,tasks,body,remaining,out)
    R.C.save(out/'resumed_attempt_provenance.json',dict(previous_artifact=R.C.relative(previous),
        previous_sha256=R.C.digest(previous),reused_attempt_count=len(saved),fresh_attempt_count=len(fresh),
        contact_cores_skipped_only_when_relaxed_nominal_ground_hull_already_failed=True))
    return saved+fresh,accepted,best


if __name__=='__main__':
    R.STAGE='step5.3/placement_base_resume_v3'
    G.Construction=build_with_ground_bound
    R.construction_attempts=continued
    R.main()
