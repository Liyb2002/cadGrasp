"""Clone immutable numerical dependencies and overlay specified controllers."""
from pathlib import Path
import argparse,hashlib,json,shutil

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'slides/Co-optimize/data/results/B'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('experiment');parser.add_argument('--parent',required=True)
    parser.add_argument('--overlays',nargs='+',required=True);args=parser.parse_args()
    old=BASE/'data'/('system_'+args.parent);new=BASE/'data'/('system_'+args.experiment)
    manifest=json.loads((old/'frozen_sources.json').read_text())['files']
    assert all(sha(ROOT/r['frozen'])==r['frozen_sha256']for r in manifest)
    new.mkdir(exist_ok=False);oldtree=old/'frozen_sources';newtree=new/'frozen_sources'
    shutil.copytree(oldtree,newtree,ignore=shutil.ignore_patterns('__pycache__'))
    for name in args.overlays:
        target=newtree/'slides/Co-optimize'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():target.chmod(0o644)
        shutil.copyfile(ROOT/'slides/Co-optimize'/name,target)
    records=[]
    for path in newtree.rglob('*.py'):
        path.chmod(0o644);path.write_text(path.read_text().replace(str(oldtree),str(newtree)));path.chmod(0o444)
        relative=path.relative_to(newtree);src=ROOT/relative
        records.append(dict(original=str(relative),original_sha256=sha(src)if src.exists()else None,
            frozen=str(path.relative_to(ROOT)),frozen_sha256=sha(path),io_adaptations=['snapshot path relocation']))
    (new/'frozen_sources.json').write_text(json.dumps(dict(complete=True,files=records,source_count=len(records),
        physics_changed=False,parent=args.parent,controller_overlays=args.overlays),indent=2)+'\n')
    for filename in ['requested_sets.json','protected_originals.json','previous_step4_results.json']:
        shutil.copy2(old/filename,new/filename)
    print('FROZEN',args.experiment,len(records))


if __name__=='__main__':main()
