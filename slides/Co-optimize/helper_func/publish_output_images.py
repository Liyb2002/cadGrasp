"""Publish stage images; Evaluation has one independent fixed image per set."""
from pathlib import Path
import argparse,hashlib,json,shutil,sys

COOPT=Path(__file__).resolve().parents[1];ROOT=COOPT.parents[1]
RESULT=COOPT/'data/results/B';VISUAL=COOPT/'output/B'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image_sources(label,experiment,include_evaluation=False):
    stage=RESULT/label/'step5/step5.3'/experiment
    if not(stage/'data/report.json').exists():stage=RESULT/label/'step5/step5.2'/experiment
    report=json.loads((stage/'data/report.json').read_text())
    if not report['complete']or not report['passed']or report.get('stage_type')=='support_pruning':
        raise ValueError('A completed final support is required: '+label)
    source=stage/'images'
    if not source.exists():source=VISUAL/label/'step5.3'
    images={p.name:p for p in source.glob('*.png')if p.name not in ['evaluation.png','static_occupancy.png']}
    if not images:raise FileNotFoundError('No saved stage images: '+str(stage))
    if include_evaluation:
        evaluation=json.loads((RESULT/label/'evaluation.json').read_text())
        if evaluation['step53_source']!=str(stage.relative_to(ROOT)):
            raise ValueError('Evaluation uses another complete support: '+label)
        if digest(VISUAL/label/'evaluation.png')!=evaluation['image_sha256']:
            raise ValueError('Changed fixed Evaluation image: '+label)
    return images


def publish_case_images(label,experiment,include_evaluation=False):
    images=image_sources(label,experiment,include_evaluation);destination=VISUAL/label/'step5.3'
    destination.mkdir(parents=True,exist_ok=True)
    for old in destination.glob('pose_*.png'):
        if old.name not in images:old.unlink()
    for name in ['evaluation.png','static_occupancy.png']:(destination/name).unlink(missing_ok=True)
    records=[]
    for name,source in images.items():
        target=destination/name
        if source.resolve()!=target.resolve():shutil.copy2(source,target)
        expected=digest(source)
        if digest(target)!=expected:raise ValueError('Image copy differs: '+str(target))
        records.append(dict(source=str(source.relative_to(ROOT)),path=str(target.relative_to(ROOT)),sha256=expected))
    return dict(id=label,images=records)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('experiment',nargs='?')
    args=parser.parse_args();sys.path.insert(0,str(COOPT/'evaluation'))
    from run import main as evaluate
    sys.argv=[str(COOPT/'evaluation/run.py'),'B']+(['--experiment',args.experiment]if args.experiment else[])
    evaluate()


if __name__=='__main__':main()
