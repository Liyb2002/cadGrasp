"""Contact sheets from each selected case's own certified result PNGs."""
from pathlib import Path
import argparse
import hashlib
import json
import math
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT=Path(__file__).resolve().parents[3]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    records=json.loads(args.manifest.read_text())
    rows=sorted(records['results'],key=lambda r:(r.get('pose_count',0),r['id']))
    args.output_dir.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    heading=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',25)
    sources={}
    for kind,key,width,height in [('all_supports','overview.png',400,390),
                                 ('all_comparisons','comparison.png',560,340)]:
        cols=7;top=65;count=math.ceil(len(rows)/cols)
        canvas=Image.new('RGB',(cols*width,count*height+top),'white');draw=ImageDraw.Draw(canvas)
        title='New Step5 | '+str(records['final_supports'])+' certified supports / '+str(records['group_count'])+' sets'
        if key=='comparison.png':title+=' | left: complete baseline, right: new result'
        draw.text((20,12),title,fill='#243745',font=heading)
        draw.text((20,43),'Each set framed separately; before/after pairs use one camera and scale.',fill='#526370',font=font)
        for i,row in enumerate(rows):
            x=(i%cols)*width;y=top+(i//cols)*height
            draw.rectangle((x+4,y+4,x+width-4,y+height-4),outline='#DDE3E8',width=1)
            label=row['id'];label=label[:48]+'\n'+label[48:]if len(label)>48 else label
            draw.multiline_text((x+12,y+10),label,fill='#243745',font=font,spacing=2)
            if not row['passed']:
                draw.text((x+18,y+height//2),'STEP4 FAIL - no certified support',fill='#A44545',font=font)
                continue
            pic=row['evaluation']['images'][key];path=ROOT/pic['path']
            actual=hashlib.sha256(path.read_bytes()).hexdigest()
            if actual!=pic['sha256']:raise ValueError('Selected image changed: '+str(path))
            panel=ImageOps.contain(Image.open(path).convert('RGB'),(width-20,height-80))
            canvas.paste(panel,(x+(width-panel.width)//2,y+57+(height-80-panel.height)//2))
            sources[str(path)]=actual
        canvas.save(args.output_dir/(kind+'.png'))
    (args.output_dir/'overview_sources.json').write_text(json.dumps(dict(
        input_manifest=str(args.manifest),source_images=sources,
        generated={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in args.output_dir.glob('all_*.png')}),indent=2)+'\n')


if __name__=='__main__':main()
