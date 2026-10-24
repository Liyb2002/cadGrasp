"""Review actual encoded post-processing frames, including the one-second start."""
from pathlib import Path
import hashlib,json,subprocess
from PIL import Image
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'slides/Co-optimize').exists())
HERE=ROOT/'slides/Co-optimize/idea/post_processing'
DATA=HERE/'assets/data/video_wrap_surface_reinforced_all52_20261015_03'
records=[]
for number,name,indices in [(1,'01_hulls_and_support.mp4',[0,23,24,40,64,88,112,127]),
                            (2,'02_pruning_and_bbx.mp4',[0,24,48,72,96,120,156,191])]:
    folder=DATA/'encoded_review'/str(number);folder.mkdir(parents=True,exist_ok=True)
    expression='+'.join('eq(n\\,'+str(i)+')' for i in indices)
    movie=HERE/name
    subprocess.run(['ffmpeg','-y','-v','error','-i',str(movie),'-vf','select='+expression+',scale=800:450',
                    '-fps_mode','vfr',str(folder/'frame_%03d.png')],check=True)
    frames=sorted(folder.glob('frame_*.png'));assert len(frames)==len(indices)
    board=Image.new('RGB',(1600,1800),'white')
    for k,path in enumerate(frames):
        with Image.open(path) as im:board.paste(im,((k%2)*800,(k//2)*450))
    sheet=DATA/f'0{number}_encoded_review.png';board.save(sheet)
    records.append(dict(movie=str(movie.relative_to(ROOT)),sha256=hashlib.sha256(movie.read_bytes()).hexdigest(),
                        frames=indices,sheet=str(sheet.relative_to(ROOT)),sheet_sha256=hashlib.sha256(sheet.read_bytes()).hexdigest()))
(DATA/'encoded_review.json').write_text(json.dumps(dict(complete=True,results=records),indent=2)+'\n')
print('POST_ENCODED_REVIEW_COMPLETE')
