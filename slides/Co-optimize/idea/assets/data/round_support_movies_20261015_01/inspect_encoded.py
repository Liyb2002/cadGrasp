"""Extract actual encoded frames covering tasks, turns and uniform time samples."""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
from PIL import Image,ImageDraw

HERE=Path(__file__).resolve().parent
stage=HERE/'staging'
selection=json.loads((stage/'assets/selection.json').read_text())
records=[]
for case in selection['results']:
    label=case['id'];movie=stage/(label+'_franka.mp4')
    segments=json.loads((stage/'assets/data'/label/'segments.json').read_text())
    indices=set(range(0,480,30))|{479};events={}
    for segment in segments['segments']:
        name=segment['label']
        if name.startswith('task_pose_')or name=='turn_empty_fixture':
            index=round((segment['start']+segment['end'])/2/segments['duration']*479)
            indices.add(index);events[str(index)]=name
    indices=sorted(indices);folder=HERE/'encoded_inspection'/label
    folder.mkdir(parents=True,exist_ok=True)
    expression='+'.join('eq(n\\,'+str(i)+')'for i in indices)
    subprocess.run(['ffmpeg','-v','error','-i',str(movie),'-vf',
        'select='+expression+',scale=480:270','-fps_mode','vfr',
        str(folder/'frame_%03d.png')],check=True)
    pictures=sorted(folder.glob('frame_*.png'));assert len(pictures)==len(indices)
    board=Image.new('RGB',(1920,292*int(np.ceil(len(indices)/4))),'white')
    ink=ImageDraw.Draw(board)
    for k,(index,pic)in enumerate(zip(indices,pictures)):
        x=(k%4)*480;y=(k//4)*292
        with Image.open(pic)as frame:board.paste(frame,(x,y))
        ink.text((x+6,y+273),f'{index/24:.2f}s  {events.get(str(index),"motion")}',fill='black')
    board.save(folder/'encoded_motion_sheet.png')
    records.append(dict(set=label,video_sha256=hashlib.sha256(movie.read_bytes()).hexdigest(),
        frames=indices,events=events,contact_sheet=str(folder/'encoded_motion_sheet.png'),
        all_task_segments_included=True,all_empty_support_turn_segments_included=True))
(HERE/'encoded_inspection.json').write_text(json.dumps(dict(complete=True,results=records),indent=2)+'\n')
print('ENCODED_FRAME_SHEETS_COMPLETE',len(records),sum(len(r['frames'])for r in records))
