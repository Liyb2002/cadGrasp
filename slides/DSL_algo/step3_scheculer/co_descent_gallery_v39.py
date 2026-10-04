"""Contact sheets and provenance for nine accepted shared fixtures; no geometry replay."""
from pathlib import Path
import json,time
from PIL import Image,ImageDraw,ImageFont
from step3_scheculer import contacts as I
from step3_scheculer.compare_shared_volume_v33 import GROUPS
from step3_scheculer import co_descent_v39 as NEW

FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

def main():
    last=I.OUTPUTS/'B'/GROUPS[-1]/'step5_evaluate'/NEW.STAGE;last.mkdir(parents=True,exist_ok=True)
    rows=[];inputs=[]
    for group in GROUPS:
        stage=NEW.STAGE
        base=I.OUTPUTS/'B'/group;s4=base/'step4'/stage;s5=base/'step5_evaluate'/stage
        r=I.check_report(s4/'report.json');metric=I.check_report(s5/'report.json');assert r['passed'] and r['constructed']
        inputs += [s4/'report.json',s5/'report.json',s4/'overview.png',s5/'overview.png']
        rows.append(dict(group=group,stage=stage,step4=str(s4.relative_to(I.ROOT)),step5=str(s5.relative_to(I.ROOT)),direction=r['common_object_exit_world'],heads=[len(x) for x in json.loads((s4/'state.json').read_text())['active_head_ids']],material_cm3=r['volume_cm3'],box_cm3=metric['aggregate']['object_and_support_poses']['box_volume_cm3'],seconds=r['seconds'],passed=True))
    width=1800;tile_w=600;tile_h=460
    for step in (4,5):
        sheet=Image.new('RGB',(width,3*tile_h),'#ffffff');draw=ImageDraw.Draw(sheet);font=ImageFont.truetype(FONT,23)
        for i,row in enumerate(rows):
            image=Image.open(I.ROOT/row[f'step{step}']/'overview.png').convert('RGB')
            if step==4:
                # One object pose and the complete fixture alone, from actual overview.
                count=len(row['heads'])+1;columns=min(3,count);panel=image.width//columns
                first=image.crop((0,0,panel,panel));last_idx=count-1
                alone=image.crop(((last_idx%columns)*panel,(last_idx//columns)*panel,(last_idx%columns+1)*panel,(last_idx//columns+1)*panel))
                image=Image.new('RGB',(2*panel,panel),'white');image.paste(first,(0,0));image.paste(alone,(panel,0))
            image.thumbnail((tile_w-12,tile_h-80),Image.Resampling.LANCZOS)
            x=(i%3)*tile_w;y=(i//3)*tile_h
            draw.text((x+12,y+9),f"B / {row['group']}",font=font,fill='#253d4b')
            sheet.paste(image,(x+(tile_w-image.width)//2,y+45+(tile_h-85-image.height)//2))
            d='('+', '.join(f'{v:.3f}' for v in row['direction'])+')'
            text=f"PASS | exit {d} | heads {sum(row['heads'])}" if step==4 else f"XYZ box {row['box_cm3']:.1f} cm3"
            draw.text((x+12,y+tile_h-30),text,font=ImageFont.truetype(FONT,20),fill='#317358')
        sheet.save(last/f'step{step}_all.png')
    lines=['# B: nine complete shared supports','', 'All nine use current immutable pose/head/load data, unchanged native object poses, and freely repaired fixture seating. Every fixture is one connected solid and passes full continuous object exit, all installed heads, working surfaces, floor coverage, contact preservation and complete 5 mm cores. All 32768 original loads per pose and the shared no-uplift equation retain their Step3 proof. Single in-memory construction acceptance; no exported-model or independent replay.','', 'All results use continuous common world-exit/head co-descent. Historical accepted fixtures remain preserved; originals are fallback incumbents.','', '| Set | Common world exit | Heads per pose | Material cm³ | Step5 XYZ box cm³ | Seconds incl. images |','|---|---|---|---:|---:|---:|']
    for r in rows:
        direction='('+', '.join(f'{v:.6f}' for v in r['direction'])+')'
        lines += [f"| {r['group']} | {direction} | {r['heads']} | {r['material_cm3']:.3f} | {r['box_cm3']:.3f} | {r['seconds']:.3f} |"]
    lines+=['','## Full-resolution images','']
    for r in rows:
        p4=I.ROOT/r['step4'];p5=I.ROOT/r['step5']
        lines += [f"- {r['group']}: [Step4 overview]({p4}/overview.png), [Step4 construction]({p4}/construction_steps.png), [Step5 boxes]({p5}/overview.png), [complete OBJ]({p4}/shape.obj)."]
    (last/'gallery.md').write_text('\n'.join(lines)+'\n')
    report=dict(complete=True,passed=True,groups=rows,full_fixtures_passed=9,load_count_per_pose=32768,provenance=dict(inputs=I.hashes(inputs),code=I.hashes([Path(__file__)])),artifacts={n:I.sha256(last/n) for n in ['step4_all.png','step5_all.png','gallery.md']})
    I.save(last/'gallery.json',report);I.check_report(last/'gallery.json');print(json.dumps(rows,indent=2));print('OUTPUT',last)

if __name__=='__main__':main()
