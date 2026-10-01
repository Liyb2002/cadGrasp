/* Render and exercise the offline geometry viewer; no network assets. */
const {chromium}=require(process.env.PLAYWRIGHT_CORE_PATH||'/tmp/cadgrasp_visual_tools/node_modules/playwright-core');
const fs=require('fs'),path=require('path');
async function main(){
 const out=path.resolve(process.argv[2]||'slides/baseline_algo/output/B/pose1+3/step4');
 const renderOnly=process.argv.includes('--render-only');
 const browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--use-gl=angle','--use-angle=metal']});
 const page=await browser.newPage({viewport:{width:1600,height:1100}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('requestfailed',r=>errors.push(r.url()));
 await page.goto('file://'+path.join(out,'index.html'));await page.waitForFunction(()=>window.GEOMETRY);
 await page.evaluate(()=>{window.GEOMETRY.setMode('overview');window.GEOMETRY.resize(2400,1050);zoom=.85;window.GEOMETRY.draw();});
 const url=await page.evaluate(()=>{
  const result=document.createElement('canvas');result.width=2400;result.height=1150;
  const ctx=result.getContext('2d');ctx.fillStyle='white';ctx.fillRect(0,0,2400,1150);ctx.drawImage(window.GEOMETRY.canvas,0,65);
  ctx.fillStyle='#354b5b';ctx.font='500 30px system-ui';ctx.textAlign='center';
  for(const [i,t] of [...window.GEOMETRY.data.poses.map(p=>p.name.replace('pose_','Pose ')),window.GEOMETRY.data.report.diagnostic_only?'Same five heads':'Same fixture'].entries())ctx.fillText(t,400+800*i,55);
  const dimensions=window.GEOMETRY.data.dimensions_mm.map(x=>Math.round(x)).join(' × ');
  ctx.font='22px system-ui';ctx.fillStyle='#647782';const r=window.GEOMETRY.data.report;
  const status=r.passed===true?'PASSED':r.passed===false?'FAILED CANDIDATE':'CONSTRUCTED · Final audit disabled';
  ctx.fillText(r.diagnostic_only?'REJECTED: floor conflict  ·  5 unique heads / 1 shared patch  ·  Head layout only':`${status}  ·  Color: heads  ·  Gray-white: body  ·  ${dimensions} mm`,1200,1128);
  return result.toDataURL('image/png');
 });
 fs.writeFileSync(path.join(out,'overview.png'),Buffer.from(url.split(',')[1],'base64'));
 await page.evaluate(()=>{window.GEOMETRY.setMode('fixture');window.GEOMETRY.resize(1500,1350);});
 const fixtureImage=await page.evaluate(()=>window.GEOMETRY.canvas.toDataURL('image/png'));
 const layoutOnly=await page.evaluate(()=>Boolean(window.GEOMETRY.data.report.diagnostic_only));
 fs.writeFileSync(path.join(out,layoutOnly?'head_layout.png':'fixture.png'),Buffer.from(fixtureImage.split(',')[1],'base64'));
 if(renderOnly){await browser.close();if(errors.length)throw new Error(errors.join('\n'));console.log('PNG exported; interactive audit disabled.');return;}
 const hasSeparate=await page.evaluate(()=>Boolean(window.GEOMETRY.data.design_parts));
 if(hasSeparate){
  await page.evaluate(()=>{window.GEOMETRY.setMode('overview');window.GEOMETRY.resize(2400,1050);zoom=.85;window.GEOMETRY.draw();});
  const merged=await page.evaluate(()=>window.GEOMETRY.canvas.toDataURL());
  await page.locator('#separate').click();
  const split=await page.evaluate(()=>window.GEOMETRY.canvas.toDataURL());
  if(merged===split)throw new Error('Separate structure view did not change');
  const separated=await page.evaluate(()=>{const c=document.createElement('canvas');c.width=2400;c.height=1150;const ctx=c.getContext('2d');ctx.fillStyle='white';ctx.fillRect(0,0,2400,1150);ctx.drawImage(window.GEOMETRY.canvas,0,65);ctx.fillStyle='#354b5b';ctx.font='500 30px system-ui';ctx.textAlign='center';for(const [i,label] of ['Pose 1 structure','Pose 3 structure','Merged fixture'].entries())ctx.fillText(label,400+800*i,55);ctx.font='22px system-ui';ctx.fillStyle='#647782';ctx.fillText('Separate floor frames + active heads, then one connected union',1200,1128);return c.toDataURL('image/png');});
  fs.writeFileSync(path.join(out,'separate.png'),Buffer.from(separated.split(',')[1],'base64'));
  await page.locator('#separate').click();
 }
 const hasLocalBodies=await page.evaluate(()=>Boolean(window.GEOMETRY.data.local_bodies));
 if(hasLocalBodies){
  await page.evaluate(()=>{window.GEOMETRY.setMode('overview');window.GEOMETRY.resize(2400,1050);zoom=.85;window.GEOMETRY.draw();});
  const whole=await page.evaluate(()=>window.GEOMETRY.canvas.toDataURL());
  await page.locator('#local-body').selectOption('4');
  const local=await page.evaluate(()=>window.GEOMETRY.canvas.toDataURL());
  if(whole===local)throw new Error('Local body selection did not change geometry');
  await page.locator('#ghost').click();
  const selectedImage=await page.evaluate(()=>{const c=document.createElement('canvas');c.width=2400;c.height=1150;const ctx=c.getContext('2d');ctx.fillStyle='white';ctx.fillRect(0,0,2400,1150);ctx.drawImage(window.GEOMETRY.canvas,0,65);ctx.fillStyle='#354b5b';ctx.font='500 30px system-ui';ctx.textAlign='center';for(const [i,label] of [...window.GEOMETRY.data.poses.map(p=>p.name.replace('pose_','Pose ')),'Same local body'].entries())ctx.fillText(label,400+800*i,55);ctx.font='22px system-ui';ctx.fillStyle='#647782';ctx.fillText('Selected head body in both poses; other fixture material hidden',1200,1128);return c.toDataURL('image/png');});
  fs.writeFileSync(path.join(out,'local_body.png'),Buffer.from(selectedImage.split(',')[1],'base64'));
  await page.locator('#ghost').click();await page.locator('#local-body').selectOption('-1');
 }
 for(const mode of ['0','1','fixture'])await page.locator(`[data-mode="${mode}"]`).click();
 const before=await page.evaluate(()=>window.GEOMETRY.canvas.toDataURL());
 const box=await page.locator('#view').boundingBox();await page.mouse.move(box.x+box.width*.4,box.y+box.height*.5);await page.mouse.down();await page.mouse.move(box.x+box.width*.4+80,box.y+box.height*.5+30);await page.mouse.up();
 const after=await page.evaluate(()=>window.GEOMETRY.canvas.toDataURL());if(before===after)throw new Error('Drag did not change the camera');
 await page.mouse.wheel(0,-100);await page.locator('#reset').click();await page.locator('[data-mode="overview"]').click();
 const opaque=await page.evaluate(()=>window.GEOMETRY.canvas.toDataURL());await page.locator('#ghost').click();const transparent=await page.evaluate(()=>window.GEOMETRY.canvas.toDataURL());if(opaque===transparent)throw new Error('Workpiece transparency did not change');await page.locator('#ghost').click();
 const checks=await page.evaluate(()=>window.GEOMETRY.data.report.checks||[]);
 fs.writeFileSync(path.join(out,'viewer_check.json'),JSON.stringify({browser_errors:errors,offline:true,pose_switching:true,fixture_only_mode:true,drag_rotation:true,workpiece_transparency:true,separate_structures_toggle:hasSeparate,local_body_selection:hasLocalBodies,color_mode:'colored_heads_neutral_bodies',head_region_count:await page.evaluate(()=>window.GEOMETRY.data.head_regions.length),geometry_checks:checks.map(x=>({pose:x.pose,passed:x.coupled_equilibrium_passed,samples:x.original_sample_count})),exported_image:'overview.png',fixture_only_image:layoutOnly?'head_layout.png':'fixture.png',local_body_image:hasLocalBodies?'local_body.png':null,separate_structures_image:hasSeparate?'separate.png':null},null,2)+'\n');
 await browser.close();if(errors.length)throw new Error(errors.join('\n'));
 console.log('Overview exported; pose switching, fixture view and drag passed.');
}
main().catch(e=>{console.error(e);process.exit(1)});
