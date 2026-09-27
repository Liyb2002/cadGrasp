/* Export the exact same offline viewer to PNG and a deterministic 24 fps MP4.
   npm install playwright-core, or set PLAYWRIGHT_CORE_PATH to its module path. */
const {chromium}=require(process.env.PLAYWRIGHT_CORE_PATH||'playwright-core');
const fs=require('fs'),path=require('path'),{spawn}=require('child_process'),{once}=require('events');
const out=path.resolve(__dirname,'..');
async function main(){
 const gpu=process.platform==='darwin'&&process.env.REUSE_SOFTWARE_RENDERING!=='1';
 const graphics=gpu?['--use-gl=angle','--use-angle=metal']:['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'];
 const browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:graphics});
 const page=await browser.newPage({viewport:{width:1650,height:1200}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));page.on('requestfailed',r=>errors.push(`${r.url()}: ${r.failure().errorText}`));
 await page.goto('file://'+path.join(out,'index.html'));
 async function save(file){const url=await page.evaluate(()=>window.REUSE.canvas.toDataURL('image/png'));fs.writeFileSync(path.join(out,file),Buffer.from(url.split(',')[1],'base64'));}
 await page.evaluate(()=>{window.REUSE.resize(3000,1100);window.REUSE.setMode('overview');});
 await save('reuse_overview.png');
 const stamps=await page.evaluate(()=>window.REUSE_DATA.storyTimes),duration=await page.evaluate(()=>window.REUSE_DATA.duration),playbackSpeed=await page.evaluate(()=>window.REUSE_DATA.playbackSpeed);
 await page.evaluate(()=>window.REUSE.resize(1600,1000));
 const cameraCheck={checked_frames:0,minimum_opening_facing_cosine:1,maximum_frame_extent:0,maximum_opening_direction_error:0,camera_and_table_stationary:true};
 let stationaryState=null;
 function checkFrame(state){
  if(state.openingFacingCosine<=0)throw new Error(`Fixture opening faces away at ${state.time}s`);
  if(state.framingMaxAbs>1)throw new Error(`Scene is cropped at ${state.time}s`);
  const fixed=[...state.camera.position,...state.camera.quaternion,...state.camera.frustum,...state.floor.position,...state.floor.quaternion,...state.floor.scale];
  if(stationaryState&&fixed.some((v,i)=>Math.abs(v-stationaryState[i])>1e-12))throw new Error(`Camera or table moves at ${state.time}s`);
  stationaryState=fixed;
  const openingError=Math.hypot(...state.openingDirection.map((v,i)=>v-[0,1,0][i]));
  if(openingError>1e-6)throw new Error(`Fixture opening changes world direction at ${state.time}s`);
  cameraCheck.maximum_opening_direction_error=Math.max(cameraCheck.maximum_opening_direction_error,openingError);
  cameraCheck.checked_frames++;
  cameraCheck.minimum_opening_facing_cosine=Math.min(cameraCheck.minimum_opening_facing_cosine,state.openingFacingCosine);
  cameraCheck.maximum_frame_extent=Math.max(cameraCheck.maximum_frame_extent,state.framingMaxAbs);
 }
 for(const t of stamps)checkFrame(await page.evaluate(t=>{window.REUSE.setTime(t);return window.REUSE.getState();},t));
 console.log('Overview exported; camera and table stay fixed, fixture opening always points along +Y.');
 if(!process.argv.includes('--images-only')){
  await page.evaluate(()=>window.REUSE.resize(1440,900));
  const temporary=path.join(out,'reuse_workflow.rendering.mp4');
  const ff=spawn('ffmpeg',['-hide_banner','-loglevel','error','-y','-f','image2pipe','-vcodec','mjpeg','-framerate','24','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',temporary],{stdio:['pipe','inherit','inherit']});
  const done=once(ff,'close');
  for(let i=0;i<Math.round(duration*24);i++){
   const {url,state}=await page.evaluate(t=>{window.REUSE.setTime(t);return {url:window.REUSE.canvas.toDataURL('image/jpeg',.94),state:window.REUSE.getState()};},i/24);
   checkFrame(state);
   if(!ff.stdin.write(Buffer.from(url.split(',')[1],'base64')))await once(ff.stdin,'drain');
   if(i%96===0)console.log(`video ${i}/${Math.round(duration*24)} frames`);
  }
  ff.stdin.end();const [code]=await done;if(code!==0)throw new Error(`ffmpeg exited ${code}`);
  fs.renameSync(temporary,path.join(out,'reuse_workflow.mp4'));
 }
 // Exercise the user controls and inspect local asset loading.
 await page.locator('[data-mode="pose1"]').click();
 await page.locator('#reset').click();
 const b=await page.locator('#screen').boundingBox();await page.mouse.move(b.x+b.width*.45,b.y+b.height*.5);await page.mouse.down();await page.mouse.move(b.x+b.width*.45+35,b.y+b.height*.5+15);await page.mouse.up();await page.mouse.wheel(0,-70);
 await page.locator('#reset').click();
 const taskSources=await page.evaluate(()=>window.REUSE_DATA.poses.map(p=>p.source)),magnification=await page.evaluate(()=>window.REUSE.getState().videoMagnification);
 const report={browser_errors:errors,exported_images:['reuse_overview.png'],working_modes:['overview','pose0','pose1','pose2','fixture','video'],task_pose_sources:taskSources,keyframe_times_seconds:stamps,video:{encoded:!process.argv.includes('--images-only'),duration_seconds:duration,playback_speed:playbackSpeed,fps:24,width:1440,height:900,composition:'single_full_scene',camera_mode:'fixed',magnification,fixed_camera:true,fixed_table:true,fixed_scale:true,level_horizon:true,cast_shadows:false,camera_check:cameraCheck},scope:'Rendering and UI checks only; no mechanics or collision certification.'};
 fs.writeFileSync(path.join(out,'render_check.json'),JSON.stringify(report,null,2)+'\n');
 await browser.close();if(errors.length)throw new Error(errors.join('\n'));console.log('Export complete; browser reported no errors.');
}
main().catch(e=>{console.error(e);process.exit(1)});
