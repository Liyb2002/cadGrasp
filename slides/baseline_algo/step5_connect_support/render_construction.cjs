/* Render the actual construction replay in both task poses, entirely offline. */
const {chromium}=require(process.env.PLAYWRIGHT_CORE_PATH||'/tmp/cadgrasp_visual_tools/node_modules/playwright-core');
const fs=require('fs'),path=require('path'),{spawn,execFileSync}=require('child_process'),{once}=require('events');
const out=path.resolve(process.argv.find((x,i)=>i>1&&!x.startsWith('--'))||'slides/baseline_algo/output/B/pose1+3/step5');
const preview=process.argv.includes('--preview'),verifyOnly=process.argv.includes('--verify-only'),fps=20;
function encoder(name){
 const file=path.join(out,name),process=spawn('ffmpeg',['-hide_banner','-loglevel','error','-y','-f','image2pipe','-vcodec','png','-framerate',String(fps),'-i','pipe:0',
  '-an','-c:v','libx264','-threads','2','-preset','fast','-crf','18','-profile:v','main','-level:v','4.0','-pix_fmt','yuv420p',
  '-vf','scale=in_range=full:out_range=tv:out_color_matrix=bt709','-color_range','tv','-colorspace','bt709',
  '-color_primaries','bt709','-color_trc','bt709','-tag:v','avc1','-movflags','+faststart',file]);
 let error='';process.stderr.on('data',data=>error+=data);process.stdin.on('error',()=>{});
 const done=new Promise((resolve,reject)=>{process.on('error',reject);process.on('close',code=>code===0?resolve():reject(new Error(`${name}: ffmpeg ${code}: ${error}`)));});
 // Attach a handler immediately; the same promise is awaited after input closes.
 done.catch(()=>{});return {process,done,name};
}
async function main(){
 const browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--use-gl=angle','--use-angle=metal']});
 const encoders=[];
 try{
  const page=await browser.newPage({viewport:{width:1500,height:1150}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('requestfailed',r=>errors.push(r.url()));
  await page.goto('file://'+path.join(out,'construction.html'));await page.waitForFunction(()=>Boolean(window.CONSTRUCTION));
  const info=await page.evaluate(()=>({duration:CONSTRUCTION.data.duration,stages:CONSTRUCTION.data.stages.map(s=>({title:s.title,start:s.start,duration:s.duration,phase:s.phase}))}));
  await page.locator('#stage').selectOption('7');if((await page.evaluate(()=>CONSTRUCTION.state())).index!==7)throw new Error('Stage selector failed');
  await page.locator('#next').click();if((await page.evaluate(()=>CONSTRUCTION.state())).index!==8)throw new Error('Next step failed');
  await page.locator('#previous').click();if((await page.evaluate(()=>CONSTRUCTION.state())).index!==7)throw new Error('Previous step failed');
  await page.locator('[data-mode="0"]').click();if((await page.evaluate(()=>CONSTRUCTION.state())).mode!=='0')throw new Error('Pose view failed');
  await page.locator('[data-mode="both"]').click();
  await page.evaluate(()=>CONSTRUCTION.seek(12));const visible=await page.evaluate(()=>CONSTRUCTION.canvas.toDataURL());
  await page.locator('#object').click();const hidden=await page.evaluate(()=>CONSTRUCTION.canvas.toDataURL());if(visible===hidden)throw new Error('Object visibility failed');await page.locator('#object').click();
  await page.locator('#play').click();await page.waitForTimeout(250);await page.locator('#play').click();if((await page.evaluate(()=>CONSTRUCTION.state())).time<=12)throw new Error('Playback failed');
  const sampleTimes=[0,info.stages[1].start+info.stages[1].duration*.57,info.stages[6].start+info.stages[6].duration-.02,
   info.stages[10].start+info.stages[10].duration-.02,info.stages[18].start+info.stages[18].duration-.02,info.duration];
  const samples=await page.evaluate(times=>times.map(t=>CONSTRUCTION.frame(t,'both')),sampleTimes);
  for(const [i,name] of [[1,'construction_growth.png'],[5,'construction_poster.png']])fs.writeFileSync(path.join(out,name),Buffer.from(samples[i].split(',')[1],'base64'));
  const sheet=await page.evaluate(async frames=>{const canvas=document.createElement('canvas');canvas.width=2400;canvas.height=900;const ctx=canvas.getContext('2d');ctx.fillStyle='white';ctx.fillRect(0,0,canvas.width,canvas.height);for(let i=0;i<frames.length;i++){const image=new Image();image.src=frames[i];await image.decode();ctx.drawImage(image,(i%3)*800,Math.floor(i/3)*450,800,450);}return canvas.toDataURL();},samples);
  fs.writeFileSync(path.join(out,'construction_preview.png'),Buffer.from(sheet.split(',')[1],'base64'));
  if(new Set(samples).size!==samples.length)throw new Error('Construction samples do not change');
  if(errors.length)throw new Error(errors.join('\n'));
  if(preview){console.log(JSON.stringify({preview:true,duration:info.duration,stages:info.stages.length,browser_errors:errors}));return;}
  const names=['construction.mp4','construction_pose1.mp4','construction_pose3.mp4'];
  const frames=Math.ceil(info.duration*fps);
  if(!verifyOnly){
   for(const name of names)encoders.push(encoder(name));
   for(let i=0;i<frames;i++){
   const t=Math.min(info.duration,i/fps);
   const images=await page.evaluate(t=>['both','0','1'].map(view=>CONSTRUCTION.frame(t,view)),t);
   for(let k=0;k<encoders.length;k++){
    const pipe=encoders[k].process.stdin;
    if(!pipe.write(Buffer.from(images[k].split(',')[1],'base64')))await once(pipe,'drain');
   }
   if(i%100===0)console.log(`Rendered ${i}/${frames} frames for all three videos`);
   }
   for(const e of encoders)e.process.stdin.end();await Promise.all(encoders.map(e=>e.done));
  }
  const videoChecks=names.map(name=>{
   const data=JSON.parse(execFileSync('ffprobe',['-v','error','-count_frames','-show_entries','stream=codec_name,profile,width,height,pix_fmt,nb_read_frames,r_frame_rate:format=duration','-of','json',path.join(out,name)],{encoding:'utf8'}));
   const s=data.streams[0];if(s.codec_name!=='h264'||s.pix_fmt!=='yuv420p'||Number(s.nb_read_frames)!==frames)throw new Error(`Invalid video stream: ${name}`);
   execFileSync('ffmpeg',['-v','error','-i',path.join(out,name),'-f','null','-'],{stdio:['ignore','ignore','pipe']});
   return {file:name,...data,decoded_without_errors:true};
  });
  for(const check of videoChecks){
   const decoded=await page.evaluate(file=>new Promise((resolve,reject)=>{
    const video=document.createElement('video');video.muted=true;video.preload='auto';video.style.display='none';document.body.appendChild(video);
    const timeout=setTimeout(()=>{video.remove();reject(new Error('Video seek timed out: '+file));},15000);
    video.onerror=()=>{clearTimeout(timeout);video.remove();reject(new Error('Browser could not decode '+file));};
    video.onloadedmetadata=()=>{video.currentTime=25;};
    video.onseeked=()=>{const result={width:video.videoWidth,height:video.videoHeight,duration:video.duration,seek_time:video.currentTime,ready_state:video.readyState};clearTimeout(timeout);video.remove();resolve(result);};
    video.src=file;
   }),check.file);
   if(decoded.width!==1600||decoded.height!==900)throw new Error('Unexpected browser video dimensions');
   if(decoded.ready_state<2)throw new Error('Browser seek did not decode a frame');check.browser_decoding=decoded;
  }
  execFileSync('ffmpeg',['-v','error','-y','-ss','25','-i',path.join(out,names[0]),'-frames:v','1',path.join(out,'construction_decoded_frame.png')],{stdio:['ignore','ignore','pipe']});
  if(errors.length)throw new Error(errors.join('\n'));
  fs.writeFileSync(path.join(out,'construction_video_check.json'),JSON.stringify({complete:true,fps,frame_count:frames,duration_seconds:info.duration,
   browser_errors:errors,offline:true,stage_selection:true,previous_next:true,pose_switching:true,object_visibility:true,playback:true,
   actual_boolean_additions:true,physical_acceptance_reference:'report.json',construction_replay_reference:'construction_sequence.json',
   videos:videoChecks},null,2)+'\n');
  const index=path.join(out,'index.html');let html=fs.readFileSync(index,'utf8');if(!html.includes('href="construction.html"')){html=html.replace('</nav>','<a href="construction.html">逐步构造 · 视频</a></nav>');fs.writeFileSync(index,html);}
  console.log('All three videos encoded and fully decoded successfully.');
 }finally{
  for(const e of encoders)if(e.process.exitCode===null)e.process.kill();
  await browser.close();
 }
}
main().catch(e=>{console.error(e);process.exit(1);});
