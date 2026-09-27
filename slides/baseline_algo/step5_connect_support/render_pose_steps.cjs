/* Six annotated, fixed-Pose1 figures using the verified construction replay. */
const {chromium}=require(process.env.PLAYWRIGHT_CORE_PATH||'/tmp/cadgrasp_visual_tools/node_modules/playwright-core');
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const out=path.resolve(process.argv[2]||'slides/baseline_algo/output/B/pose1+3/step5');
const figures=path.join(out,'pose1_steps');
const hash=file=>crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
async function main(){
 const prepared=JSON.parse(fs.readFileSync(path.join(figures,'pose1_steps_data.json'),'utf8'));
 const design=JSON.parse(fs.readFileSync(path.join(out,'design.json'),'utf8'));
 const browser=await chromium.launch({executablePath:process.env.CHROME_PATH||'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',headless:true,args:['--use-gl=angle','--use-angle=metal']});
 try{
  const page=await browser.newPage({viewport:{width:1600,height:1100}}),errors=[];
  page.on('pageerror',e=>errors.push(e.message));
  await page.goto('file://'+path.join(out,'construction.html'));await page.waitForFunction(()=>window.CONSTRUCTION);
  const images=await page.evaluate(({info,design})=>{
   const W=1400,H=1000,plot={x:20,y:155,w:1000,h:710},s=scenes[0],cam=s.camera;
   const canvas=document.createElement('canvas');canvas.width=W;canvas.height=H;const ink=canvas.getContext('2d');
   const notes=new THREE.Group();s.scene.add(notes);
   const facing=new THREE.Vector3(Math.cos(-2.35)*Math.cos(.5),Math.sin(-2.35)*Math.cos(.5),Math.sin(.5));
   function fit(camera,bounds,width,height,margin=1.13){const focus=bounds.getCenter(new THREE.Vector3());camera.up.set(0,0,1);camera.position.copy(focus).add(facing.clone().multiplyScalar(2));camera.lookAt(focus);camera.updateMatrixWorld(true);
    const projected=new THREE.Box3();for(const x of [bounds.min.x,bounds.max.x])for(const y of [bounds.min.y,bounds.max.y])for(const z of [bounds.min.z,bounds.max.z])projected.expandByPoint(new THREE.Vector3(x,y,z).applyMatrix4(camera.matrixWorldInverse));
    const aspect=width/height,half=Math.max(Math.abs(projected.min.y),Math.abs(projected.max.y),Math.abs(projected.min.x)/aspect,Math.abs(projected.max.x)/aspect)*margin;
    camera.left=-half*aspect;camera.right=half*aspect;camera.top=half;camera.bottom=-half;camera.near=.001;camera.far=10;camera.updateProjectionMatrix();
   }
   fit(cam,s.bounds,plot.w,plot.h);
   function worldScreen(p){const q=new THREE.Vector3(...p).project(cam);return [plot.x+(q.x+1)*plot.w/2,plot.y+(1-q.y)*plot.h/2];}
   function text(value,x,y,size=25,color='#435d69',weight=400){ink.textAlign='left';ink.font=`${weight} ${size}px system-ui`;ink.fillStyle=color;ink.fillText(value,x,y);}
   function wrap(value,width,size){ink.font=`${size}px system-ui`;const lines=[];let line='';for(const c of value){if(ink.measureText(line+c).width>width&&line){lines.push(line);line=c;}else line+=c;}if(line)lines.push(line);return lines;}
   function card(title,lines,x,y,width=320,accent='#42776c',point=null){const body=lines.flatMap(line=>wrap(line,width-32,24)),height=67+body.length*34;
    if(point){const [px,py]=worldScreen(point);ink.strokeStyle=accent;ink.lineWidth=2;ink.beginPath();ink.moveTo(px,py);ink.lineTo(x-20,y+28);ink.lineTo(x,y+28);ink.stroke();ink.fillStyle=accent;ink.beginPath();ink.arc(px,py,5,0,Math.PI*2);ink.fill();}
    ink.fillStyle='#f3f7f7';ink.beginPath();ink.roundRect(x,y,width,height,10);ink.fill();ink.fillStyle=accent;ink.fillRect(x,y+10,4,height-20);
    text(title,x+17,y+34,26,accent,600);body.forEach((line,i)=>text(line,x+17,y+70+i*34,24));
   }
   function polygons(geo){if(!geo)return [];if(geo.type==='Polygon')return [geo.coordinates];if(geo.type==='MultiPolygon')return geo.coordinates;if(geo.type==='GeometryCollection')return geo.geometries.flatMap(polygons);return [];}
   function markFloor(geo){for(const poly of polygons(geo))for(const ring of poly){const points=ring.map(p=>new THREE.Vector3(p[0],p[1],.0003));const g=new THREE.BufferGeometry().setFromPoints(points);const line=new THREE.Line(g,new THREE.LineBasicMaterial({color:0xcf921e,depthTest:false}));line.renderOrder=9;notes.add(line);}}
   function tag(label,p,color='#ad7815',dx=0,dy=0){const [x,y]=worldScreen(p);ink.font='600 25px system-ui';const width=ink.measureText(label).width+18;ink.fillStyle='#fff9e9';ink.strokeStyle=color;ink.lineWidth=1.4;ink.beginPath();ink.roundRect(x-width/2+dx,y-16+dy,width,32,7);ink.fill();ink.stroke();ink.textAlign='center';ink.fillStyle=color;ink.fillText(label,x+dx,y+9+dy);ink.textAlign='left';}
   function mean(points){return points[0].map((_,i)=>points.reduce((sum,p)=>sum+p[i],0)/points.length);}
   function endOf(index){const stage=DATA.stages[index];return stage.start+stage.duration*.99;}
   function footPoint(number){const f=info.pose1_feet.find(f=>f.number===number);return [...mean(f.xy),0];}
   function base(number,title,subtitle){ink.fillStyle='white';ink.fillRect(0,0,W,H);ink.fillStyle='#e2eee9';ink.beginPath();ink.roundRect(36,26,70,60,12);ink.fill();text(String(number).padStart(2,'0'),48,69,35,'#397163',650);
    text(title,125,67,35,'#283f4b',650);text(subtitle,40,116,25,'#70818b');ink.strokeStyle='#e6eded';ink.beginPath();ink.moveTo(40,137);ink.lineTo(W-40,137);ink.stroke();}
   function footer(lines){lines.forEach((line,i)=>text(line,40,914+i*37,25,i===0?'#355849':'#6a7e86',i===0?550:400));}
   function render(index,floorKey=null){updateGeometry(index===0?0:endOf(index));notes.clear();if(floorKey)markFloor(info.floors[floorKey].geometry);renderer.setScissorTest(false);renderer.setSize(plot.w,plot.h,false);renderer.setViewport(0,0,plot.w,plot.h);renderer.render(s.scene,cam);ink.drawImage(webgl,plot.x,plot.y);}
   function insetTeal(x,y,width,height){const scene=new THREE.Scene();scene.add(new THREE.HemisphereLight(0xffffff,0xaababa,2.1));const light=new THREE.DirectionalLight(0xffffff,2.8);light.position.set(-.3,-.4,.7);scene.add(light);
    const body=makeDecorated(decorated(geometry(info.teal_body)));scene.add(body);scene.updateMatrixWorld(true);const bounds=new THREE.Box3().setFromObject(body),focus=bounds.getCenter(new THREE.Vector3());
    const ground=new THREE.Mesh(new THREE.PlaneGeometry(.1,.1),new THREE.MeshBasicMaterial({color:0xf0f4f4,side:THREE.DoubleSide}));ground.position.set(focus.x,focus.y,-.0004);scene.add(ground);
    const camera=new THREE.OrthographicCamera();fit(camera,bounds,width,height,1.24);renderer.setSize(width,height,false);renderer.setViewport(0,0,width,height);renderer.render(scene,camera);ink.drawImage(webgl,x,y);ink.strokeStyle='#d9e5e1';ink.strokeRect(x,y,width,height);
   }
   const result=[];
   base(1,'固定 Pose1：先看这些头','物体、支架摆放与相机保持不动；先不添加身体。');render(0);
   card('当前负责接触物体',['橙、紫、蓝三个头','接触面保持原样'],1040,215,320,'#6a668d',info.head_centers_m[1]);
   card('另一姿态使用的头',['青、绿、另一橙头','它们的身体可以做脚'],1040,470,320,'#427f74',info.head_centers_m[4]);
   footer(['下面追踪青色头：它在 Pose1 不负责接触物体，但它的身体可以落地。','彩色始终标记原头；灰白色表示后来增加的身体。']);result.push(canvas.toDataURL());

   base(2,'先长出初始身体，形成小块脚面','关注青色这块：它向当前 Pose1 的地面展开。');render(6,'6');
   card('这块仍在半空',['紫色身体为 Pose3 预备','不要求它在这里落地'],1040,210,320,'#7d7191',mean(DATA.stages[2].target_polygon));
   card('这块已经贴地',['青色身体的下端在 z=0','金色轮廓标出真实接地面'],1040,490,320,'#427f74',info.teal_initial_floor_point_m);
   footer(['同一批身体里，有的给 Pose1 落地，有的给 Pose3 落地。','本图始终是 Pose1；没有翻动物体，也没有生成第二份支架。']);result.push(canvas.toDataURL());

   base(3,'俯视同一个地面：哪里还缺支撑范围？','这里仅把相机移到正上方，支架仍然处于 Pose1。');
   const feet=info.pose1_feet,cloud=info.floor_demands_xy_m,initial=info.floors['6'];
   const all=cloud.concat(feet.flatMap(f=>f.xy),initial.support_hull_xy_m),xs=all.map(p=>p[0]),ys=all.map(p=>p[1]);
   const min=[Math.min(...xs),Math.min(...ys)],max=[Math.max(...xs),Math.max(...ys)],scale=Math.min(820/(max[0]-min[0]),640/(max[1]-min[1]));
   const center=[(min[0]+max[0])/2,(min[1]+max[1])/2],xy=p=>[505+(p[0]-center[0])*scale,510-(p[1]-center[1])*scale];
   function path2(points,close=true){ink.beginPath();points.forEach((p,i)=>{const [x,y]=xy(p);i?ink.lineTo(x,y):ink.moveTo(x,y);});if(close)ink.closePath();}
   ink.save();ink.beginPath();ink.rect(70,155,930,720);ink.clip();
   const object=DATA.poses[0].object;ink.fillStyle='#f0f3f4';for(let i=0;i<object.faces.length;i+=3){path2(object.faces.slice(i,i+3).map(j=>object.vertices.slice(j*3,j*3+2)));ink.fill();}
   ink.fillStyle='rgba(104,123,133,.18)';for(const p of cloud){const [x,y]=xy(p);ink.fillRect(x,y,1.7,1.7);}
   for(const rings of polygons(initial.geometry)){ink.beginPath();for(const ring of rings){ring.forEach((p,i)=>{const [x,y]=xy(p);i?ink.lineTo(x,y):ink.moveTo(x,y);});ink.closePath();}ink.fillStyle='#75a89e';ink.fill('evenodd');}
   ink.strokeStyle='#46778f';ink.lineWidth=3;ink.setLineDash([10,7]);path2(initial.support_hull_xy_m);ink.stroke();ink.setLineDash([]);
   for(const f of feet){path2(f.xy);ink.fillStyle='#fff1ce';ink.fill();ink.strokeStyle='#b77f18';ink.lineWidth=2;ink.stroke();const [x,y]=xy(mean(f.xy));text(['①','②','③','④'][f.number-1],x-14,y-19,29,'#a9720c',650);}
   const pivot=xy(info.original_object_floor_point_m);ink.strokeStyle='#354955';ink.lineWidth=3;ink.beginPath();ink.moveTo(pivot[0]-7,pivot[1]-7);ink.lineTo(pivot[0]+7,pivot[1]+7);ink.moveTo(pivot[0]-7,pivot[1]+7);ink.lineTo(pivot[0]+7,pivot[1]-7);ink.stroke();ink.restore();
   card('绿块：已有脚面',['灰点：承载需求撒点','×：物体原有地面支点'],1040,180,320,'#467f72');
   card('蓝虚线：支撑范围',[`${initial.outside_original_sample_count.toLocaleString()} 个原样本点在外面`,'需要向外补足范围'],1040,385,320,'#46778f');
   card('①—④：补充脚端',['选几块分散的小区域','不把虚线做成实体圈'],1040,590,320,'#a87919');
   footer(['支撑范围的边界是虚拟的；真正需要材料的是小块脚面及其身体。','这一步用撒点判断范围，最终还要做完整受力验算。']);result.push(canvas.toDataURL());

   base(4,'局部插值：把青色身体展开到脚端②','头部接触面不动，改变它后面的身体形状。');render(8);tag('②',footPoint(2),undefined,0,12);
   card('从已有身体向外展开',['斜着连接到局部脚面②','不用连接整个包围圈'],1040,195,320,'#427f74',info.head_centers_m[4]);
   text('单独看这块青色身体',1040,500,25,'#427f74',600);insetTeal(1040,520,320,260);
   footer(['右下角单独显示同一块真实身体：一端保留头部，另一端展开成脚面。','主图保留其它部分作为参照；这一步仅突出②的构造。']);result.push(canvas.toDataURL());

   base(5,'四块脚端都接到各自的身体','每个脚端只与分配给它的局部身体连接。');render(10,'10');
   for(const f of feet)tag(['①','②','③','④'][f.number-1],footPoint(f.number),undefined,f.number===2?-10:0,12);
   card('蓝色头也可以长脚',['蓝头正在服务 Pose1','其身体同时接到①和③'],1040,220,320,'#587b9f',info.head_centers_m[2]);
   card('另外两个脚端',['②接青色身体','④接另一橙色身体'],1040,490,320,'#a87919',footPoint(4));
   footer(['“头在当前 pose 使用”与“身体能否落地”可以同时成立。','脚面保持分散；这些编号之间没有沿地面连接成圈。']);result.push(canvas.toDataURL());

   base(6,'补齐另一姿态需要的部分，再连成一件','最后仍从 Pose1 看同一件完整支架。');render(19,'18');
   card('高处的展开留给 Pose3',['在当前画面可以悬空','换到 Pose3 才作为脚端'],1040,205,320,'#7d7191',mean(DATA.stages[11].target_polygon));
   card('补上四处短连接',['连接在地面上方','让所有身体连成一件'],1040,505,320,'#657c72',mean(design.connections[0].path_m));
   footer(['本图是之前认可的最终实体，完整几何和两姿态原载荷验收均已通过。','图1、2、4、5、6使用完全相同的相机；图3只改成地面俯视。']);result.push(canvas.toDataURL());
   return result;
  },{info:prepared,design});
  if(errors.length)throw new Error(errors.join('\n'));
  const files=images.map((url,i)=>{const name=`pose1_steps_${String(i+1).padStart(2,'0')}.png`;fs.writeFileSync(path.join(figures,name),Buffer.from(url.split(',')[1],'base64'));return name;});
  const sheet=await page.evaluate(async images=>{const c=document.createElement('canvas');c.width=2800;c.height=3160;const ctx=c.getContext('2d');ctx.fillStyle='#f1f5f5';ctx.fillRect(0,0,c.width,c.height);ctx.fillStyle='#29434e';ctx.font='600 48px system-ui';ctx.fillText('Pose1：从头到脚，逐步看同一件支架如何形成',48,66);ctx.font='28px system-ui';ctx.fillStyle='#6e808a';ctx.fillText('按编号从左到右、从上到下看。灰白是身体；脚面分散；虚线只表示支撑范围。',48,115);for(let i=0;i<images.length;i++){const img=new Image();img.src=images[i];await img.decode();ctx.drawImage(img,(i%2)*1400,160+Math.floor(i/2)*1000);}return c.toDataURL();},images);
  fs.writeFileSync(path.join(figures,'pose1_steps.png'),Buffer.from(sheet.split(',')[1],'base64'));
  const captions=['原始头：区分当前和另一姿态的用途','初始身体：看清哪里真的落地','地面俯视：脚面、撒点和虚拟支撑范围','局部插值：青色身体连接脚端②','四处脚端：正在使用的蓝头身体也能当脚','完整实体：保留另一姿态用途，再补短连接'];
  fs.writeFileSync(path.join(figures,'index.html'),`<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pose1 · 分步图解</title><style>body{margin:0;background:#f1f5f5;color:#29434e;font:16px system-ui}main{max-width:1450px;margin:auto;padding:24px}h1{font-size:27px}p{line-height:1.6;color:#617681}nav{display:flex;gap:20px;margin:18px 0}a{color:#326d63}figure{margin:0 0 24px}img{display:block;width:100%;background:white;border-radius:9px}figcaption{padding:10px 5px}section{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px}@media(max-width:950px){section{grid-template-columns:1fr}}</style><main><h1>Pose1 · 从头到脚的六步图解</h1><p>始终保持同一个 Pose1 摆放。第3张俯视地面，其余使用同一个相机。点击图片可放大查看。</p><nav><a href="pose1_steps.png">整组大图</a><a href="../construction.html">构造动画</a><a href="../index.html">完整实体</a></nav><section>${files.map((file,i)=>`<figure><a href="${file}"><img src="${file}" alt="${captions[i]}"></a><figcaption>${i+1}. ${captions[i]}</figcaption></figure>`).join('')}</section></main></html>`);
  for(const target of ['index.html','construction.html']){const file=path.join(out,target);let html=fs.readFileSync(file,'utf8').replaceAll('href="pose1_steps.html"','href="pose1_steps/index.html"');if(!html.includes('href="pose1_steps/index.html"'))html=html.replace('</nav>','<a href="pose1_steps/index.html">Pose1 · 分步图解</a></nav>');fs.writeFileSync(file,html);}
  fs.writeFileSync(path.join(figures,'pose1_steps_check.json'),JSON.stringify({complete:true,browser_errors:errors,pose:'pose_1',fixed_camera_panels:[1,2,4,5,6],top_down_panel:3,
   actual_construction_geometry:true,original_samples_only:true,physical_geometry_changed:false,
   final_geometry_vertices_and_faces_exactly_match:prepared.final_geometry_vertices_and_faces_exactly_match,
   source_replay_symmetric_difference_m3:prepared.source_replay_symmetric_difference_m3,
   final_floor_max_vertex_to_contact_distance_m:prepared.final_floor_max_vertex_to_contact_distance_m,
   code:{[__filename]:hash(__filename)},artifacts:Object.fromEntries([...files,'pose1_steps.png','index.html'].map(name=>[name,hash(path.join(figures,name))]))},null,2)+'\n');
  console.log('Six annotated Pose1 figures, overview sheet, and reading page exported.');
 }finally{await browser.close();}
}
main().catch(e=>{console.error(e);process.exit(1);});
