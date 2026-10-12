'use strict';
// Deliberately text-free imagery, including all PNG and MP4 exports.
const D=window.REUSE_DATA,screen=document.getElementById('screen'),ctx=screen.getContext('2d');
const renderer=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});
renderer.setClearColor(0xffffff);renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.shadowMap.enabled=false;
const scene=new THREE.Scene();scene.add(new THREE.HemisphereLight(0xffffff,0xc5cdd4,1.7));
for(const [p,k] of [[[.7,-.8,1],1.8],[[-.6,.2,.6],.7]]){const l=new THREE.DirectionalLight(0xffffff,k);l.position.set(...p);if(k>1){l.castShadow=true;l.shadow.mapSize.set(2048,2048);Object.assign(l.shadow.camera,{left:-1,right:1,top:1,bottom:-1,near:.01,far:4});l.shadow.bias=-.0001;l.shadow.normalBias=.0005;}scene.add(l);}
const camera=new THREE.OrthographicCamera(-1,1,1,-1,.001,10);camera.up.set(0,0,1);
const material=c=>new THREE.MeshStandardMaterial({color:c,roughness:.86,metalness:0,side:THREE.DoubleSide,flatShading:true});
function geom(m){const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(m.v,3));g.setIndex(m.f);if(m.n)g.setAttribute('normal',new THREE.Float32BufferAttribute(m.n,3));else g.computeVertexNormals();return g;}
function quat(r){return new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().set(...r[0],0,...r[1],0,...r[2],0,0,0,0,1));}
const fq=D.poses.map(p=>quat(p.fixtureR)),oq=D.poses.map(p=>quat(p.objectR));
const centers=D.poses.map(p=>{const c=new THREE.Vector3();for(let i=0;i<p.object.v.length;i+=3)c.add(new THREE.Vector3(...p.object.v.slice(i,i+3)));return c.multiplyScalar(3/p.object.v.length);});
const fixture=new THREE.Group();scene.add(fixture);
fixture.add(new THREE.Mesh(geom(D.fixture),material(D.fixtureColor)));
for(const part of D.headOverlays){const mat=material(part.color);mat.polygonOffset=true;mat.polygonOffsetFactor=-1;mat.polygonOffsetUnits=-1;fixture.add(new THREE.Mesh(geom(part),mat));}
const raw=D.poses[0].object,local=[],inv=oq[0].clone().invert();
for(let i=0;i<raw.v.length;i+=3)local.push(...new THREE.Vector3(...raw.v.slice(i,i+3)).sub(centers[0]).applyQuaternion(inv).toArray());
const og=geom({v:local,f:raw.f}).toNonIndexed();og.computeVertexNormals();
const om=material(0xffffff);om.vertexColors=true;om.transparent=false;om.opacity=1;om.depthWrite=true;om.side=THREE.FrontSide;
const object=new THREE.Mesh(og,om);scene.add(object);let workIndex=-1;
function work(k){if(k===workIndex)return;workIndex=k;const ids=new Set(D.poses[k].work),colors=[];for(let f=0;f<raw.f.length/3;f++){const c=new THREE.Color(ids.has(f)?'#b4c7b2':'#bac0c5');for(let j=0;j<3;j++)colors.push(c.r,c.g,c.b);}og.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));}
const floor=new THREE.Mesh(new THREE.PlaneGeometry(1,1),material('#f0f2f4'));floor.position.z=-.002;floor.receiveShadow=true;scene.add(floor);
let mode='overview',time=0,playing=false,last=0,az=-Math.PI*.75,el=.526,zoom=1,videoFrame=null;
const VIDEO_MAGNIFICATION=1;
const V=a=>new THREE.Vector3(...a),mix=(a,b,t)=>a.clone().lerp(b,t),ease=t=>t*t*(3-2*t);
function state(k){return {k,op:centers[k].clone(),oq:oq[k].clone(),fp:V(D.poses[k].fixtureT),fq:fq[k].clone(),phase:0};}
const robot=new THREE.Group();scene.add(robot);
const robotLinks=D.robot.map(parts=>{const g=new THREE.Group();for(const p of parts){const mat=material(new THREE.Color(...p.color));mat.flatShading=false;g.add(new THREE.Mesh(geom(p),mat));}robot.add(g);return g;});
for(const root of [fixture,object,robot])root.traverse(m=>{if(m.isMesh){m.castShadow=true;m.receiveShadow=false;}});
const wrist=robotLinks[7],gm=material('#414c54');
function box(size,pos,parent=wrist){const m=new THREE.Mesh(new THREE.BoxGeometry(...size),gm);m.position.set(...pos);m.castShadow=true;m.receiveShadow=false;parent.add(m);return m;}
box([.065,.046,.025],[0,0,.05]);box([.158,.04,.025],[0,0,.073]);
const fingers=[-1,1].map(sign=>{const g=new THREE.Group();wrist.add(g);box([.014,.035,.018],[0,0,.086],g);box([.008,.020,.052],[0,0,.113],g);return {sign,g};});
function sample(t){let low=0,high=D.motion.length-1;while(low+1<high){const mid=(low+high)>>1;if(D.motion[mid].t<=t)low=mid;else high=mid;}const a=D.motion[low],b=D.motion[high],u=Math.max(0,Math.min(1,(t-a.t)/(b.t-a.t)));return {a,b,u};}
const pq=(a,b,u)=>({p:V(a.p).lerp(V(b.p),u),q:new THREE.Quaternion(...a.q).slerp(new THREE.Quaternion(...b.q),u)});
function animated(t){const {a,b,u}=sample(t),o=pq(a.o,b.o,u),f=pq(a.f,b.f,u);return {k:a.k,op:o.p,oq:o.q,fp:f.p,fq:f.q,robot:{a,b,u}};}
function updateRobot(s){const {a,b,u}=s.robot||sample(0),K=D.kinematics;const pos=V(K.base),rot=new THREE.Quaternion(...K.rotation);for(let i=0;i<8;i++){if(i){pos.add(V(K.offsets[i-1]).applyQuaternion(rot));const q=a.joints[i-1]+(b.joints[i-1]-a.joints[i-1])*u;rot.multiply(new THREE.Quaternion().setFromAxisAngle(V(K.axes[i-1]),q));}robotLinks[i].position.copy(pos);robotLinks[i].quaternion.copy(rot);}const gap=a.gap+(b.gap-a.gap)*u;for(const {sign,g} of fingers)g.position.x=sign*gap;robot.updateMatrixWorld(true);}
function apply(s){updateRobot(s);object.position.copy(s.op);object.quaternion.copy(s.oq);fixture.position.copy(s.fp);fixture.quaternion.copy(s.fq);work(s.k);object.visible=mode!=='fixture';fixture.updateMatrixWorld(true);object.updateMatrixWorld(true);}
function sceneCorners(){
 const points=[];
 for(const root of [fixture,object,robot])root.traverse(m=>{if(!m.isMesh)return;if(!m.geometry.boundingBox)m.geometry.computeBoundingBox();const bb=m.geometry.boundingBox;for(const x of [bb.min.x,bb.max.x])for(const y of [bb.min.y,bb.max.y])for(const z of [bb.min.z,bb.max.z])points.push(V([x,y,z]).applyMatrix4(m.matrixWorld));});
 return points;
}
function videoDirection(){
 // One stationary camera. The task assemblies themselves are arranged with
 // their openings along +Y; no camera or table motion conceals a backwards pose.
 const angle=Math.atan2(1,.28)+az+Math.PI*.75,elevation=el;
 return V([Math.cos(angle)*Math.cos(elevation),Math.sin(angle)*Math.cos(elevation),Math.sin(elevation)]);
}
function basis(dir){const right=new THREE.Vector3().crossVectors(V([0,0,1]),dir).normalize();return {right,up:new THREE.Vector3().crossVectors(dir,right).normalize()};}
function frameVideo(s){
 if(!videoFrame){
  const bounds=new THREE.Box3(),frames=[];
  // Fit the complete motion once, keeping camera position, target and scale
  // constant from the first frame to the last.
  for(const frame of D.motion){
   const pose=animated(frame.t);apply(pose);
   const {right,up}=basis(videoDirection()),points=sceneCorners();
   const xs=points.map(p=>p.dot(right)),ys=points.map(p=>p.dot(up));
   for(const p of points)bounds.expandByPoint(p);
   frames.push({right,up,minX:Math.min(...xs),maxX:Math.max(...xs),minY:Math.min(...ys),maxY:Math.max(...ys)});
  }
  const focus=bounds.getCenter(new THREE.Vector3());let halfWidth=0,halfHeight=0;
  for(const f of frames){const x=focus.dot(f.right),y=focus.dot(f.up);halfWidth=Math.max(halfWidth,Math.abs(f.minX-x),Math.abs(f.maxX-x));halfHeight=Math.max(halfHeight,Math.abs(f.minY-y),Math.abs(f.maxY-y));}
  videoFrame={focus,halfWidth,halfHeight};apply(s);
 }
 return videoFrame;
}
function view(s,rect,video=false,wide=false){
 apply(s);robot.visible=video;
 const bound=new THREE.Box3().setFromObject(fixture);if(object.visible)bound.union(new THREE.Box3().setFromObject(object));
 const focus=bound.getCenter(new THREE.Vector3());
 const angle=az;
 const dir=video?videoDirection():V([Math.cos(angle)*Math.cos(el),Math.sin(angle)*Math.cos(el),Math.sin(el)]);
 const {right,up}=basis(dir),aspect=rect.w/rect.h;let height;
 if(wide&&mode==='video'){
  const framing=frameVideo(s);focus.copy(framing.focus);
  height=2*Math.max(framing.halfHeight,framing.halfWidth/aspect)*1.10*zoom/VIDEO_MAGNIFICATION;
  // Lower the fixed target slightly to keep the base inside the tighter frame.
  focus.addScaledVector(up,-height*.045);
 }else{
  let points=[];
  if(wide){points=sceneCorners();new THREE.Box3().setFromPoints(points).getCenter(focus);}
  else {for(const x of [bound.min.x,bound.max.x])for(const y of [bound.min.y,bound.max.y])for(const z of [0,bound.max.z])points.push(V([x,y,z]));}
  const xs=points.map(p=>p.clone().sub(focus).dot(right)),ys=points.map(p=>p.clone().sub(focus).dot(up)),padding=wide?1.10:1.23;
  height=Math.max((Math.max(...ys)-Math.min(...ys))*padding,(Math.max(...xs)-Math.min(...xs))/aspect*padding)*zoom;
  focus.addScaledVector(right,(Math.max(...xs)+Math.min(...xs))/2).addScaledVector(up,(Math.max(...ys)+Math.min(...ys))/2);
 }
 camera.left=-height*aspect/2;camera.right=height*aspect/2;camera.top=height/2;camera.bottom=-height/2;camera.position.copy(focus).addScaledVector(dir,4);camera.lookAt(focus);camera.updateProjectionMatrix();
 if(video){floor.scale.set(1.20,1.00,1);floor.position.set(.15,.16,-.002);}else{floor.scale.set(Math.max(.26,bound.max.x-bound.min.x+.08),Math.max(.26,bound.max.y-bound.min.y+.08),1);floor.position.set(focus.x,focus.y,-.002);}
 renderer.setSize(Math.round(rect.w),Math.round(rect.h),false);renderer.render(scene,camera);ctx.drawImage(renderer.domElement,rect.x,rect.y,rect.w,rect.h);
}
function draw(){
 ctx.fillStyle='white';ctx.fillRect(0,0,screen.width,screen.height);
 const W=screen.width,H=screen.height;
 if(mode==='overview'){const count=D.poses.length,margin=W*.015,gap=W*.01,w=(W-margin*2-gap*(count-1))/count;for(let i=0;i<count;i++)view(state(i),{x:margin+i*(w+gap),y:H*.035,w,h:H*.93});}
 else if(mode==='fixture')view(state(0),{x:W*.035,y:H*.03,w:W*.93,h:H*.94});
 else {const s=mode==='video'?animated(time):state(Number(mode.slice(-1)));view(s,{x:W*.02,y:H*.025,w:W*.96,h:H*.95},mode==='video',mode==='video');}
 document.getElementById('time').value=time;
}
function setMode(m){mode=m;playing=false;document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===m));draw();}
document.querySelectorAll('[data-mode]').forEach(b=>b.onclick=()=>{setMode(b.dataset.mode);if(mode==='video'){time=0;playing=true;}});
document.getElementById('play').onclick=()=>{if(mode!=='video')setMode('video');playing=!playing;if(time>=D.duration)time=0;};
document.getElementById('time').oninput=e=>{const t=Number(e.target.value);setMode('video');time=t;draw();};
document.getElementById('reset').onclick=()=>{az=-Math.PI*.75;el=.526;zoom=1;videoFrame=null;draw();};
document.getElementById('save').onclick=()=>{const a=document.createElement('a');a.download=`reuse_${mode}.png`;a.href=screen.toDataURL('image/png');a.click();};
let drag=null;screen.onpointerdown=e=>{drag=[e.clientX,e.clientY];screen.setPointerCapture(e.pointerId);};screen.onpointermove=e=>{if(!drag)return;az-=(e.clientX-drag[0])*.007;el=Math.max(.08,Math.min(1.45,el+(e.clientY-drag[1])*.006));drag=[e.clientX,e.clientY];videoFrame=null;draw();};screen.onpointerup=()=>drag=null;
screen.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.65,Math.min(1.7,zoom*Math.exp(e.deltaY*.001)));draw();},{passive:false});
function tick(now){if(playing){time=Math.min(D.duration,time+(now-last)/1000);draw();if(time>=D.duration)playing=false;}last=now;requestAnimationFrame(tick);}requestAnimationFrame(tick);
document.getElementById('time').max=D.duration;
function getState(){
 const k=mode==='video'?sample(time).a.k:mode.startsWith('pose')?Number(mode.slice(-1)):0;
 const opening=V(D.poses[k].withdrawalDirection).applyQuaternion(fq[k].clone().invert()).applyQuaternion(fixture.quaternion),towardCamera=V([0,0,1]).applyQuaternion(camera.quaternion);
 const projected=mode==='video'?sceneCorners().map(p=>p.project(camera)):[];
 let framingMaxAbs=Math.max(0,...projected.map(p=>Math.max(Math.abs(p.x),Math.abs(p.y))));
 if(framingMaxAbs>1){
  // Rotated link bounding boxes contain empty corners. Check actual vertices
  // before reporting clipping in the closer composition.
  framingMaxAbs=0;const p=new THREE.Vector3();
  for(const root of [fixture,object,robot])root.traverse(m=>{if(!m.isMesh)return;const positions=m.geometry.getAttribute('position');for(let i=0;i<positions.count;i++){p.fromBufferAttribute(positions,i).applyMatrix4(m.matrixWorld).project(camera);framingMaxAbs=Math.max(framingMaxAbs,Math.abs(p.x),Math.abs(p.y));}});
 }
 return {mode,time,playing,videoMagnification:VIDEO_MAGNIFICATION,camera:{position:camera.position.toArray(),quaternion:camera.quaternion.toArray(),frustum:[camera.left,camera.right,camera.top,camera.bottom]},floor:{position:floor.position.toArray(),quaternion:floor.quaternion.toArray(),scale:floor.scale.toArray()},openingDirection:opening.toArray(),openingFacingCosine:opening.dot(towardCamera),framingMaxAbs};
}
window.REUSE={setMode,setCamera:(a,e,z=1)=>{az=a;el=e;zoom=z;videoFrame=null;draw();},setTime:t=>{playing=false;mode='video';time=t;draw();},resize:(w,h)=>{screen.width=w;screen.height=h;draw();},draw,getState,canvas:screen};draw();
