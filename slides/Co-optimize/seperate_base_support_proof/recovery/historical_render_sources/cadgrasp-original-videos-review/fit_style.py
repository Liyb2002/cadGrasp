from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

ROOT=Path('/home/yli581/Desktop/cadGrasp')
FOLDER=Path('/tmp/cadgrasp-original-videos-review')
DATA=ROOT/'slides/Co-optimize/seperate_base_support_proof/output/B/illegal/pose11+19/step5/step5.1/data'

def boundary(xy,n=500):
    ends=np.roll(xy,-1,axis=0)
    lengths=np.linalg.norm(ends-xy,axis=1)
    cumulative=np.r_[0,np.cumsum(lengths)]
    distance=np.linspace(0,cumulative[-1],n,endpoint=False)
    i=np.searchsorted(cumulative,distance,side='right')-1
    return xy[i]+((distance-cumulative[i])/lengths[i])[:,None]*(ends[i]-xy[i])

def project(xy,x):
    phi,theta,scale,dx,dy=x
    matrix=scale*np.array([[-np.sin(phi),np.cos(phi)],
                          [np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi)]])
    return xy@matrix.T+[dx,dy]

def fit(pose,index,purple):
    im=cv2.imread(str(FOLDER/f'02_ground_collision_{index}.png'))
    rgb=im[:,:,::-1].astype(float)
    r,g,b=rgb.transpose(2,0,1)
    if purple: mask=(b>r+9)&(r>g+7)&(r<180)
    else: mask=(r>g+16)&(g>b+10)&(r<225)&(g>65)
    coordinates=np.argwhere(mask)[:,::-1].astype(float)
    cv2.imwrite(str(FOLDER/f'mask_{pose}.png'),mask.astype('uint8')*255)
    with np.load(DATA/f'pose_{pose}.npz') as d:
        hull=d['floor_demands_world_m'][d['hull_indices'],:2]
        cloud=d['floor_demands_world_m'][::64,:2]
    samples=boundary(hull)
    tree=cKDTree(coordinates)
    def residual(x):
        projected=project(samples,x)
        return np.r_[tree.query(projected)[0],cKDTree(projected).query(coordinates[::4])[0]]
    edge=np.roll(hull,-1,axis=0)-hull
    e=edge[np.argmax(np.linalg.norm(edge,axis=1))]
    phi=np.arctan2(-e[0],e[1]) if purple else -1.14
    best=None
    for angle in np.linspace(-np.pi,np.pi,12,endpoint=False):
        matrix=np.array([[-np.sin(angle),np.cos(angle)], [.48*np.cos(angle),.48*np.sin(angle)]])
        q=samples@matrix.T
        scale=np.ptp(coordinates,axis=0)[0]/np.ptp(q,axis=0)[0]
        shift=coordinates.mean(0)-scale*q.mean(0)
        x=[angle,.45,scale,*shift]
        result=least_squares(residual,x,bounds=([-7,.05,500,-2000,-2000],[7,1.3,20000,3000,3000]),max_nfev=140)
        err=float(np.mean(residual(result.x)**2))
        if best is None or err<best[0]:best=(err,result.x)
    print(pose,len(coordinates),best,flush=True)
    out=im.copy()
    for x,y in project(cloud,best[1]):
        cv2.circle(out,(round(x),round(y)),1,(54,130,223) if not purple else (171,94,137),-1)
    cv2.imwrite(str(FOLDER/f'fit_{pose}.png'),out)
    return best

def registration():
    a=cv2.imread(str(FOLDER/'01_hulls_and_support_0.png'))
    b=cv2.imread(str(FOLDER/'04_work_cones_0.png'))
    feature=cv2.SIFT_create(nfeatures=2500)
    boxes=[(0,0,900,900),(900,0,350,450),(1250,0,350,450),(900,450,350,450),(1250,450,350,450)]
    for x,y,w,h in boxes:
        ia=a[y:y+h,x:x+w];ib=b[y:y+h,x:x+w]
        ka,da=feature.detectAndCompute(ia,None);kb,db=feature.detectAndCompute(ib,None)
        matches=cv2.BFMatcher().knnMatch(da,db,k=2)
        good=[m for m,n in matches if m.distance<.75*n.distance]
        pa=np.array([ka[m.queryIdx].pt for m in good]);pb=np.array([kb[m.trainIdx].pt for m in good])
        matrix,inliers=cv2.estimateAffinePartial2D(pa,pb,method=cv2.RANSAC,ransacReprojThreshold=2)
        print('panel',x,y,'matches',len(good),'affine',matrix,'inliers',sum(inliers)[0],flush=True)

if __name__=='__main__':
    fit(19,0,True)
    fit(11,3,False)
    registration()
