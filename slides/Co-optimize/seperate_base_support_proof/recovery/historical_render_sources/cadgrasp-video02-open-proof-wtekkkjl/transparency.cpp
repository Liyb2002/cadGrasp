#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
struct Layer{float z,a,r,g,b;};
extern "C" void raster_layers(const float* xyz,const float* rgb,int count,int width,int height,const float* alpha,uint8_t* pixels){
 constexpr int K=4;int size=width*height;float far=-std::numeric_limits<float>::infinity();
 std::fill(pixels,pixels+size*3,uint8_t(255));std::vector<float> depth(size,far);std::vector<Layer> layers(size*K,Layer{far,0,255,255,255});
 for(int pass=0;pass<2;++pass)for(int f=0;f<count;++f){
  if((alpha[f]<1.f)!=(pass==1)||alpha[f]<=0)continue;
  const float *a=xyz+9*f,*b=a+3,*c=a+6,*col=rgb+9*f;
  float det=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1]);if(std::abs(det)<1e-8f)continue;
  int x0=std::max(0,int(std::floor(std::min({a[0],b[0],c[0]})))),x1=std::min(width-1,int(std::ceil(std::max({a[0],b[0],c[0]}))));
  int y0=std::max(0,int(std::floor(std::min({a[1],b[1],c[1]})))),y1=std::min(height-1,int(std::ceil(std::max({a[1],b[1],c[1]}))));
  float ux=(b[1]-c[1])/det,uy=(c[0]-b[0])/det,vx=(c[1]-a[1])/det,vy=(a[0]-c[0])/det;
  for(int y=y0;y<=y1;++y){float u=ux*(x0+.5f-c[0])+uy*(y+.5f-c[1]),v=vx*(x0+.5f-c[0])+vy*(y+.5f-c[1]);
   for(int x=x0;x<=x1;++x,u+=ux,v+=vx){float w=1-u-v;if(u< -1e-5f||v< -1e-5f||w< -1e-5f)continue;
    int at=y*width+x;float z=u*a[2]+v*b[2]+w*c[2];if(z<=depth[at])continue;
    float color[3];for(int j=0;j<3;++j)color[j]=std::clamp(u*col[j]+v*col[j+3]+w*col[j+6],0.f,255.f);
    if(pass==0){depth[at]=z;for(int j=0;j<3;++j)pixels[3*at+j]=uint8_t(color[j]);}
    else{Layer* p=&layers[K*at];int target=K;bool dup=false;
     for(int j=0;j<K;++j){if(std::abs(z-p[j].z)<2e-7f){dup=true;break;}if(target==K&&z>p[j].z)target=j;}
     if(dup||target==K)continue;for(int j=K-1;j>target;--j)p[j]=p[j-1];p[target]=Layer{z,alpha[f],color[0],color[1],color[2]};}
   }
  }
 }
 for(int at=0;at<size;++at){float color[3]={float(pixels[3*at]),float(pixels[3*at+1]),float(pixels[3*at+2])};
  for(int j=K-1;j>=0;--j){const Layer&p=layers[K*at+j];color[0]=p.a*p.r+(1-p.a)*color[0];color[1]=p.a*p.g+(1-p.a)*color[1];color[2]=p.a*p.b+(1-p.a)*color[2];}
  for(int j=0;j<3;++j)pixels[3*at+j]=uint8_t(std::clamp(color[j],0.f,255.f));}
}
