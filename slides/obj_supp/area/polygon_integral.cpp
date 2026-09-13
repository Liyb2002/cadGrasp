// Polygon area kernels for deterministic position/direction integration.
#include <cmath>
#include <vector>
#include <algorithm>
#include <cstdint>
using std::vector;
struct V  {
    double x,y;
};
using Poly=vector<V>;
static double cross(V a,V b) {
    return a.x*b.y-a.y*b.x;
}
static V sub(V a,V b) {
    return  {
        a.x-b.x,a.y-b.y
    };
}
static double signed_area(const Poly&p) {
    double a=0;
    for(size_t i=0;i<p.size();i++)a+=cross(p[i],p[(i+1)%p.size()]);
    return a*.5;
}
static double area(const Poly&p) {
    return std::abs(signed_area(p));
}
static Poly triangle() {
    return  {
        {
            0,0
        }, {
            1,0
        }, {
            0,1
        }
    };
}
static void clip(Poly&p,double a,double b,double c) {
    if(p.empty())return;
    double mn=1e300,mx=-1e300;
    for(auto x:p) {
        double f=a*x.x+b*x.y+c;
        mn=std::min(mn,f);
        mx=std::max(mx,f);
    }if(mx<=0)return;
    if(mn>0) {
        p.clear();
        return;
    }Poly out;
    out.reserve(p.size()+1);
    V last=p.back();
    double fl=a*last.x+b*last.y+c;
    for(auto x:p) {
        double f=a*x.x+b*x.y+c;
        if((f<=0)!=(fl<=0)) {
            double t=fl/(fl-f);
            out.push_back( {
                last.x+t*(x.x-last.x),last.y+t*(x.y-last.y)
            });
        }if(f<=0)out.push_back(x);
        last=x;
        fl=f;
    }p.swap(out);
}
static Poly intersection(Poly p,const Poly&c) {
    for(size_t i=0;i<c.size()&&!p.empty();i++) {
        V a=c[i],e=sub(c[(i+1)%c.size()],a);
        clip(p,e.y,-e.x,e.x*a.y-e.y*a.x);
    }return p;
}
static vector<Poly> difference(Poly p,const Poly&c) {
    vector<Poly> out;
    for(size_t i=0;i<c.size()&&!p.empty();i++) {
        V a=c[i],e=sub(c[(i+1)%c.size()],a);
        Poly outside=p;
        clip(outside,-e.y,e.x,-e.x*a.y+e.y*a.x);
        if(area(outside)>1e-16)out.push_back(std::move(outside));
        clip(p,e.y,-e.x,e.x*a.y-e.y*a.x);
    }return out;
}
static double mod(double x) {
    x=std::fmod(x,2*M_PI);
    return x<0?x+2*M_PI:x;
}
static void angular(double a,double b,double c,const double*l,double &low,double &high) {
    double u0=l[0],u1=l[1],p0=l[2],p1=l[3];
    double phase=mod(std::atan2(c,b)-p0),radius=std::hypot(b,c);
    double f0=b*std::cos(p0)+c*std::sin(p0),f1=b*std::cos(p1)+c*std::sin(p1);
    double hmax=phase<=p1-p0?radius:std::max(f0,f1),hmin=mod(phase+M_PI)<=p1-p0?-radius:std::min(f0,f1);
    double vals[2];
    for(int j=0;j<2;j++) {
        double h=j?hmax:hmin;
        double g0=a*u0+h*std::sqrt(std::max(0.,1-u0*u0)),g1=a*u1+h*std::sqrt(std::max(0.,1-u1*u1));
        double e=j?std::max(g0,g1):std::min(g0,g1);
        double length=std::hypot(a,h),root=std::abs(a)/std::max(length,1e-300);
        if(a*h>0&&root>=u0&&root<=u1) {
            double val=a>0?length:-length;
            e=j?std::max(e,val):std::min(e,val);
        }vals[j]=e;
    }low=vals[0];
    high=vals[1];
}
extern "C" void cap_bounds(int count,int facets,const double*coeff,const double*limits,double*out) {
    for(int i=0;i<count;i++) {
        Poly low=triangle(),high=triangle();
        for(int h=0;h<facets;h++) {
            double lo[3],hi[3];
            for(int v=0;v<3;v++) {
                const double*c=coeff+((i*facets+h)*3+v)*4;
                angular(c[1],c[2],c[3],limits,lo[v],hi[v]);
                lo[v]+=c[0]-1e-9;
                hi[v]+=c[0]+1e-9;
            }clip(low,hi[1]-hi[0],hi[2]-hi[0],hi[0]);
            clip(high,lo[1]-lo[0],lo[2]-lo[0],lo[0]);
            if(high.empty())break;
        }out[2*i]=2*area(low);
        out[2*i+1]=2*area(high);
    }
}
// Projected blocker vertices hold (barycentric x, y, height).  projection holds
// the barycentric coordinates of the two local tangent vectors for each face.
extern "C" void evaluate(int nd,const double*angles,int nf,int nr,const int*nh,const int64_t*co,const double*coeff,const int*mode,const double*weight,const int*fo,const int*vo,const double*vertices,const double*projection,const double*constant,double*out) {
    for(int a=0;a<nd;a++) {
        double psi=angles[2*a],phi=angles[2*a+1];
        double u=std::cos(psi),ss=std::sin(psi),dx=ss*std::cos(phi),dy=ss*std::sin(phi),sx=dx/u,sy=dy/u;
        double*answer=out+a*(nr+1);
        for(int r=0;r<=nr;r++)answer[r]=constant[r];
        for(int f=0;f<nf;f++) {
            vector<Poly> visible {
                triangle()
            };
            const double*pr=projection+4*f;
            for(int b=fo[f];b<fo[f+1]&&!visible.empty();b++) {
                Poly shadow;
                for(int v=vo[b];v<vo[b+1];v++) {
                    const double*p=vertices+3*v;
                    shadow.push_back( {
                        p[0]+p[2]*(pr[0]*sx+pr[1]*sy),p[1]+p[2]*(pr[2]*sx+pr[3]*sy)
                    });
                }if(signed_area(shadow)<0)std::reverse(shadow.begin(),shadow.end());
                shadow=intersection(std::move(shadow),triangle());
                if(area(shadow)<=1e-16)continue;
                vector<Poly> next;
                for(auto&p:visible) {
                    auto parts=difference(std::move(p),shadow);
                    for(auto&t:parts)next.push_back(std::move(t));
                }visible.swap(next);
            }
            double va=0;
            for(const auto&p:visible)va+=area(p);
            va*=2;
            answer[nr]+=weight[f]*va;
            if(va<=0)continue;
            for(int r=0;r<nr;r++) {
                int m=mode[f*nr+r];
                if(m<0)continue;
                if(m>0) {
                    answer[r]+=weight[f]*va;
                    continue;
                }Poly feasible=triangle();
                for(int h=0;h<nh[r]&&!feasible.empty();h++) {
                    double val[3];
                    for(int v=0;v<3;v++) {
                        const double*c=coeff+co[r]+((f*nh[r]+h)*3+v)*4;
                        val[v]=c[0]+c[1]*u+c[2]*dx+c[3]*dy;
                    }clip(feasible,val[1]-val[0],val[2]-val[0],val[0]);
                }if(feasible.empty())continue;
                double fa=0;
                if(fo[f]==fo[f+1])fa=area(feasible);
                else for(const auto&p:visible)fa+=area(intersection(p,feasible));
                answer[r]+=2*weight[f]*fa;
            }
        }
        const double density=ss/(2*M_PI*(1-std::cos(M_PI/12)));
        for(int r=0;r<=nr;r++)answer[r]*=density;
    }
}
extern "C" double clipped_area(int n,const double*planes) {
    Poly p=triangle();
    for(int i=0;i<n;i++)clip(p,planes[3*i],planes[3*i+1],planes[3*i+2]);
    return 2*area(p);
}
