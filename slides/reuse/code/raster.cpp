// Opaque triangle rasterizer for deterministic offline PNG/MP4 export.
// No graphics context, browser, GPU, or window system is required.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

extern "C" void raster(const float* xyz, const float* rgb, int count,
                       int width, int height, uint8_t* pixels) {
    std::fill(pixels, pixels + width * height * 3, uint8_t(255));
    std::vector<float> depth(width * height, -std::numeric_limits<float>::infinity());
    for (int f = 0; f < count; ++f) {
        const float *a=xyz+9*f, *b=a+3, *c=a+6, *col=rgb+9*f;
        float det=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1]);
        if (std::abs(det)<1e-8f) continue;
        int x0=std::max(0, int(std::floor(std::min({a[0],b[0],c[0]}))));
        int x1=std::min(width-1, int(std::ceil(std::max({a[0],b[0],c[0]}))));
        int y0=std::max(0, int(std::floor(std::min({a[1],b[1],c[1]}))));
        int y1=std::min(height-1, int(std::ceil(std::max({a[1],b[1],c[1]}))));
        float ux=(b[1]-c[1])/det, uy=(c[0]-b[0])/det;
        float vx=(c[1]-a[1])/det, vy=(a[0]-c[0])/det;
        for (int y=y0; y<=y1; ++y) {
            float u=ux*(x0+.5f-c[0])+uy*(y+.5f-c[1]);
            float v=vx*(x0+.5f-c[0])+vy*(y+.5f-c[1]);
            for (int x=x0; x<=x1; ++x, u+=ux, v+=vx) {
                float w=1-u-v;
                if (u < -1e-5f || v < -1e-5f || w < -1e-5f) continue;
                float z=u*a[2]+v*b[2]+w*c[2];
                int at=y*width+x;
                if (z<=depth[at]) continue;
                depth[at]=z;
                for (int j=0; j<3; ++j)
                    pixels[3*at+j]=uint8_t(std::clamp(u*col[j]+v*col[j+3]+w*col[j+6],0.f,255.f));
            }
        }
    }
}
