//--------------------------------------------------------------------------------------
// Copyright (c) XU, Tianchen. All rights reserved.
//--------------------------------------------------------------------------------------

import slangpy;

#define GROUP_SIZE 8
#define MAX_BLUR_RADIUS 20
#define SHARED_MEM_SIZE (GROUP_SIZE + 2 * MAX_BLUR_RADIUS)

#define DIV_UP(x, n)    (((x) + (n) - 1) / (n))

groupshared float3 g_srcs[SHARED_MEM_SIZE][SHARED_MEM_SIZE];
groupshared float3 g_dsts[SHARED_MEM_SIZE][GROUP_SIZE];

float GaussianSigmaFromRadius(float r)
{
    //return (r - 1) / 6.0;
    //return (r + 1) / sqrt(2.0 * log(255.0));
    //return (r + 1) / sqrt(6.0);
    return (r + 1.0f) / 3.0;
}

float Gaussian(float r, float sigma)
{
    const float a = r / sigma;

    return exp(-0.5 * a * a);
}

[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void main(uint2 DTid: SV_DispatchThreadID, uint2 Gid: SV_GroupID, uint2 GTid: SV_GroupThreadID,
          uniform int BlurRadius, uniform Tensor<float3, 2> Src, uniform RWTensor<float3, 2> Dst)
{
    const int2 texSize = int2(Dst.shape[1], Dst.shape[0]);

    //assert(BlurRadius <= MAX_BLUR_RADIUS);
    BlurRadius = min(BlurRadius, MAX_BLUR_RADIUS);

    // Load data into group-shared memory
    const uint n = DIV_UP(SHARED_MEM_SIZE, GROUP_SIZE);
    const int2 uvStart = GROUP_SIZE * (int2)Gid - BlurRadius;
    int i;
    for (i = 0; i < n; ++i)
    {
        const int x = GROUP_SIZE * i + GTid.x;
        if (x < SHARED_MEM_SIZE)
        {
            for (int j = 0; j < n; ++j)
            {
                const int y = GROUP_SIZE * j + GTid.y;
                if (y < SHARED_MEM_SIZE)
                {
                    const int2 uv = clamp(uvStart + int2(x, y), 0, texSize - 1);
                    g_srcs[y][x] = Src.getv(uv);
                }
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();

    const float sigma = GaussianSigmaFromRadius(BlurRadius);
    const int x = GTid.x + BlurRadius;

    // Horizontal filter
    for (uint k = 0; k < n; ++k)
    {
        const uint y = GROUP_SIZE * k + GTid.y;
        if (y >= SHARED_MEM_SIZE) break;

        float4 mu = 0.0;
        for (i = -BlurRadius; i <= BlurRadius; ++i)
        {
            const int xi = x + i;
            const float3 src = g_srcs[y][xi];

            const float w = Gaussian(i, sigma);
            mu.xyz += src * w;
            mu.w += w;
        }

        g_dsts[y][GTid.x] = mu.xyz / mu.w;
    }

    GroupMemoryBarrierWithGroupSync();

    if (any(DTid >= texSize)) return;

    // Vertical filter
    const int y = GTid.y + BlurRadius;

    float4 mu = 0.0;
    for (i = -BlurRadius; i <= BlurRadius; ++i)
    {
        const int yi = y + i;
        const float3 src = g_dsts[yi][GTid.x];

        const float w = Gaussian(i, sigma);
        mu.xyz += src * w;
        mu.w += w;
    }

    Dst.setv(DTid, mu.xyz / mu.w);
}

[numthreads(GROUP_SIZE, GROUP_SIZE, 1)]
void copy(uint2 DTid: SV_DispatchThreadID, uniform Tensor<float3, 2> Src, uniform RWTensor<float3, 2> Dst)
{
    Dst.setv(DTid, Src.getv(DTid));
}
