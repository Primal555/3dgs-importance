#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>

// Compile the existing, unmodified MaskGaussian preprocessing implementation
// into this separate library. Invoke its kernel on PyTorch's current stream.
#include "forward.cu"

namespace {
void check_float(const torch::Tensor& tensor) {
    TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                tensor.scalar_type() == torch::kFloat32,
                "pixel-local requires contiguous CUDA float32 tensors");
}

__global__ void pack_projected(int n, const int* radius, const float2* xy,
    const float* depth, const float4* conic, const float* mask,
    const float* rgb, float* output) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || radius[i] == 0) return;
    float* p = output + 12*i;
    p[0] = xy[i].x; p[1] = xy[i].y; p[2] = depth[i];
    p[3] = conic[i].x; p[4] = conic[i].y; p[5] = conic[i].z;
    p[6] = conic[i].w; p[7] = mask[i];
    p[8] = rgb[3*i]; p[9] = rgb[3*i+1]; p[10] = rgb[3*i+2];
    p[11] = radius[i];
}

__global__ void composite_tiles(const float* base, const float* candidate,
    const int64_t* ranges, const int64_t* entries, const int64_t* tiles,
    int height, int width, float background, float* output) {
    const int slot = blockIdx.x, lane = threadIdx.x;
    const int tile = tiles[slot], gx = (width+15)/16;
    const int x = (tile % gx)*16 + lane%16;
    const int y = (tile / gx)*16 + lane/16;
    const bool valid = x < width && y < height;
    bool done = !valid;
    float trans = 1.f, color[3] = {0.f, 0.f, 0.f};
    __shared__ float points[256][12];
    const int64_t begin = ranges[2*slot], end = ranges[2*slot+1];
    for (int64_t first = begin; first < end; first += 256) {
        if (__syncthreads_count(done) == 256) break;
        const int64_t offset = first + lane;
        if (offset < end) {
            const int64_t entry = entries[offset];
            const float* p = entry >= 0 ? base+12*entry : candidate+12*(-entry-1);
            #pragma unroll
            for (int k = 0; k < 12; ++k) points[lane][k] = p[k];
        }
        __syncthreads();
        const int count = static_cast<int>((end-first) < 256 ? end-first : 256);
        for (int j = 0; j < count && !done; ++j) {
            const float* p = points[j];
            const float dx = p[0]-float(x), dy = p[1]-float(y);
            const float power = -.5f*(p[3]*dx*dx+p[5]*dy*dy)-p[4]*dx*dy;
            if (power > 0.f) continue;
            const float alpha = min(.99f, p[6]*exp(power));
            if (alpha < 1.f/255.f) continue;
            const float next = trans*(1.f-p[7]*alpha);
            if (next < .0001f) { done = true; continue; }
            #pragma unroll
            for (int ch = 0; ch < 3; ++ch)
                color[ch] += p[8+ch]*p[7]*alpha*trans;
            trans = next;
        }
        __syncthreads();
    }
    #pragma unroll
    for (int ch = 0; ch < 3; ++ch)
        output[(slot*3+ch)*256+lane] = valid ? color[ch]+trans*background : 0.f;
}
}

torch::Tensor pixel_project_cuda(torch::Tensor xyz, torch::Tensor opacity,
    torch::Tensor scales, torch::Tensor rotations, torch::Tensor sh,
    torch::Tensor mask, torch::Tensor view, torch::Tensor projection,
    torch::Tensor center, int64_t height, int64_t width, double tanx,
    double tany, int64_t degree) {
    for (const auto& tensor : {xyz, opacity, scales, rotations, sh, mask, view, projection, center})
        check_float(tensor);
    const c10::cuda::CUDAGuard guard(xyz.device());
    const int n = xyz.size(0);
    auto result = torch::zeros({n, 12}, xyz.options());
    if (!n) return result;
    // Every scratch buffer is an owning tensor; no borrowed rasterizer buffer
    // or pointer survives this call. PyTorch allocator tracks the current stream.
    auto radius = torch::zeros({n}, xyz.options().dtype(torch::kInt32));
    auto touched = torch::zeros_like(radius);
    auto xy = torch::zeros({n, 2}, xyz.options());
    auto depth = torch::zeros({n}, xyz.options());
    auto geom_mask = torch::zeros_like(depth);
    auto cov = torch::zeros({n, 6}, xyz.options());
    auto rgb = torch::zeros({n, 3}, xyz.options());
    auto conic = torch::zeros({n, 4}, xyz.options());
    auto clamped = torch::zeros({n, 3}, xyz.options().dtype(torch::kBool));
    auto stream = c10::cuda::getCurrentCUDAStream();
    // Original RasterizeGaussiansCUDA receives float FOV tangents before
    // computing focal lengths. Preserve that rounding order exactly.
    const float tan_fovx = static_cast<float>(tanx);
    const float tan_fovy = static_cast<float>(tany);
    const float focal_x = static_cast<float>(width)/(2.f*tan_fovx);
    const float focal_y = static_cast<float>(height)/(2.f*tan_fovy);
    preprocessCUDA<NUM_CHANNELS><<<(n+255)/256, 256, 0, stream>>>(
        n, degree, sh.size(1), xyz.data_ptr<float>(),
        reinterpret_cast<const glm::vec3*>(scales.data_ptr<float>()), 1.f,
        reinterpret_cast<const glm::vec4*>(rotations.data_ptr<float>()),
        opacity.data_ptr<float>(), mask.data_ptr<float>(), sh.data_ptr<float>(),
        clamped.data_ptr<bool>(), nullptr, nullptr, view.data_ptr<float>(),
        projection.data_ptr<float>(), reinterpret_cast<const glm::vec3*>(center.data_ptr<float>()),
        width, height, tan_fovx, tan_fovy, focal_x, focal_y,
        radius.data_ptr<int>(), reinterpret_cast<float2*>(xy.data_ptr<float>()),
        depth.data_ptr<float>(), geom_mask.data_ptr<float>(), cov.data_ptr<float>(),
        rgb.data_ptr<float>(), reinterpret_cast<float4*>(conic.data_ptr<float>()),
        dim3((width+15)/16, (height+15)/16),
        reinterpret_cast<uint32_t*>(touched.data_ptr<int>()), false);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    pack_projected<<<(n+255)/256, 256, 0, stream>>>(n, radius.data_ptr<int>(),
        reinterpret_cast<const float2*>(xy.data_ptr<float>()), depth.data_ptr<float>(),
        reinterpret_cast<const float4*>(conic.data_ptr<float>()), geom_mask.data_ptr<float>(),
        rgb.data_ptr<float>(), result.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

torch::Tensor pixel_render_cuda(torch::Tensor base, torch::Tensor candidate,
    torch::Tensor ranges, torch::Tensor entries, torch::Tensor tiles,
    int64_t height, int64_t width, bool white) {
    check_float(base); check_float(candidate);
    const c10::cuda::CUDAGuard guard(base.device());
    for (const auto& tensor : {ranges, entries, tiles})
        TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() &&
                    tensor.scalar_type() == torch::kInt64, "CUDA int64 indices required");
    auto output = torch::empty({tiles.numel(), 3, 16, 16}, base.options());
    if (!tiles.numel()) return output;
    composite_tiles<<<tiles.numel(), 256, 0, c10::cuda::getCurrentCUDAStream()>>>(
        base.data_ptr<float>(), candidate.data_ptr<float>(), ranges.data_ptr<int64_t>(),
        entries.data_ptr<int64_t>(), tiles.data_ptr<int64_t>(), height, width,
        white ? 1.f : 0.f, output.data_ptr<float>());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
