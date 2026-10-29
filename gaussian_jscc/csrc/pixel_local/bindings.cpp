#include <torch/extension.h>

torch::Tensor pixel_project_cuda(torch::Tensor xyz, torch::Tensor opacity,
    torch::Tensor scales, torch::Tensor rotations, torch::Tensor sh,
    torch::Tensor mask, torch::Tensor view, torch::Tensor projection,
    torch::Tensor center, int64_t height, int64_t width, double tanx,
    double tany, int64_t degree);
torch::Tensor pixel_render_cuda(torch::Tensor base, torch::Tensor candidate,
    torch::Tensor ranges, torch::Tensor entries, torch::Tensor tiles,
    int64_t height, int64_t width, bool white);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("project", &pixel_project_cuda);
    m.def("render", &pixel_render_cuda);
}
