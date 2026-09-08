The prefix masking, packing and zero-padding mechanism in codec.py is adapted
from hansung-choi/ROI-JSCC, commit 6f947d1, model/common_component.py and
model/JSCC.py. Its image transforms and fixed ROI allocator are replaced by
Gaussian attribute transforms and external per-Gaussian tiers.

MIT License

Copyright (c) 2025 hansung-choi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

FCGS Spatial_CTX (YihangChen-ee/FCGS, commit 31e59a4) inspired the mathematical
multi-resolution weighted grid splat/query operation. It is reimplemented in
PyTorch here, without copying its CUDA implementation, entropy decoder,
autoregressive schedule or learned weights. This is not an official FCGS or
ROI-JSCC reproduction. The surrounding repository license still applies.
