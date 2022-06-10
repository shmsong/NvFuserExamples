#include <ATen/core/ivalue.h>
#include <common/manual_scheduled_kernel.h>
#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/utils.h>
#include <torch/extension.h>

#include <memory>

using namespace torch::jit::fuser::cuda;

namespace {

TensorView *makeConcreteTensor(std::vector<int64_t> shape,
                               DataType dtype = DataType::Float) {
  return TensorViewBuilder().shape(shape).dtype(dtype).contiguity(std::vector<bool>(shape.size(), true)).build();
}

class ExampleMul0 : public NVFuserOpBase<ExampleMul0> {
public:
  c10::optional<LaunchParams> fusedKernel(Fusion &fusion,
                                          std::vector<c10::IValue> inputs) {
    auto a = makeConcreteTensor({128,128});
    auto b = makeConcreteTensor({128,128});
    
    fusion.addInput(a);
    fusion.addInput(b);

    auto out = mul(a,b);

    fusion.addOutput(out);

    out->split(-1, 16);
    // [128, 8, 16]

    out->axis(-1)->parallelize(ParallelType::TIDx);
    out->axis(-2)->parallelize(ParallelType::TIDy);
    out->axis(-3)->parallelize(ParallelType::BIDx);

    return c10::nullopt;
  }
} _nvfuser_example0_kernel;

} // namespace

at::Tensor nvfuser_example0_mul128_128(const at::Tensor &a, const at::Tensor &b) {
  auto outputs = _nvfuser_example0_kernel.run({a, b});
  return outputs[0];
}

TORCH_LIBRARY(examples, m) {
  m.def("nvfuser_example0_mul128_128", nvfuser_example0_mul128_128);
}

TORCH_LIBRARY_IMPL(examples, CUDA, m) {
  m.impl("nvfuser_example0_mul128_128", nvfuser_example0_mul128_128);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
