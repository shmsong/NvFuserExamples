#include <torch/csrc/jit/codegen/cuda/arith.h>
#include <torch/csrc/jit/codegen/cuda/executor.h>
#include <torch/csrc/jit/codegen/cuda/scheduler/all_schedulers.h>
#include <ATen/core/ivalue.h>
#include <torch/extension.h>
#include <common/manual_scheduled_kernel.h>

#include <memory>

using namespace torch::jit::fuser::cuda;

namespace{
    
    class SinhKernel : public NVFuserOpBase<SinhKernel>{
        public:
            c10::optional<LaunchParams> fusedKernel(Fusion& fusion, std::vector<c10::IValue> input){
                int dim = input[0].toTensor().dim();
                auto dtype = input[0].toTensor().scalar_type();
                auto x =
                    TensorViewBuilder().ndims(dim).dtype(aten_to_data_type(dtype)).build();
                fusion.addInput(x);

                // Using equation sinh(x) = [ exp(x) - exp(-1) ] / 2
                auto output = div(sub(exp(x), exp(neg(x))), IrBuilder::create<Double>(2.0));
                fusion.addOutput(output);

                std::cout << "Create fusion:" << std::endl;
                fusion.print();

                auto lparams = schedulePointwise(&fusion, input);

                return lparams;
            }
    } _sinh_kernel;

} // namespace

at::Tensor sinh_nvfuser(const at::Tensor& input) {
  auto outputs = _sinh_kernel.run({input});
  return outputs[0];
}

TORCH_LIBRARY(manual_fmha, m) {
  m.def("sinh_nvfuser", sinh_nvfuser);
}

TORCH_LIBRARY_IMPL(manual_fmha, CUDA, m) {
  m.impl("sinh_nvfuser", sinh_nvfuser);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
