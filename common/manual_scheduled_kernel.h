
using namespace torch::jit::fuser::cuda;

namespace torch {
namespace jit {
namespace fuser {
namespace cuda {

template <typename NvFuserOp> class NVFuserOpBase {
public:
  std::vector<at::Tensor> run(std::vector<c10::IValue> input) {
    if (!compiled_kernel_.compiled()) {
      compileFusion(input);
    }
    return compiled_kernel_.runFusion(input, launch_constraint_);
  }

  c10::optional<LaunchParams> fusedKernel(Fusion &fusion,
                                          std::vector<c10::IValue> input) {
    TORCH_CHECK(false, "Undefined nvfuser op");
    return c10::nullopt;
  }

private:
  void compileFusion(std::vector<c10::IValue> input) {
    Fusion fusion;
    FusionGuard fg(&fusion);

    // Compose and schedule the fusion
    auto maybe_lparams = makeFusion(fusion, input);

    if (maybe_lparams.has_value()) {
      launch_constraint_ = maybe_lparams.value();
    }

    // Compile fusion into kernel
    compiled_kernel_.compileFusion(&fusion, input, launch_constraint_);
  }

  // Internal CRTP dispatcher.
  c10::optional<LaunchParams> makeFusion(Fusion &fusion,
                                         std::vector<c10::IValue> input) {
    NvFuserOp &nvfuser_op = static_cast<NvFuserOp &>(*this);
    return nvfuser_op.fusedKernel(fusion, input);
  }

private:
  FusionExecutor compiled_kernel_;
  LaunchParams launch_constraint_;
};

} // namespace cuda
} // namespace fuser
} // namespace jit
} // namespace torch
