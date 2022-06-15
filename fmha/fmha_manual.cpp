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
  return TensorViewBuilder().shape(shape).dtype(dtype).build();
}

class FMHAKernel : public NVFuserOpBase<FMHAKernel> {
public:
  c10::optional<LaunchParams> fusedKernel(Fusion &fusion,
                                          std::vector<c10::IValue> inputs) {
    const int seql_q = 128;
    const int seql_k = 128;
    const int hidden_size = 1024;
    const int num_heads = 16;
    const int head_dim = hidden_size / num_heads;
    const int batch = 32;

    // Gemm 1:
    const int M1 = seql_q, N1 = seql_k, K1 = head_dim;

    // Gemm 2:
    const int N2 = head_dim, K2 = seql_k;

    // Fusion definition (TN -> TT)
    // [M,K1]
    auto inp = makeConcreteTensor({batch * num_heads, M1, K1}, DataType::Half);
    // Query matrix
    auto qk = makeConcreteTensor({batch * num_heads, N1, K1}, DataType::Half);
    // V matrix
    auto v = makeConcreteTensor({batch * num_heads, K2, N2}, DataType::Half);

    // Mask 1.0 or 0.0
    auto amask =
        makeConcreteTensor({batch * num_heads, M1, N1}, DataType::Half);

    fusion.addInput(inp);
    fusion.addInput(qk);
    fusion.addInput(v);
    fusion.addInput(amask);

    // [B,M,N,K]
    auto tv0b = broadcast(inp, {false, false, true, false});
    auto tv1b = broadcast(qk, {false, true, false, false});
    auto tv2b = broadcast(v, {false, true, false, false});

    // [B,M,K2,R]
    auto tvp = fusedMultiplySum(tv0b, tv1b, {3});

    // Dropout
    // p_masked = p / math.sqrt(d) + (1.0 - amask) * -10000.0
    auto mask_value = mul(sub(IrBuilder::create<Int>(1), amask),
                          IrBuilder::create<Double>(-10000));

    // Assume d = 64 for all these
    auto tvp_scaled = div(tvp, IrBuilder::create<Double>(8));

    // Approximated masking from contrib reference.
    auto tvp_masked = add(tvp_scaled, mask_value);

    // [B, M, N]
    const int kReductionAxis = 2;
    std::vector<bool> broadcast_mask{false, false, true};

    // Inline define softmax for now for scheduling
    auto max_val = max(tvp_masked, {kReductionAxis});
    auto bcast_max = broadcast(max_val, broadcast_mask);
    auto x_max_sub = sub(tvp_masked, bcast_max);
    auto exp_val = exp(x_max_sub);
    auto sum_exp = sum(exp_val, {kReductionAxis});
    auto bcast_sum = broadcast(sum_exp, broadcast_mask);
    auto recip = reciprocal(bcast_sum);
    auto tvpsfm = mul(exp_val, recip);

    auto tvph = castOp(DataType::Half, tvpsfm);

    // Second matmul:
    auto tvpb = broadcast(tvph, {false, false, false, true});

    // TODO: should we just add NN matmul?
    auto tvout = fusedMultiplySum(tvpb, tv2b, {2});

    auto tvouth = castOp(DataType::Half, tvout);

    fusion.addOutput(tvouth);

    // Fusion:
    //  Gemm(M,K2,K1) x Gemm(M,N,K2)
    MatMulTileOptions gemm_tile;

    // TODO: use very small tiles for now since
    //  alias pass is not re-using smem. Fix later.
    gemm_tile.cta_tile = GemmTile(16, 128, 64);

    // These set of kernels concerns in-CTA persistence.
    //  So the whole N-dimension need to fit in one CTA.
    //
    // TODO: also add padded version.
    TORCH_CHECK(gemm_tile.cta_tile.n == N1);

    // Divide N dimension into warps -> 4 warps
    gemm_tile.warp_tile = GemmTile(16, 32, 64);

    // Using Ampere mma macro
    gemm_tile.instruction_tile = GemmTile(16, 8, 16);

    auto mma_builder1 =
        MmaBuilder(MmaOptions::MacroType::Ampere_16_8_16, gemm_tile)
            .layout(MmaOptions::MmaInputLayout::TN);

    mma_builder1.configureMma(tvp);

    // Configure gemm 2
    MatMulTileOptions gemm_tile2;

    // TODO: use very small tiles for now since
    //  alias pass is not re-using smem. Fix later.
    gemm_tile2.cta_tile = GemmTile(16, 64, 128);

    // Dimension check:
    TORCH_CHECK(gemm_tile2.cta_tile.k == gemm_tile.cta_tile.n);

    // Divide N dimension into warps -> 4 warps
    // TODO: need to merge the Rfactor PR to be able
    //  to do In-CTA split K
    gemm_tile2.warp_tile = GemmTile(16, 16, 128);

    // Using Ampere mma macro
    gemm_tile2.instruction_tile = GemmTile(16, 8, 16);

    auto mma_builder2 =
        MmaBuilder(MmaOptions::MacroType::Ampere_16_8_16, gemm_tile2)
            .layout(MmaOptions::MmaInputLayout::TT);

    // Global read for matmul 1
    auto tv0r = inp->cacheAfter();
    auto tv1r = qk->cacheAfter();

    // Gemm 1 main loop read
    auto tv0cw = tv0r->cacheAfter();
    auto tv0cr = tv0cw->cacheAfter(LoadStoreOpType::LdMatrix);
    auto tv1cw = tv1r->cacheAfter();
    auto tv1cr = tv1cw->cacheAfter(LoadStoreOpType::LdMatrix);

    // Gemm 1 accumulator reg
    auto tvpc = tvp->cacheBefore();

    // Softmax conversion:
    // Step 1: Unswizzle after matmul1 and do
    //  the softmax
    auto tvpccr = tvp->cacheAfter();

    // Gemm 2 prolog:

    // Cache the first matmul result
    auto tvphcw = tvph->cacheAfter();
    auto tvphcr = tvphcw->cacheAfter(LoadStoreOpType::LdMatrix);

    // v gmem read
    auto vr = v->cacheAfter();
    auto vcw = vr->cacheAfter();
    auto vcr = vcw->cacheAfter(LoadStoreOpType::LdMatrixTranspose);

    // Schedule matmul 1:
    // ------------------------------------------------------------------

    // CTA tile:
    // [Mo, Mi128, N128]
    // TODO: need padding to support irregular input shape

    tvp->split(-2, gemm_tile.cta_tile.m);

    // [Mo, Mi128, No, Ni128]
    tvp->reorder({{-1, -2}, {-2, -1}});

    // inline into the CTA tile
    // [Mo, Mi128, Ni128]
    inp->computeAt(tvp, -3);
    qk->computeAt(tvp, -3);

    // Schedule K dim for matmul 1:

    // Order K
    //  0   1      2    3
    // [Mo, M128, N128, K]
    tvpc->split(-1, gemm_tile.cta_tile.k);

    //  0   1      2    3  4
    // [Mo, M128, N128, K, Ki32]
    tvpc->reorder({{-2, -4}, {-3, -2}, {-4, -3}});

    // Inline the prolog:
    //  0   1  2   3     4    5
    // [Mo,No, Ko M128, N128, K32]
    tv0r->computeAt(tvpc, -4);
    tv1r->computeAt(tvpc, -4);

    // Make warp tile:
    // -------------------------------------------------------------------------
    // This is short hand utility, nothing more than a bunch of splits and
    // re-orders
    //  to make the warp level tiling.
    scheduler_utils::matmul_utils::scheduleWarpTileWithReduction(tvpc,
                                                                 gemm_tile);

    // Inner dims might have been re-ordered
    if (tvp->axis(-1)->extent()->evaluateInt() == 16) {
      tvp->reorder({{-1, -2}, {-2, -1}});
    }

    // Tile the output cache. This step will eventually be fully automatic.
    // I.e not even needing to call this API.
    scheduler_utils::matmul_utils::scheduleWarpTileWithNoReduction(tvp,
                                                                   gemm_tile);

    //  0   1  2   3  4   5   6   7   8
    // [Mo,No, Ko Mw, Nw, Kw, Mi, Ni, Ki]
    tv0cr->computeAt(tvpc, -4);
    tv1cr->computeAt(tvpc, -4);

    // Schedule gmem read and smem write:
    // ---------------------------------------------------------------------------
    // [Mo,Ko,M,K]
    tv0cw->merge(-2);
    tv0r->merge(-2);

    // 128b vector load from gmem
    scheduler_utils::matmul_utils::scheduleContiguousVectorLoad(tv0cw,
                                                                gemm_tile, 8);

    // 128b vector store into smem
    scheduler_utils::matmul_utils::scheduleContiguousVectorLoad(tv0r, gemm_tile,
                                                                8);

    tv0cw->setMemoryType(MemoryType::Shared);

    // [No,Ko,N,K]
    if (tv1cw->axis(-1)->extent()->evaluateInt() == 128) {
      tv1cw->reorder({{-1, -2}, {-2, -1}});
    }
    if (tv1r->axis(-1)->extent()->evaluateInt() == 128) {
      tv1r->reorder({{-1, -2}, {-2, -1}});
    }
    tv1cw->merge(-2);
    tv1r->merge(-2);
    // [No,Ko,i,wy,wx,v]
    scheduler_utils::matmul_utils::scheduleContiguousVectorLoad(tv1cw,
                                                                gemm_tile, 8);
    scheduler_utils::matmul_utils::scheduleContiguousVectorLoad(tv1r, gemm_tile,
                                                                8);
    tv1cw->setMemoryType(MemoryType::Shared);

    // Schedule mma input
    // ---------------------------------------------------------------------------
    tv0cr->applyMmaSwizzle(
        mma_builder1.operand(MmaOptions::Operand::A).build());
    // [... Mi, Ni, Ki] want [Ni, Mi, Ki]
    // Need to apply mma swizzle to the broadcast tensor to match
    //  the thread data layout.
    tv0b->reorder({{-2, -3}, {-3, -2}});
    tv0b->applyMmaSwizzle(mma_builder1.operand(MmaOptions::Operand::A).build());

    tv1cr->applyMmaSwizzle(
        mma_builder1.operand(MmaOptions::Operand::B).build());
    tv1b->applyMmaSwizzle(mma_builder1.operand(MmaOptions::Operand::B).build());

    // // Schedule mma output
    // //
    // ---------------------------------------------------------------------------
    tvpc->applyMmaSwizzle(
        mma_builder1.operand(MmaOptions::Operand::Accumulator).build());
    tvp->applyMmaSwizzle(
        mma_builder1.operand(MmaOptions::Operand::Accumulator).build());

    // mma_util::WarpMmaSwizzler::scheduleMmaWarpOutput(tvpccw,
    // mma_builder1.build());

    // Put tvp result in smem
    tvp->setMemoryType(MemoryType::Shared);

    // Set the outer most dimension
    tvp->axis(1)->parallelize(ParallelType::BIDx);
    tvpc->axis(1)->parallelize(ParallelType::BIDx);
    tvp->axis(0)->parallelize(ParallelType::BIDy);
    tvpc->axis(0)->parallelize(ParallelType::BIDy);
    tvpc->axis(4)->parallelize(ParallelType::TIDy);
    tvp->axis(3)->parallelize(ParallelType::TIDy);

    // Load un-swizzled value back to do register persistence

    // Softmax on a 16x128 tile:

    // distribute load to 4 warps:
    // [Batch, Block, I16, R128]

    auto schedule_epilog_tv = [&gemm_tile](TensorView *tv) {
      tv->split(1, gemm_tile.cta_tile.m);
      tv->split(-2, 4);
      tv->split(-1, 32);
      tv->axis(1)->parallelize(ParallelType::BIDx);
      tv->axis(0)->parallelize(ParallelType::BIDy);
      tv->axis(-1)->parallelize(ParallelType::TIDx);
      tv->axis(-3)->parallelize(ParallelType::TIDy);
    };

    // Just apply a simple persistent schedule
    schedule_epilog_tv(tvpccr);
    schedule_epilog_tv(mask_value);
    schedule_epilog_tv(tvp_scaled);
    schedule_epilog_tv(tvp_masked);
    schedule_epilog_tv(max_val);
    schedule_epilog_tv(bcast_max);
    schedule_epilog_tv(x_max_sub);
    schedule_epilog_tv(exp_val);
    schedule_epilog_tv(sum_exp);
    schedule_epilog_tv(bcast_sum);
    schedule_epilog_tv(recip);
    schedule_epilog_tv(tvpsfm);
    schedule_epilog_tv(tvph);

    // Put tvph result into smem:
    schedule_epilog_tv(tvphcw);

    tvphcw->setMemoryType(MemoryType::Shared);

    // order to MNK: (TT original is MKN)
    tvout->reorder({{-1, -2}, {-2, -1}});

    mma_builder2.configureMma(tvout);

    // Accumulator register:
    auto tvout_acc = tvout->cacheBefore();

    tvout->split(-2, gemm_tile2.cta_tile.m);
    // [Mo, M16, N64]
    tvouth->split(-2, gemm_tile2.cta_tile.m);

    tvout_acc->split(-3, gemm_tile2.cta_tile.m);

    // [Mo, M16, K128, N64]
    vr->computeAt(tvout_acc, -4);
    tvphcr->computeAt(tvout_acc, -4);

    // Schedule tvout accumulator:
    scheduler_utils::matmul_utils::scheduleWarpTileWithReduction(tvout_acc,
                                                                 gemm_tile2);

    // Tile the output
    scheduler_utils::matmul_utils::scheduleWarpTileWithNoReduction(tvouth,
                                                                   gemm_tile2);

    // Schedule v gmem read:
    vcw->merge(-2);
    vr->merge(-2);
    scheduler_utils::matmul_utils::scheduleContiguousVectorLoad(vr, gemm_tile2,
                                                                8);

    scheduler_utils::matmul_utils::scheduleContiguousVectorLoad(vcw, gemm_tile2,
                                                                8);

    vcw->setMemoryType(MemoryType::Shared);

    vcr->computeAt(tvout_acc, -4);

    tvphcr->computeAt(tvout_acc, -4);

    tvphcr->applyMmaSwizzle(
        mma_builder2.operand(MmaOptions::Operand::A).build());

    tvpb->reorder({{-2, -3}, {-3, -2}});
    tvpb->applyMmaSwizzle(mma_builder2.operand(MmaOptions::Operand::A).build());

    vcr->applyMmaSwizzle(mma_builder2.operand(MmaOptions::Operand::B).build());
    tv2b->applyMmaSwizzle(mma_builder2.operand(MmaOptions::Operand::B).build());

    tvout_acc->applyMmaSwizzle(
        mma_builder2.operand(MmaOptions::Operand::Accumulator).build());

    tvouth->applyMmaSwizzle(
        mma_builder2.operand(MmaOptions::Operand::Accumulator).build());
    auto tvouthc = tvouth->cacheBefore();

    tvout->computeAt(tvouth, -2);
    amask->computeAt(tvp_masked, -1);

    // Lift the serial reduction part
    max_val->rFactor({-2});
    sum_exp->rFactor({-2});

    // Parallelize:
    tvout_acc->axis(0)->parallelize(ParallelType::BIDy);
    tvout_acc->axis(1)->parallelize(ParallelType::BIDx);
    tvout_acc->axis(3)->parallelize(ParallelType::TIDy);

    tvout->axis(0)->parallelize(ParallelType::BIDy);
    tvout->axis(1)->parallelize(ParallelType::BIDx);
    tvout->axis(3)->parallelize(ParallelType::TIDy);

    tvouthc->axis(0)->parallelize(ParallelType::BIDy);
    tvouthc->axis(1)->parallelize(ParallelType::BIDx);
    tvouthc->axis(3)->parallelize(ParallelType::TIDy);

    tvouth->axis(0)->parallelize(ParallelType::BIDy);
    tvouth->axis(1)->parallelize(ParallelType::BIDx);
    tvouth->axis(3)->parallelize(ParallelType::TIDy);
    tvouth->axis(-2)->parallelize(ParallelType::TIDx);
    tvouth->axis(-1)->parallelize(ParallelType::Vectorize);

    return c10::nullopt;
  }
} _fmha_kernel;

} // namespace

at::Tensor fmha_nvfuser(const at::Tensor &input, const at::Tensor &qk,
                        const at::Tensor &v, const at::Tensor &amask) {
  auto outputs = _fmha_kernel.run({input, qk, v, amask});
  return outputs[0];
}

TORCH_LIBRARY(mha_manual, m) { m.def("fmha_nvfuser", fmha_nvfuser); }

TORCH_LIBRARY_IMPL(mha_manual, CUDA, m) {
  m.impl("fmha_nvfuser", fmha_nvfuser);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}
