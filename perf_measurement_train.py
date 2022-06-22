import torch
import mha_manual  # noqa: F401

seql_q = 128
seql_k = 128
hidden_size = 1024
num_heads = 16
head_dim = hidden_size // num_heads
batch = 32
dropout_prob = 0.5
scale = 1./8 # assume d = 64, so scaler is 8

#reference implementation
def reference_mha(q, k, v, pad_mask, dropout_mask):
     out1 = q.matmul(k.permute((0,2,1)))
     out2 = out1 * scale
     out3 = out2 + pad_mask
     out4 = torch.softmax(out3, -1)
     out5 = (out4 / dropout_prob) * dropout_mask
     out6 = torch.matmul(out5, v)
     return out4, out5, out6

fused_mha = torch.ops.mha_manual.fmha_train_nvfuser
fused_mha_saveim = torch.ops.mha_manual.fmha_train_saveim_nvfuser

# test inputs
q = torch.randn((batch*num_heads, seql_q, head_dim), dtype=torch.half, device='cuda')
k = torch.randn((batch*num_heads, seql_k, head_dim), dtype=torch.half, device='cuda')
v = torch.randn((batch*num_heads, seql_k, head_dim), dtype=torch.half, device='cuda')
scale = 1. / 8 # assume d=64
pad_mask = torch.empty((batch*num_heads, seql_q, seql_k),device="cuda").random_(2).to(torch.half) * (-1000)
dropout_mask = torch.empty((batch*num_heads, seql_q, seql_k),device="cuda").random_(2).to(torch.half)

# Utility to run perf measurement
def run_perf_test(op,description,*args):
    # setup timer
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    warm_up_iter = 10
    measure_iter = 1000

    # warm up run
    with torch.no_grad():
        for _ in range(warm_up_iter):
            op(*args)
        start.record()
        for _ in range(measure_iter):
            op(*args)
        stop.record()
    start.synchronize()
    stop.synchronize()

    iter_time_ms = start.elapsed_time(stop) / measure_iter

    print(f"{description} : {iter_time_ms} ms / iter")

run_perf_test(fused_mha, "fused mha no permute", q,k,v, pad_mask, dropout_mask)
run_perf_test(fused_mha_saveim, "fused mha no permute, save im", q,k,v, pad_mask, dropout_mask)
run_perf_test(reference_mha, "eager mode mha no permute", q,k,v, pad_mask, dropout_mask)