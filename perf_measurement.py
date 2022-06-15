import torch
import mha_manual  # noqa: F401

seql_q = 128
seql_k = 128
hidden_size = 1024
num_heads = 16
head_dim = hidden_size // num_heads
batch = 32

#reference implementation
def eager_mha(q,k,v,amask):
    p = q.matmul(k.permute((0,2,1)))
    # dropout emulation, assume d=64
    p_masked = p / 8 + (1.0 - amask) * -10000.0
    s = torch.softmax(p_masked, -1)
    ctx = torch.matmul(s, v)
    return ctx

fused_mha = torch.ops.mha_manual.fmha_nvfuser

# test inputs
q = torch.randn((batch*num_heads, seql_q, head_dim), dtype=torch.half, device='cuda')
k = torch.randn((batch*num_heads, seql_k, head_dim), dtype=torch.half, device='cuda')
v = torch.randn((batch*num_heads, seql_k, head_dim), dtype=torch.half, device='cuda')
amask = torch.empty((batch*num_heads, seql_q, seql_k),device="cuda").random_(2).to(torch.half)

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

run_perf_test(fused_mha, "fused mha no permute", q,k,v,amask)
run_perf_test(eager_mha, "eager mode mha no permute", q,k,v,amask)