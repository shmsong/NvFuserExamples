import torch
import examples  # noqa: F401

a = torch.randn((128,128), device='cuda')
b = torch.randn((128,128), device='cuda')

#reference implementation
def reference(a, b):
    return a*b
    
fused_op = torch.ops.examples.nvfuser_example0_mul128_128

perf_test_args = (a,b)

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

run_perf_test(fused_op, "fuser implementation", *perf_test_args)
run_perf_test(reference, "eager implementation", *perf_test_args)