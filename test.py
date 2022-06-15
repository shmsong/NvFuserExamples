import torch
import mha_manual  # noqa: F401

seql_q = 128
seql_k = 128
hidden_size = 1024
num_heads = 16
head_dim = hidden_size // num_heads
batch = 32

#reference implementation
def reference_mha(q,k,v,amask):
    p = q.to(torch.float).matmul(k.permute((0,2,1)).to(torch.float))
    # dropout emulation, assume d=64
    p_masked = p / 8 + (1.0 - amask.to(torch.float)) * -10000.0
    s = torch.softmax(p_masked, -1)
    ctx = torch.matmul(s, v.to(torch.float)).to(torch.half)
    return ctx

q = torch.randn((batch*num_heads, seql_q, head_dim), dtype=torch.half, device='cuda')
k = torch.randn((batch*num_heads, seql_k, head_dim), dtype=torch.half, device='cuda')
v = torch.randn((batch*num_heads, seql_k, head_dim), dtype=torch.half, device='cuda')
amask = torch.empty((batch*num_heads, seql_q, seql_k),device="cuda").random_(2).to(torch.half)

fused_output = torch.ops.mha_manual.fmha_nvfuser(q,k,v,amask)
ctx = reference_mha(q,k,v,amask)

print(f"ABS diff: {(ctx-fused_output).abs().max()}")
assert torch.allclose(ctx, fused_output, atol=1e-3)
