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
 
q = torch.randn((batch*num_heads, seql_q, head_dim), dtype=torch.half, device='cuda')
k = torch.randn((batch*num_heads, seql_k, head_dim), dtype=torch.half, device='cuda')
v = torch.randn((batch*num_heads, seql_k, head_dim), dtype=torch.half, device='cuda')
scale = 1. / 8 # assume d=64
pad_mask = torch.empty((batch*num_heads, seql_q, seql_k),device="cuda").random_(2).to(torch.half) * (-1000)
dropout_mask = torch.empty((batch*num_heads, seql_q, seql_k),device="cuda").random_(2).to(torch.half)

# fused_output = torch.ops.mha_manual.fmha_nvfuser(q,k,v,amask)
softmax_o, dropout_o, ctx = reference_mha(q, k, v, pad_mask, dropout_mask)
ctx_fused = torch.ops.mha_manual.fmha_train_nvfuser(q,k,v, pad_mask, dropout_mask)

print(f"ABS diff: {(ctx_fused-ctx).abs().max()}")
assert torch.allclose(ctx_fused, ctx, atol=1e-2)

