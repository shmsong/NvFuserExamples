import torch
import examples  # noqa: F401

a = torch.randn((128,128), device='cuda')
b = torch.randn((128,128), device='cuda')

fused_output = torch.ops.examples.nvfuser_example0_mul128_128(a,b)

print(f"max abs diff {(fused_output-a*b).abs().max()}")
assert fused_output.allclose(a*b)
