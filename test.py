import torch
import mha_manual  # noqa: F401

t = torch.randn((5, 5), device='cuda')
expected = torch.sinh(t)
output = torch.ops.manual_fmha.sinh_nvfuser(t)

print("Expected:", expected)
print("Output:", output)

assert torch.allclose(output, expected)
print("They match!")
