Examples of manually fused kernels with nvfuser, currently only fused multihead attention forward pass.

# Build

```
python setup.py install
```

# Test

```
python test.py
```

# See perf

```
python perf_measurement.py 
```

# See generated kernel

```
PYTORCH_NVFUSER_DUMP=cuda_kernel python test.py
```

