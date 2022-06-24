Examples of manually fused kernels with nvfuser. Be sure to have a cuda-enabled pytorch installed before building this one.

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

