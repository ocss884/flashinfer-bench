# flashinfer-bench

A lightweight benchmark project for experimenting with custom CUDA kernels via the `fibench` Python package.

## Quick start
```bash
docker run --gpus all -it ocss884/flashinfer-bench:dev-cu12.9.1
docker run --gpus all -it ocss884/flashinfer-bench:dev-cutile-cu13.1.1
# From the repository root:
cd /sgl-workspace/flashinfer-bench
pip install -e ./fibench
```

## Run Benchmark

Run the GEMM benchmark script from the repo root:

```bash
sys profile --stats=true -o ./nsys --force-overwrite=true python3 bench/gemm.py
```
