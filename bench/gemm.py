import cuda.tile as ct
from fibench.cutile.gemm import ct_matmul
import torch


def bench_gemm():
    M, N, K = 4096, 4096, 4096
    tileM, tileN, tileK = 256, 256, 64
    A = torch.randn(size=[M, K], dtype=torch.bfloat16, device="cuda")
    B = torch.randn(size=[K, N], dtype=torch.bfloat16, device="cuda")
    C = torch.randn(size=[M, N], dtype=torch.float32, device="cuda")
    
    grid = (ct.cdiv(M, tileM), ct.cdiv(N, tileN))
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        ct_matmul,
        (A, B, C, False, False, tileM, tileN, tileK)
    )
    
    real = A @ B
    print(C - real)

if __name__ == "__main__":
    bench_gemm()