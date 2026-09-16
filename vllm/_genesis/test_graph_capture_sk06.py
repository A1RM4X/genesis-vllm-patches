import torch
from vllm._genesis.kernels.sk06_mlp_down import mlp_down_gemm
from vllm._genesis.kernels.sk_ops import _sk_gemm

def test_capture():
    device = "cuda:0"
    torch.cuda.set_device(device)
    stream = torch.cuda.Stream()
    
    M, K, N = 4, 8704, 5120
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    b_shifts = torch.randn(K // 128, N // 128, dtype=torch.float32, device=device)
    b_tensor = torch.randint(-128, 127, (K, N), dtype=torch.int8, device=device).t().contiguous().t()
    b_scales = torch.ones(N, dtype=torch.float32, device=device)
    epi = torch.zeros(M, N, dtype=torch.bfloat16, device=device)
    
    # Warmup
    for _ in range(3):
        x_2d = x.reshape(-1, x.shape[-1])
        K_dim = x_2d.shape[-1]
        x_b = x_2d.view(-1, K_dim // 128, 128).float()
        a_sc_b = x_b.abs().amax(dim=(0, 2)).clamp_min(1e-5) / 127.0
        a_i8 = (x_b / a_sc_b[None, :, None]).round().clamp(-128, 127).to(torch.int8).view(x_2d.shape[0], K_dim)
        a_scales = torch.ones(x_2d.shape[0], dtype=torch.float32, device=x.device)
        cur_shifts = b_shifts * a_sc_b[:, None]
        out = _sk_gemm(a_i8, b_tensor, a_scales, b_scales, cur_shifts, epi, 6, torch.bfloat16)
    
    torch.cuda.synchronize()
    print("Warmup succeeded!")
    
    # Now CUDA Graph capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=stream):
        x_2d = x.reshape(-1, x.shape[-1])
        K_dim = x_2d.shape[-1]
        x_b = x_2d.view(-1, K_dim // 128, 128).float()
        a_sc_b = x_b.abs().amax(dim=(0, 2)).clamp_min(1e-5) / 127.0
        a_i8 = (x_b / a_sc_b[None, :, None]).round().clamp(-128, 127).to(torch.int8).view(x_2d.shape[0], K_dim)
        a_scales = torch.ones(x_2d.shape[0], dtype=torch.float32, device=x.device)
        cur_shifts = b_shifts * a_sc_b[:, None]
        out = _sk_gemm(a_i8, b_tensor, a_scales, b_scales, cur_shifts, epi, 6, torch.bfloat16)
    
    torch.cuda.synchronize()
    print("Graph capture succeeded!")
    g.replay()
    torch.cuda.synchronize()
    print("Graph replay succeeded!")

if __name__ == "__main__":
    test_capture()
