"""SK-18 A1 — cuanto cuesta la atencion de DECODE con FlashInfer y KV fp8 en la forma real.

1 request, 4 posiciones (MTP K=3), 12 cabezas Q / 2 KV por placa, head 256, contexto L,
KV fp8 paginada (bloques de 16). Tiempo por llamada del wrapper de decode (una capa).
El paso de decode completo del server se estima con los tok/s y el TAR medidos.
"""
import sys, time, torch, flashinfer
dev = "cuda"; torch.manual_seed(0)
Hq, Hk, D, PAGE = 12, 2, 256, 16
for L in [int(x) for x in (sys.argv[1:] or ["16000", "57000", "100000"])]:
    npag = (L + PAGE - 1) // PAGE
    kv = (torch.randn(npag, 2, PAGE, Hk, D, device=dev) * 0.5).to(torch.float8_e4m3fn)
    ws = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev)
    w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
    B, Q = 1, 4
    qo = torch.tensor([0, Q], dtype=torch.int32, device=dev)
    kvi = torch.tensor([0, npag], dtype=torch.int32, device=dev)
    idx = torch.arange(npag, dtype=torch.int32, device=dev)
    last = torch.tensor([L - (npag - 1) * PAGE], dtype=torch.int32, device=dev)
    w.plan(qo, kvi, idx, last, Hq, Hk, D, PAGE, causal=True, pos_encoding_mode="NONE",
           q_data_type=torch.float16, kv_data_type=torch.float8_e4m3fn)
    q = torch.randn(Q, Hq, D, device=dev, dtype=torch.float16)
    for _ in range(5): w.run(q, kv)
    torch.cuda.synchronize(); t = time.perf_counter(); n = 50
    for _ in range(n): w.run(q, kv)
    torch.cuda.synchronize(); dt = (time.perf_counter() - t) / n
    print(f"L={L:6d}: atencion decode (1 capa, 4 posiciones) {dt*1e3:6.2f} ms  -> x16 capas = {16*dt*1e3:6.1f} ms por paso", flush=True)
    del kv; torch.cuda.empty_cache()
