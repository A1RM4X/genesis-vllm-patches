"""Los tres kernels que reemplazaron a ~20 lanzamientos de torch por paso, contra su referencia.

  preparar_paso_kernel   bits de ancestros a los dos buffers + delta de RoPE + los 3 ancestros
  compactar_kernel       estados ocultos + KV de PN131 (layout fisico) + filas de cinta
  reponer_cadena_kernel  deja los tres buffers en cadena para el borrador
"""
import torch
import vllm._genesis.arbol_borrador as ab
dev = "cuda"; torch.manual_seed(11)
R, K = 6, 8; T = K + 1; SLOTS = 24
g = torch.Generator(device=dev).manual_seed(4)

# ---- arboles al azar por slot, en orden topologico ----
padre_v = torch.full((SLOTS + 1, T), -1, dtype=torch.int32, device=dev)
for s in range(SLOTS + 1):
    for t in range(1, T):
        padre_v[s, t] = int(torch.randint(0, t, (1,), generator=g, device=dev))
bits_v = ab.bits_ancestros(padre_v).contiguous()
prof = torch.zeros(SLOTS + 1, T, dtype=torch.int64, device=dev)
for t in range(1, T):
    prof[:, t] = prof.gather(1, padre_v[:, t:t + 1].long())[:, 0] + 1
delta_v = (prof - torch.arange(T, device=dev)[None]).contiguous()
idx = torch.randperm(SLOTS, generator=g, device=dev)[:R].to(torch.int32)

anc131 = torch.zeros(SLOTS * T, dtype=torch.int32, device=dev)
anc_gdn = torch.zeros_like(anc131)
anc3 = torch.zeros(SLOTS * T, 3, dtype=torch.int32, device=dev)
pos0 = torch.randint(0, 5000, (3, SLOTS * T), dtype=torch.int64, device=dev)
pos = pos0.clone()
ab.preparar_paso_kernel(idx, padre_v, bits_v, delta_v, anc131, anc_gdn, anc3, pos, R, T)
torch.cuda.synchronize()
il = idx.long()
ok = torch.equal(anc131[:R * T], bits_v[il].flatten()) and torch.equal(anc_gdn[:R * T], bits_v[il].flatten())
ok &= torch.equal(pos[:, :R * T], pos0[:, :R * T] + delta_v[il].flatten()[None])
ok &= torch.equal(pos[:, R * T:], pos0[:, R * T:])                    # no toca lo de mas alla
# anc3 de referencia
ref3 = torch.zeros(R * T, 3, dtype=torch.int32, device=dev)
for r in range(R):
    pv = padre_v[il[r]].tolist()
    for t in range(T):
        c, j = [], (pv[t] if t > 0 else -1)
        while len(c) < 3:
            c.append(j); j = pv[j] if j > 0 else j - 1
        ref3[r * T + t] = torch.tensor(c, dtype=torch.int32, device=dev)
ok &= torch.equal(anc3[:R * T], ref3)
print("preparar_paso_kernel == referencia torch:", bool(ok))

# ---- compactar ----
NH, BS, QD = 2, 832, 256
BLK = BS * NH * QD * 2 + BS * NH * 4
NB, D, NCAP = 40, 512, 3
kv = [torch.randint(-128, 127, (NB, BLK), dtype=torch.int8, device=dev) for _ in range(NCAP)]
kv_ref = [x.clone() for x in kv]
kv_addrs = torch.tensor([x.data_ptr() for x in kv], dtype=torch.int64, device=dev)
oc = [torch.randn(R * T, D, dtype=torch.float16, device=dev) for _ in range(2)]
oc_ref = [x.clone() for x in oc]
slot_map = torch.randperm(NB * BS, generator=g, device=dev)[:R * T].to(torch.int32)
cam = torch.full((R, T - 1), -1, dtype=torch.int64, device=dev)
nacc = torch.randint(1, T + 1, (R,), generator=g, device=dev, dtype=torch.int32)
for r in range(R):                     # camino valido y creciente
    prev = 0
    for j in range(int(nacc[r]) - 1):
        prev = int(torch.randint(prev + 1, T - (int(nacc[r]) - 2 - j), (1,), generator=g, device=dev))
        cam[r, j] = prev
filas = torch.arange(K, dtype=torch.int32, device=dev)[None].repeat(R, 1).contiguous()
cinta = torch.zeros(SLOTS + 1, K, dtype=torch.int32, device=dev)
ab.compactar_kernel(idx, cam, nacc, filas, cinta, oc, kv_addrs, slot_map, R, T, (NH, BS, BLK))
torch.cuda.synchronize()
bien = torch.equal(cinta[il + 1], filas)
for r in range(R):
    for j in range(int(nacc[r]) - 1):
        s, d = int(cam[r, j]), j + 1
        for h, h0 in zip(oc, oc_ref):
            bien &= torch.equal(h[r * T + d], h0[r * T + s])
        ss, dd = int(slot_map[r * T + s]), int(slot_map[r * T + d])
        for x, x0 in zip(kv, kv_ref):
            bs_, os_ = ss // BS, ss % BS; bd_, od_ = dd // BS, dd % BS
            KOFF = BS * NH * QD
            bien &= torch.equal(x[bd_, od_ * NH * QD:(od_ + 1) * NH * QD],
                                x0[bs_, os_ * NH * QD:(os_ + 1) * NH * QD])
            bien &= torch.equal(x[bd_, KOFF + od_::BS][:NH * QD], x0[bs_, KOFF + os_::BS][:NH * QD])
            bien &= torch.equal(x[bd_, 2 * KOFF + od_ * NH * 4:2 * KOFF + (od_ + 1) * NH * 4],
                                x0[bs_, 2 * KOFF + os_ * NH * 4:2 * KOFF + (os_ + 1) * NH * 4])
print("compactar_kernel (ocultos + KV + cinta) == referencia:", bool(bien),
      f"| copias {int((nacc - 1).sum())}")

# ---- reponer ----
ab.reponer_cadena_kernel(anc131, anc_gdn, anc3, R, T)
torch.cuda.synchronize()
cad = ((1 << torch.arange(T, device=dev, dtype=torch.int32)) - 1).repeat(R)
ar = torch.arange(T, device=dev, dtype=torch.int32)
r3 = torch.stack([ar - 1, ar - 2, ar - 3], 1).repeat(R, 1)
print("reponer_cadena_kernel == cadena:",
      bool(torch.equal(anc131[:R * T], cad) and torch.equal(anc_gdn[:R * T], cad)
           and torch.equal(anc3[:R * T], r3)))

# ---- aceptar_kernel con indexado por SLOT (idx_map): el hueco que dejo el refactor ----
print("\n--- aceptar_kernel con idx_map ---")
g2 = torch.Generator(device=dev).manual_seed(77)
mal = 0
for rep in range(30):
    # arboles por SLOT, y un lote que los toma en orden arbitrario
    pv_s = torch.full((SLOTS + 1, T), -1, dtype=torch.int32, device=dev)
    for s in range(SLOTS + 1):
        for t in range(1, T):
            pv_s[s, t] = int(torch.randint(0, t, (1,), generator=g2, device=dev))
    idx2 = torch.randperm(SLOTS, generator=g2, device=dev)[:R].to(torch.int32)
    tok = torch.randint(0, 40, (R, T), generator=g2, device=dev, dtype=torch.int64)
    mu = torch.randint(0, 40, (R, T), generator=g2, device=dev, dtype=torch.int64)
    for r in range(R):                              # que el target quiera hijos de verdad
        for t in range(T):
            h = [c for c in range(1, T) if int(pv_s[idx2[r].long(), c]) == t]
            if h and float(torch.rand(1, generator=g2, device=dev)) < 0.7:
                mu[r, t] = tok[r, h[int(torch.randint(0, len(h), (1,), generator=g2, device=dev))]]
    cu2 = (torch.arange(R + 1, device=dev) * T).to(torch.int32)
    s2, n2, c2, f2 = ab.aceptar_kernel(tok.flatten(), mu.flatten(), cu2, pv_s, None, T, K,
                                       idx_map=idx2)
    # referencia: el MISMO arbol, pero pasando ya las filas ordenadas (sin idx_map)
    pv_r = pv_s[idx2.long()].contiguous()
    s3, n3, c3, f3 = ab.aceptar_kernel(tok.flatten(), mu.flatten(), cu2, pv_r, None, T, K)
    ok2 = torch.equal(n2, n3) and torch.equal(c2, c3) and torch.equal(f2, f3) and torch.equal(s2, s3)
    # y contra la referencia en torch
    cam_r, nacc_r, bono_r = ab.aceptar(tok, pv_r.long(), mu)
    ok2 &= torch.equal(n2.long(), nacc_r) and torch.equal(c2, cam_r)
    mal += not ok2
print(f"aceptar_kernel(idx_map) == sin idx_map == referencia torch: {30 - mal}/30")
