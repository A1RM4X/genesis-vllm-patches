import sys; sys.argv=[sys.argv[0]]
src=open("/p/sk18d_bloques.py").read().split('DELTAS = ')[0]
exec(src)
q, k, v = A.cargar(35); kr = A.rotar_ptx(k)
p0=20000; pos=torch.arange(p0,p0+4,device=dev); qr=A.rotar_ptx(q[pos])
p=preparar(qr[:, :6].reshape(-1,D), kr[:p0+4,0], v[:p0+4,0], pos.repeat_interleave(6), 512)
oi,sw,nsel=cadena(p,None)
print("Sz", p["Sz"][0,:8].tolist(), p["Sz"][0,p0-2:p0+6].tolist())
print("oi abs sum", oi.abs().sum().item(), "sw", sw[:6].tolist(), "nsel", nsel.item())
print("out nz", (p["out"]!=0).sum().item(), "lo nz", (p["lo"]!=0).sum().item(), "sumw nz", (p["sumw"]!=0).sum().item())
