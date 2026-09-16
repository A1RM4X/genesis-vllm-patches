# SPDX-License-Identifier: Apache-2.0
"""TDD para PN118 — conversion AWQ int4 -> INT8 per-canal en la carga.

Lo que se protege:

1. **El orden de bits del desempaquetado.** `int8_mlp_weights` reimplementa
   `compressed_tensors.unpack_from_int32` para no depender del paquete en
   tiempo de carga. Si los dos se separan, los pesos salen permutados y el
   modelo escupe basura SIN fallar: el test compara contra el original.

2. **Que el zero point se desempaquete por la dimension correcta.** Es el error
   facil de este formato: `weight_packed` va empaquetado a lo largo de K y
   `weight_zero_point` a lo largo de N. Confundirlos da un resultado plausible
   pero equivocado.

3. **Que la conversion por bloques de filas de el mismo resultado** que hacerla
   de una. El bloqueo existe para acotar el pico de memoria (una gate_up en
   fp32 son 356 MB), y es facil equivocarse en el recorte del zp, que se
   redondea a multiplos de 8 filas.

4. **Que PN118 no toque nada si el checkpoint ya viene en W8A8.**
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
conv = pytest.importorskip("vllm._genesis.int8_mlp_weights")


def _empaquetar_k(w4: "torch.Tensor") -> "torch.Tensor":
    """[N, K] valores 0..15 -> [N, K/8] int32, orden de compressed-tensors."""
    N, K = w4.shape
    out = torch.zeros((N, K // 8), dtype=torch.int32)
    for i in range(8):
        out |= (w4[:, i::8].to(torch.int32) & 0xF) << (4 * i)
    return out


def _empaquetar_n(v: "torch.Tensor") -> "torch.Tensor":
    """[N, G] valores 0..15 -> [N/8, G] int32, empaquetado a lo largo de N."""
    N, G = v.shape
    out = torch.zeros((N // 8, G), dtype=torch.int32)
    for i in range(8):
        out |= (v[i::8, :].to(torch.int32) & 0xF) << (4 * i)
    return out


def _caso(N=64, K=256, semilla=0):
    g = torch.Generator().manual_seed(semilla)
    G = K // conv.GROUP
    w4 = torch.randint(0, 16, (N, K), generator=g, dtype=torch.int32)
    zp = torch.randint(0, 16, (N, G), generator=g, dtype=torch.int32)
    sc = (torch.rand(N, G, generator=g) * 0.02 + 0.001).to(torch.bfloat16)
    return w4, zp, sc, _empaquetar_k(w4), _empaquetar_n(zp)


def test_desempaquetado_coincide_con_compressed_tensors():
    """Si se separa del upstream, los pesos salen permutados en silencio."""
    ct = pytest.importorskip(
        "compressed_tensors.compressors.pack_quantized.helpers")
    w4, zp, sc, wp, zpp = _caso()
    N, K = w4.shape
    mio = conv._desempaquetar(wp, torch.Size((N, K)), dim=1)
    suyo = ct.unpack_from_int32(wp, 4, torch.Size((N, K)), packed_dim=1)
    assert torch.equal(mio.int(), suyo.int())


def test_zero_point_se_desempaqueta_por_N_no_por_K():
    """weight_packed va por K y weight_zero_point por N. Confundirlos da un
    resultado plausible pero equivocado."""
    w4, zp, sc, wp, zpp = _caso()
    N, G = zp.shape
    assert zpp.shape == (N // 8, G)
    # +8 porque _desempaquetar aplica el offset de signo de upstream
    rec = conv._desempaquetar(zpp, torch.Size((N, G)), dim=0)
    assert torch.equal(rec.int() + 8, zp)


def test_el_offset_de_signo_se_cancela_en_la_resta():
    """`unpack_from_int32` resta 8 para pasar de 0..15 a -8..7. Eso NO cambia
    la dequantizacion porque se le resta lo mismo al peso y al zero point:
    (w4-8) - (zp-8) = w4 - zp. El test lo deja escrito para que nadie "arregle"
    el offset creyendo que corrige un bug."""
    w4, zp, sc, wp, zpp = _caso()
    N, K = w4.shape
    G = sc.shape[1]
    w_s = conv._desempaquetar(wp, torch.Size((N, K)), dim=1).float()
    zp_s = conv._desempaquetar(zpp, torch.Size((N, G)), dim=0).float()
    assert w_s.min() < 0, "no aplico el offset de signo"
    dif_con = w_s - zp_s.repeat_interleave(K // G, dim=1)
    dif_sin = w4.float() - zp.float().repeat_interleave(K // G, dim=1)
    assert torch.equal(dif_con, dif_sin)


def test_dequantizar_reconstruye_el_peso_original():
    w4, zp, sc, wp, zpp = _caso()
    N, K = w4.shape
    G = sc.shape[1]
    esperado = (w4.float() - zp.float().repeat_interleave(K // G, dim=1)) \
        * sc.float().repeat_interleave(K // G, dim=1)
    obtenido = conv.dequantizar(wp, sc, zpp)
    assert torch.allclose(obtenido, esperado, atol=1e-5)


def test_int8_per_canal_usa_todo_el_rango():
    """La escala tiene que saturar en 127 para el maximo de cada canal."""
    w = torch.randn(32, 512)
    w8, s = conv.a_int8_per_canal(w)
    assert w8.dtype is torch.int8
    assert s.shape == (32,)
    assert w8.abs().amax(dim=1).min() >= 126


def test_conversion_por_bloques_da_lo_mismo_que_de_una():
    """El bloqueo acota el pico de memoria; el recorte del zp se redondea a
    multiplos de 8 filas y es donde se rompe."""
    w4, zp, sc, wp, zpp = _caso(N=128, K=256)
    a8, ae = conv.convertir(wp, sc, zpp, filas=128)
    b8, be = conv.convertir(wp, sc, zpp, filas=16)
    assert torch.equal(a8, b8)
    assert torch.allclose(ae, be)


def test_el_error_de_conversion_es_chico():
    """~1% es lo medido en el checkpoint real; 5% seria una regresion."""
    w4, zp, sc, wp, zpp = _caso(N=256, K=512)
    e = conv.medir_error(wp, sc, zpp)
    assert e["err_medio"] < 0.05, f"error {e['err_medio']:.4f} demasiado alto"
    assert e["rango_escalas"] > 0


def test_el_cache_esta_prendido_por_defecto(monkeypatch):
    monkeypatch.delenv("GENESIS_INT8_CACHE", raising=False)
    assert conv._cache_activo() is True


def test_el_cache_ida_y_vuelta(tmp_path, monkeypatch):
    monkeypatch.setenv("GENESIS_INT8_CACHE", "1")
    monkeypatch.setenv("GENESIS_INT8_CACHE_DIR", str(tmp_path))
    w4, zp, sc, wp, zpp = _caso()
    a8, ae = conv.obtener(wp, sc, zpp, modelo="m", capa="c", rank=0)
    assert list(tmp_path.glob("*.pt")), "no escribio el cache"
    b8, be = conv.obtener(wp, sc, zpp, modelo="m", capa="c", rank=0)
    assert torch.equal(a8, b8) and torch.allclose(ae, be)


def test_es_opt_in_y_esta_registrado():
    from vllm._genesis.dispatcher import PATCH_REGISTRY
    meta = PATCH_REGISTRY["PN118"]
    assert meta["default_on"] is False
    assert meta["env_flag"] == "GENESIS_ENABLE_PN118_INT8_MLP"


class Capa:
    def __init__(self, pref): self.prefix = pref


def test_por_defecto_solo_toca_capas_mlp(monkeypatch):
    """El default sigue siendo `.mlp.`: convertir el resto no entra en VRAM."""
    monkeypatch.delenv("GENESIS_PN118_ALL_LAYERS", raising=False)
    monkeypatch.delenv("GENESIS_PN118_CAPAS", raising=False)
    p = pytest.importorskip("vllm._genesis.int8_mlp_dispatch")
    assert p._patrones() == (".mlp.",)
    assert p._aplica(Capa("model.layers.0.mlp.gate_up_proj")) is True
    assert p._aplica(Capa("model.layers.0.self_attn.qkv_proj")) is False
    assert p._aplica(Capa("model.layers.0.linear_attn.in_proj_qkv")) is False


def test_el_filtro_de_capas_acepta_varios_patrones(monkeypatch):
    """Marlin W4A16 se lleva el 25,3% del prefill (medido con nsys) y son las
    proyecciones que `.mlp.` saltea. Convertirlas todas no entra, asi que hay
    que poder elegir un subconjunto."""
    monkeypatch.delenv("GENESIS_PN118_ALL_LAYERS", raising=False)
    monkeypatch.setenv("GENESIS_PN118_CAPAS", ".mlp.,self_attn")
    p = pytest.importorskip("vllm._genesis.int8_mlp_dispatch")
    assert p._patrones() == (".mlp.", "self_attn")
    assert p._aplica(Capa("model.layers.0.mlp.gate_up_proj")) is True
    assert p._aplica(Capa("model.layers.0.self_attn.qkv_proj")) is True
    assert p._aplica(Capa("model.layers.0.linear_attn.in_proj_qkv")) is False


def test_all_layers_ignora_el_filtro(monkeypatch):
    monkeypatch.setenv("GENESIS_PN118_ALL_LAYERS", "1")
    monkeypatch.setenv("GENESIS_PN118_CAPAS", ".mlp.")
    p = pytest.importorskip("vllm._genesis.int8_mlp_dispatch")
    assert p._patrones() == ()
    assert p._aplica(Capa("model.layers.0.linear_attn.in_proj_qkv")) is True


def test_el_int4_se_libera_por_defecto(monkeypatch):
    """El punto del parche es que el int4 no ocupe VRAM. Si se conservara, el
    modelo pagaria las dos copias y no ahorraria nada."""
    monkeypatch.delenv("GENESIS_PN118_KEEP_INT4", raising=False)
    p = pytest.importorskip("vllm._genesis.int8_mlp_dispatch")
    assert p._conservar_int4() is False


def test_es_un_text_patch_no_un_monkeypatch():
    """vLLM carga el modelo en procesos WORKER donde apply_all no corre, asi que
    un monkeypatch nunca llega. La primera version de PN118 era monkeypatch: se
    instalaba, el log lo confirmaba, y no convertia una sola capa."""
    p = pytest.importorskip(
        "vllm._genesis.wiring.quantization.patch_PN118_int8_mlp_boot_quant")
    src = __import__("inspect").getsource(p)
    assert "TextPatcher" in src
    # lo definitorio: escribe el archivo en disco, no toca clases en memoria
    assert "resolve_vllm_file" in src
    # lo que define un monkeypatch es ASIGNAR sobre la clase en memoria.
    # Mencionarla en el docstring es legitimo.
    import re
    asigna = re.search(r"^\s*CompressedTensors\w*\.\w+\s*=", src, re.M)
    assert asigna is None, f"asigna sobre la clase: {asigna.group(0)!r}"


def test_el_ancla_toca_los_dos_metodos():
    """Convertir sin cablear el forward no acelera nada; cablear el forward sin
    cortar el camino original explota, porque los tensores AWQ ya se liberaron."""
    p = pytest.importorskip(
        "vllm._genesis.wiring.quantization.patch_PN118_int8_mlp_boot_quant")
    assert "process_weights_after_loading" in p.ANCHOR_OLD
    assert "apply_weights" in p.ANCHOR_OLD
    assert "convertir_capa" in p.ANCHOR_NEW
    assert "forward_int8" in p.ANCHOR_NEW
    # el corte: si convirtio, NO se llama al kernel original
    assert "if _g118.convertir_capa(layer):\n            return\n" in p.ANCHOR_NEW
