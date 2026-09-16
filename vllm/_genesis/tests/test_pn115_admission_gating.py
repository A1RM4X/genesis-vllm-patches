# SPDX-License-Identifier: Apache-2.0
"""TDD para PN115 — admisión por headroom de KV + bypass por prioridad.

Cubre las cuatro regresiones que tenía la v1 del parche:

1. **Corrupción del paso al preemptar.** El bypass sacaba la víctima de
   `self.running` y llamaba `_preempt_request`, pero NO deshacía el estado
   que esa víctima ya tenía agendado en el mismo paso
   (`num_scheduled_tokens`, `req_to_new_blocks`,
   `scheduled_spec_decode_tokens`, `scheduled_encoder_inputs`,
   `scheduled_running_reqs`). El `SchedulerOutput` seguía mandando al model
   runner un request cuyos bloques KV se acababan de liberar.

2. **Inanición.** El gate proyectaba `committed + prompt_entero` contra una
   capacidad estimada en tokens. Con `--max-model-len 262144` un solo prompt
   largo se pasaba del umbral, se re-encolaba cada paso y no entraba nunca.

3. **Doble conteo.** `committed_kv_tokens` sumaba `num_output_tokens` además
   de `num_tokens`, que en vLLM ya incluye lo generado.

4. **Estrangulamiento del PID.** La línea base se calibraba una sola vez con
   el motor casi vacío, así que a concurrencia alta el costo normal del
   batching se leía como congestión y el límite caía a `min_concurrency`.
"""
from __future__ import annotations

import ast
import importlib
import os
import tempfile

import pytest
def _aislar_archivos(mod):
    """Cada test con su propio par de archivos.

    `get_pid_status()` PREFIERE el archivo de estado al estado vivo (asi lo
    publica el sidecar), asi que si dos tests comparten la ruta el segundo lee
    el snapshot del primero. Antes apuntaban todos a ``os.devnull + ".pn115"``,
    que no es /dev/null sino un archivo comun de verdad.
    """
    d = tempfile.mkdtemp(prefix="pn115-")
    mod.PID_STATUS_FILE = os.path.join(d, "estado.json")
    mod.PID_CONTROL_FILE = os.path.join(d, "control.json")




@pytest.fixture()
def gating(monkeypatch):
    """Módulo recargado con el gating encendido y el PID de latencia apagado."""
    monkeypatch.setenv("GENESIS_ENABLE_PN115_PID_GATING", "1")
    monkeypatch.setenv("GENESIS_PN115_KV_GATING", "1")
    monkeypatch.setenv("GENESIS_PN115_LATENCY_PID", "0")
    monkeypatch.setenv("GENESIS_PID_MIN_CONCURRENCY", "2")
    monkeypatch.setenv("GENESIS_PID_HEADROOM_RATIO", "0.10")
    monkeypatch.setenv("GENESIS_PID_HIGH_PRIO_THRESHOLD", "0")
    mod = importlib.import_module("vllm._genesis.dynamic_pid_gating")
    mod = importlib.reload(mod)
    # /dev/shm puede no existir en el runner; el status es best-effort.
    _aislar_archivos(mod)
    yield mod
    importlib.reload(mod)


# ───────────────────────────── dobles de prueba ─────────────────────────────


class FakeRequest:
    def __init__(self, rid, *, priority=0, num_tokens=100, num_computed=100,
                 num_output=0, arrival=0.0, status="RUNNING",
                 num_prompt_tokens=None, kv_transfer_params=None):
        self.request_id = rid
        self.priority = priority
        self.num_tokens = num_tokens
        self.num_computed_tokens = num_computed
        self.num_output_tokens = num_output
        self.arrival_time = arrival
        self.status = status
        # num_tokens = num_prompt_tokens + generados. Un request en decode tiene
        # num_computed < num_tokens y NO esta en prefill.
        self.num_prompt_tokens = (num_prompt_tokens if num_prompt_tokens is not None
                                  else num_tokens - num_output)
        self.kv_transfer_params = kv_transfer_params

    def get_num_encoder_embeds(self, i):
        return 10


class FakeQueue:
    def __init__(self, requests):
        self._q = list(requests)

    def peek_request(self):
        return self._q[0] if self._q else None

    def pop_request(self):
        return self._q.pop(0)

    def __bool__(self):
        return bool(self._q)


class FakeBlockPool:
    def __init__(self, free, total):
        self._free = free
        self.num_gpu_blocks = total

    def get_num_free_blocks(self):
        return self._free


class FakeKVManager:
    def __init__(self, free, total, watermark=0):
        self.block_pool = FakeBlockPool(free, total)
        self.watermark_blocks = watermark
        self.usage = 1.0 - (free / total)


class FakeCacheConfig:
    block_size = 16
    num_gpu_blocks = 1000


class FakeSchedulerConfig:
    max_num_seqs = 10
    max_num_batched_tokens = 4096
    async_scheduling = False


class FakeScheduler:
    def __init__(self, running, *, free_blocks=800, total_blocks=1000, waiting=()):
        self.running = list(running)
        self.waiting = FakeQueue(waiting)
        self.num_waiting_for_streaming_input = 0
        self.max_num_running_reqs = 10
        self.max_num_scheduled_tokens = 4096
        self.cache_config = FakeCacheConfig()
        self.scheduler_config = FakeSchedulerConfig()
        self.kv_cache_manager = FakeKVManager(free_blocks, total_blocks)
        self.preempted = []

    def _select_waiting_queue_for_scheduling(self):
        return self.waiting or None

    def _preempt_request(self, request, timestamp, drop_stale_output=False):
        assert request.status == "RUNNING", "solo se puede preemptar un RUNNING"
        request.status = "PREEMPTED"
        request.num_computed_tokens = 0
        self.preempted.append(request)


# ══════════════ 1. rollback completo al preemptar por prioridad ══════════════


def test_priority_preempt_deshace_el_estado_agendado_de_la_victima(gating):
    victim = FakeRequest("victima", priority=10, num_tokens=500, num_computed=500)
    keeper = FakeRequest("queda", priority=0)
    incoming = FakeRequest("vip", priority=-8, num_tokens=50, num_computed=0)

    sched = FakeScheduler([victim, keeper], waiting=[incoming])
    sched.max_num_running_reqs = 2

    # Estado que el bucle de running ya agendó en ESTE paso.
    preempted_reqs: list = []
    scheduled_running_reqs = [victim, keeper]
    num_scheduled_tokens = {"victima": 4, "queda": 4}
    req_to_new_blocks = {"victima": object(), "queda": object()}
    scheduled_spec_decode_tokens = {"victima": [1, 2, 3], "queda": [4, 5, 6]}
    scheduled_encoder_inputs = {"victima": [0, 1]}

    tok, enc, idx = gating.try_priority_preempt(
        scheduler=sched,
        timestamp=123.0,
        preempted_reqs=preempted_reqs,
        scheduled_running_reqs=scheduled_running_reqs,
        num_scheduled_tokens=num_scheduled_tokens,
        req_to_new_blocks=req_to_new_blocks,
        scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
        scheduled_encoder_inputs=scheduled_encoder_inputs,
    )

    assert sched.preempted == [victim]
    assert preempted_reqs == [victim]
    assert victim not in sched.running

    # El corazón del bug: la víctima no puede seguir en el SchedulerOutput.
    assert "victima" not in num_scheduled_tokens
    assert "victima" not in req_to_new_blocks
    assert "victima" not in scheduled_spec_decode_tokens
    assert "victima" not in scheduled_encoder_inputs
    assert victim not in scheduled_running_reqs

    # Y el que se queda no puede haber sido tocado.
    assert num_scheduled_tokens == {"queda": 4}
    assert scheduled_spec_decode_tokens == {"queda": [4, 5, 6]}

    # Deltas para reponer los contadores locales de schedule().
    assert (tok, enc, idx) == (4, 20, -1)


def test_sin_head_de_prioridad_alta_no_preempta_nada(gating):
    victim = FakeRequest("victima", priority=10)
    normal = FakeRequest("normal", priority=0, num_computed=0)
    sched = FakeScheduler([victim], waiting=[normal])

    num_scheduled_tokens = {"victima": 4}
    deltas = gating.try_priority_preempt(
        scheduler=sched, timestamp=1.0, preempted_reqs=[],
        scheduled_running_reqs=[victim], num_scheduled_tokens=num_scheduled_tokens,
        req_to_new_blocks={"victima": object()}, scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
    )
    assert deltas == (0, 0, 0)
    assert sched.preempted == []
    assert num_scheduled_tokens == {"victima": 4}


def test_no_se_elige_victima_que_no_este_RUNNING(gating):
    # _preempt_request asertea status == RUNNING; elegir otra cosa lo revienta.
    no_running = FakeRequest("a", priority=10, status="PREEMPTED")
    incoming = FakeRequest("vip", priority=-8)
    assert gating.select_preemption_victim([no_running], incoming) is None


def test_no_se_elige_victima_de_prioridad_igual_o_mayor(gating):
    par = FakeRequest("par", priority=-8)
    incoming = FakeRequest("vip", priority=-8)
    assert gating.select_preemption_victim([par], incoming) is None


# ════════════════════ 2. invariantes anti-deadlock del gate ════════════════════


def test_nunca_difiere_por_debajo_de_min_concurrency(gating):
    """Sin este invariante, un request que no entra en el headroom queda en la
    cola para siempre: no hay nadie corriendo que pueda liberar bloques."""
    enorme = FakeRequest("enorme", num_tokens=262144, num_computed=0)
    # Motor casi lleno y un solo request corriendo (min_concurrency=2).
    sched = FakeScheduler([FakeRequest("solo")], free_blocks=101, total_blocks=1000)
    assert gating.should_gate_waiting(enorme, sched) is False


def test_prompt_gigante_no_se_mata_de_hambre_con_chunked_prefill(gating):
    """Con chunked prefill el request reserva el chunk del paso, no el prompt
    entero: proyectar el prompt completo lo dejaba fuera para siempre."""
    enorme = FakeRequest("enorme", num_tokens=262144, num_computed=0)
    sched = FakeScheduler(
        [FakeRequest(f"r{i}") for i in range(4)], free_blocks=800, total_blocks=1000
    )
    # 4096 tokens / 16 = 256 bloques pedidos; 800 - 256 = 544 > 100 de headroom.
    assert gating.should_gate_waiting(enorme, sched) is False


def test_difiere_cuando_no_queda_headroom_fisico(gating):
    req = FakeRequest("nuevo", num_tokens=1600, num_computed=0)
    sched = FakeScheduler(
        [FakeRequest(f"r{i}") for i in range(4)], free_blocks=150, total_blocks=1000
    )
    # 1600/16 = 100 bloques; 150 - 100 = 50 < 100 de headroom -> difiere.
    assert gating.should_gate_waiting(req, sched) is True
    assert gating.get_pid_status()["gated_requests_total"] >= 1


def test_el_bypass_de_prioridad_nunca_se_difiere(gating):
    vip = FakeRequest("vip", priority=-8, num_tokens=1600, num_computed=0)
    sched = FakeScheduler(
        [FakeRequest(f"r{i}") for i in range(4)], free_blocks=100, total_blocks=1000
    )
    assert gating.should_gate_waiting(vip, sched) is False


def test_apagado_no_difiere_nada(gating):
    gating._CONTROLLER.enabled = False
    req = FakeRequest("nuevo", num_tokens=1600, num_computed=0)
    sched = FakeScheduler([FakeRequest(f"r{i}") for i in range(4)], free_blocks=0,
                          total_blocks=1000)
    assert gating.should_gate_waiting(req, sched) is False


# ═══════════════════ 3. committed_kv_tokens sin doble conteo ═══════════════════


def test_committed_kv_tokens_no_cuenta_dos_veces_el_output(gating):
    # num_tokens ya incluye los 300 generados.
    r = FakeRequest("r", num_tokens=1000, num_computed=1000, num_output=300)
    sched = FakeScheduler([r])
    gating.update_step_with_scheduler(sched, 20.0)
    assert gating._CONTROLLER.committed_kv_tokens == 1000


# ════════════════ 4. el PID compara contra su propia concurrencia ════════════════


def test_el_pid_no_estrangula_por_el_costo_normal_del_batching(monkeypatch):
    monkeypatch.setenv("GENESIS_ENABLE_PN115_PID_GATING", "1")
    monkeypatch.setenv("GENESIS_PN115_LATENCY_PID", "1")
    monkeypatch.setenv("GENESIS_PID_MIN_CONCURRENCY", "2")
    mod = importlib.reload(importlib.import_module("vllm._genesis.dynamic_pid_gating"))
    _aislar_archivos(mod)
    try:
        ctrl = mod._CONTROLLER

        # Fase 1: el motor arranca con 1 secuencia a 20 ms/paso.
        s1 = FakeScheduler([FakeRequest("a")])
        for _ in range(20):
            mod.update_step_with_scheduler(s1, 20.0)

        limite_inicial = ctrl.current_concurrency_limit

        # Fase 2: 10 secuencias a 45 ms/paso. Es el costo normal del batching,
        # no congestión. La v1 comparaba contra la base de 20 ms y colapsaba.
        s10 = FakeScheduler([FakeRequest(f"r{i}") for i in range(10)])
        for _ in range(40):
            mod.update_step_with_scheduler(s10, 45.0)

        assert ctrl.current_concurrency_limit == limite_inicial, (
            "el PID bajó la concurrencia por el costo normal del batching"
        )
        assert ctrl._baselines[10][0] == pytest.approx(45.0, rel=0.05)

        # Fase 3: congestión real a la MISMA concurrencia (45 -> 90 ms).
        for _ in range(30):
            mod.update_step_with_scheduler(s10, 90.0)
        assert ctrl.current_concurrency_limit < limite_inicial, (
            "el PID no reaccionó a congestión real"
        )
        assert ctrl.current_concurrency_limit >= ctrl.min_concurrency
    finally:
        importlib.reload(mod)


def test_el_pid_se_autodesactiva_con_async_scheduling(monkeypatch):
    monkeypatch.setenv("GENESIS_ENABLE_PN115_PID_GATING", "1")
    monkeypatch.setenv("GENESIS_PN115_LATENCY_PID", "1")
    mod = importlib.reload(importlib.import_module("vllm._genesis.dynamic_pid_gating"))
    _aislar_archivos(mod)
    try:
        sched = FakeScheduler([FakeRequest("a")])
        sched.scheduler_config.async_scheduling = True
        mod.update_step_with_scheduler(sched, 20.0)
        assert mod._CONTROLLER.latency_pid_enabled is False
        assert "async_scheduling" in mod._CONTROLLER.latency_pid_forced_off_reason
    finally:
        FakeSchedulerConfig.async_scheduling = False
        importlib.reload(mod)


# ══════════════════════════ prioridad efectiva ══════════════════════════


def test_prioridad_explicita_gana_sobre_el_mapa_de_agentes(gating):
    assert gating.resolve_priority(FakeRequest("explorer-1", priority=-10)) == -10


def test_el_mapa_de_agentes_atraviesa_el_prefijo_de_openai(gating):
    r = FakeRequest("chatcmpl-coach-abc123")
    assert gating.resolve_priority(r) == gating.AGENT_PRIORITY_MAP["coach"]


def test_id_autogenerado_no_matchea_ningun_agente(gating):
    assert gating.resolve_priority(FakeRequest("chatcmpl-7f3a9b21")) == 0


# ═════════════════ el texto inyectado en scheduler.py es sano ═════════════════


def test_el_texto_inyectado_preserva_el_contrato_de_upstream():
    from vllm._genesis.wiring.perf_hotfix import patch_N115_dynamic_pid_admission as p

    nuevo = p.ADMISSION_GATE_NEW
    # El assert de upstream vuelve (la v1 lo había degradado a un break).
    assert "assert request_queue is not None" in nuevo
    # El import se iza fuera del while: el bucle de waiting es camino caliente.
    assert nuevo.index("import dynamic_pid_gating") < nuevo.index("while (self.waiting")
    assert nuevo.count("import dynamic_pid_gating") == 1
    # Un paso que preempta no admite trabajo nuevo.
    assert "try_priority_preempt" in nuevo and "break" in nuevo
    # Y compila dentro de un cuerpo de función.
    ast.parse("def f():\n    while True:\n" + "\n".join(
        "    " + ln for ln in nuevo.splitlines()[2:]
    ))


# ══════════════ 5. serializacion de prefills ══════════════


@pytest.fixture()
def serial(monkeypatch):
    """Gating con un solo prefill concurrente."""
    monkeypatch.setenv("GENESIS_ENABLE_PN115_PID_GATING", "1")
    monkeypatch.setenv("GENESIS_PN115_KV_GATING", "1")
    monkeypatch.setenv("GENESIS_PN115_LATENCY_PID", "0")
    monkeypatch.setenv("GENESIS_PN115_MAX_CONCURRENT_PREFILLS", "1")
    mod = importlib.reload(importlib.import_module("vllm._genesis.dynamic_pid_gating"))
    _aislar_archivos(mod)
    yield mod
    importlib.reload(mod)


def _prefilling(rid):
    """Request computando su prompt: 40k de prompt, 10k computados."""
    return FakeRequest(rid, num_tokens=40_000, num_computed=10_000,
                       num_prompt_tokens=40_000)


def _decoding(rid):
    """Request en decode: prompt entero computado, generando.

    num_computed(1000) < num_tokens(1050) pero NO esta en prefill. Este es el
    caso que rompia el predicado viejo.
    """
    return FakeRequest(rid, num_tokens=1050, num_computed=1000,
                       num_prompt_tokens=1000, num_output=50)


def test_decode_no_cuenta_como_prefill(serial):
    """num_computed < num_tokens es cierto para TODO request en decode."""
    assert serial._is_prefilling(_decoding("d")) is False
    assert serial._is_prefilling(_prefilling("p")) is True


def test_se_difiere_mientras_hay_un_prefill_en_vuelo(serial):
    sched = FakeScheduler([_prefilling("grande"), _decoding("d1")])
    nuevo = FakeRequest("nuevo", num_tokens=40_000, num_computed=0,
                        num_prompt_tokens=40_000)
    assert serial.should_gate_waiting(nuevo, sched) is True
    assert serial.get_pid_status()["gated_by_prefill_total"] >= 1


def test_no_se_difiere_si_nadie_esta_prefilleando(serial):
    sched = FakeScheduler([_decoding("d1"), _decoding("d2")])
    nuevo = FakeRequest("nuevo", num_tokens=40_000, num_computed=0,
                        num_prompt_tokens=40_000)
    assert serial.should_gate_waiting(nuevo, sched) is False


def test_el_motor_vacio_nunca_difiere(serial):
    """Invariante absoluto: sin nadie corriendo no hay quien libere nada."""
    sched = FakeScheduler([], free_blocks=1, total_blocks=1000)
    nuevo = FakeRequest("nuevo", num_tokens=200_000, num_computed=0,
                        num_prompt_tokens=200_000)
    assert serial.should_gate_waiting(nuevo, sched) is False


def test_prioridad_alta_no_espera_detras_de_un_prefill(serial):
    sched = FakeScheduler([_prefilling("grande")])
    vip = FakeRequest("vip", priority=-8, num_tokens=40_000, num_computed=0,
                      num_prompt_tokens=40_000)
    assert serial.should_gate_waiting(vip, sched) is False


def test_apagado_por_default_no_serializa(gating):
    """Sin GENESIS_PN115_MAX_CONCURRENT_PREFILLS el comportamiento no cambia."""
    assert gating._CONTROLLER.max_concurrent_prefills == 0
    sched = FakeScheduler([_prefilling("grande"), _decoding("d")])
    nuevo = FakeRequest("nuevo", num_tokens=40_000, num_computed=0,
                        num_prompt_tokens=40_000)
    assert gating.should_gate_waiting(nuevo, sched) is False


# ══════════════ prioridad desde kv_transfer_params ══════════════


def test_prioridad_desde_kv_transfer_params_agente(gating):
    r = FakeRequest("chatcmpl-7f3a", kv_transfer_params={"genesis_agent": "coach"})
    assert gating.resolve_priority(r) == gating.AGENT_PRIORITY_MAP["coach"]


def test_prioridad_explicita_en_kv_transfer_params(gating):
    r = FakeRequest("chatcmpl-7f3a", kv_transfer_params={"priority": -42})
    assert gating.resolve_priority(r) == -42


def test_kv_transfer_params_con_agente_desconocido_no_rompe(gating):
    r = FakeRequest("chatcmpl-7f3a", kv_transfer_params={"genesis_agent": "nadie"})
    assert gating.resolve_priority(r) == 0
