# SPDX-License-Identifier: Apache-2.0
"""Instrumentación del laboratorio ox-alpha para trazar el circuito del token.

Se inyecta vía PYTHONPATH como ``sitecustomize`` y corre en TODOS los
procesos Python de vLLM (main, EngineCore, workers spawn). Con ``LAB_TRACE=1``:

1. Envuelve ``execute_model`` de AMBOS runners (V1 ``v1/worker/gpu_model_runner``
   y V2 ``v1/worker/gpu/model_runner`` — v0.23 elige V2 para Qwen3 denso),
   mide tiempos por paso y abre una ventana de ``torch.profiler``
   (pasos ``[LAB_PROF_START, LAB_PROF_END]``) capturando todos los eventos
   CUDA incluidos Memcpy con bytes.
2. Envuelve los puntos exactos de cruce GPU<->CPU:
   - V1 ``_to_list`` / ``AsyncGPUModelRunnerOutput`` (camino viejo)
   - V2 ``sample_tokens`` / ``AsyncOutput.__init__`` / ``get_output``
     (camino actual: D2H async en copy stream + ``.tolist()`` diferido)
   - ``RejectionSampler.parse_output`` (spec-decode)
   - ``CpuGpuBuffer.copy_to_gpu`` (H2D genérico pinned)
3. Log por pid en ``$LAB_RESULTS_DIR/lab_<pid>.log``; traza chrome en
   ``worker_trace_<pid>.json``.

Nunca lanza hacia el proceso host: todo envuelto en try/except.
"""
import os

if os.environ.get("LAB_TRACE") == "1":
    try:
        import functools
        import time

        RESULTS = os.environ.get("LAB_RESULTS_DIR", "/lab/results")
        os.makedirs(RESULTS, exist_ok=True)
        _logf = open(os.path.join(RESULTS, f"lab_{os.getpid()}.log"),
                     "a", buffering=1)

        def log(msg: str) -> None:
            _logf.write(f"[{time.time():.4f}] {msg}\n")

        log(f"=== sitecustomize active pid={os.getpid()} "
            f"prof_window=[{os.environ.get('LAB_PROF_START', '5')},"
            f"{os.environ.get('LAB_PROF_END', '25')}] ===")

        import torch
        from torch.profiler import ProfilerActivity, profile

        PROF_START = int(os.environ.get("LAB_PROF_START", "5"))
        PROF_END = int(os.environ.get("LAB_PROF_END", "25"))
        _state = {"step": 0, "profiler": None}

        def _bytes_of(t) -> int:
            try:
                return t.numel() * t.element_size()
            except Exception:
                return -1

        def _profiler_tick(n: int) -> None:
            """Abre/avanza/cierra la ventana de profiling en el paso n."""
            if n == PROF_START and _state["profiler"] is None:
                p = profile(activities=[ProfilerActivity.CPU,
                                        ProfilerActivity.CUDA],
                            with_stack=False, record_shapes=False)
                p.start()
                _state["profiler"] = p
                log(f"PROFILER START at step {n}")
            elif _state["profiler"] is not None and PROF_START <= n <= PROF_END:
                try:
                    _state["profiler"].step()
                except Exception:
                    pass
            if n == PROF_END and _state["profiler"] is not None:
                p = _state["profiler"]
                _state["profiler"] = None
                p.stop()
                out = os.path.join(RESULTS,
                                   f"worker_trace_{os.getpid()}.json")
                try:
                    p.export_chrome_trace(out)
                    log(f"PROFILER STOP -> exported {out}")
                except Exception as exc:
                    log(f"PROFILER EXPORT FAILED: {exc}")

        def instrument_execute_model(cls, label: str) -> None:
            orig = cls.execute_model

            def execute_model(self, *args, **kwargs):
                _state["step"] += 1
                n = _state["step"]
                t0 = time.perf_counter()
                result = orig(self, *args, **kwargs)
                dt = (time.perf_counter() - t0) * 1000.0
                # LAB_LOG_ALL_STEPS=1 registra el timing de CADA paso sin
                # abrir la ventana del profiler (que sí distorsionaría la
                # medición); la ventana PROF_* sigue funcionando igual.
                if (PROF_START <= n <= PROF_END
                        or os.environ.get("LAB_LOG_ALL_STEPS") == "1"):
                    log(f"STEP {n} {label}.execute_model {dt:.2f} ms")
                    _profiler_tick(n)
                if (os.environ.get("LAB_DUMP_MODEL") == "1"
                        and not getattr(self, "_lab_dumped", False)):
                    self._lab_dumped = True
                    try:
                        mdl = getattr(self, "model", None)
                        log(f"DUMP model class={type(mdl).__name__}")
                        agg: dict[str, int] = {}
                        import re as _re
                        extra = {}
                        for name, p in mdl.named_parameters():
                            parts = name.split(".")
                            k = "/".join(parts[:2])
                            nb = p.numel() * p.element_size()
                            agg[k] = agg.get(k, 0) + nb
                            m = _re.match(r".*layers\.(\d+)\.", name)
                            if m and int(m.group(1)) >= 64:
                                extra[name] = (
                                    f"{p.numel()*p.element_size()/1e6:.2f}MB "
                                    f"{p.dtype}")
                        log("DUMP top-level param groups:")
                        for k, v in sorted(agg.items(),
                                           key=lambda kv: -kv[1])[:18]:
                            log(f"  {v/1e6:9.1f} MB  {k}")
                        log(f"DUMP capas >=64 ({len(extra)}):")
                        for k, v in list(extra.items())[:12]:
                            log(f"   {k}  {v}")
                        log("DUMP attrs mtp/draft en model: "
                            f"{[a for a in dir(mdl) if ('mtp' in a.lower() or 'draft' in a.lower())]}")
                        prop = getattr(self, "drafter", None)
                        log(f"DUMP runner.drafter={type(prop).__name__ if prop else None}")
                        dm = getattr(prop, "model", None) if prop else None
                        if dm is not None:
                            log(f"DUMP drafter.model class={type(dm).__name__}")
                            tot = 0
                            for name, p in dm.named_parameters():
                                nb = p.numel() * p.element_size()
                                tot += nb
                                if nb > 20e6:
                                    log(f"   {nb/1e6:9.1f} MB  {name}  {p.dtype}")
                            log(f"DUMP drafter total={tot/1e9:.2f} GB")
                            lh = getattr(dm, "lm_head", None)
                            if lh is not None:
                                w = getattr(lh, "weight", None)
                                qm = getattr(lh, "quant_method", None)
                                log(f"DRAFT-LMHEAD dtype={w.dtype if w is not None else '-'} "
                                    f"qm={type(qm).__name__ if qm is not None else '-'}")
                    except Exception as exc:
                        log(f"DUMP failed: {exc}")
                return result

            cls.execute_model = execute_model
            log(f"wrapped {label}.execute_model")

        # ------------------------------------------------------------------
        # Runner V1 (clásico): vllm.v1.worker.gpu_model_runner.GPUModelRunner
        # ------------------------------------------------------------------
        try:
            from vllm.v1.worker.gpu_model_runner import (
                GPUModelRunner as GMR_V1,)
            instrument_execute_model(GMR_V1, "V1")

            for meth in ("_update_states", "_prepare_inputs", "_sample"):
                try:
                    orig = getattr(GMR_V1, meth)

                    def _mk(m, fn):
                        @functools.wraps(fn)
                        def w(self, *a, **k):
                            t = time.perf_counter()
                            r = fn(self, *a, **k)
                            d = (time.perf_counter() - t) * 1000.0
                            if PROF_START <= _state["step"] <= PROF_END:
                                log(f"  STEP {_state['step']} {m} {d:.2f} ms")
                            return r
                        return w

                    setattr(GMR_V1, meth, _mk(meth, orig))
                    log(f"wrapped V1.{meth}")
                except AttributeError:
                    log(f"MISSING V1.{meth} (skip)")

            _orig_to_list = GMR_V1._to_list

            def _to_list(self, tensor):
                log(f"D2H_SYNC V1._to_list shape={tuple(tensor.shape)} "
                    f"dtype={tensor.dtype} bytes={_bytes_of(tensor)}")
                return _orig_to_list(self, tensor)

            GMR_V1._to_list = _to_list
            log("wrapped V1._to_list")

            from vllm.v1.worker.gpu_model_runner import (
                AsyncGPUModelRunnerOutput as _AsyncOutV1,)
            _oi1, _og1 = _AsyncOutV1.__init__, _AsyncOutV1.get_output

            def _ai1(self, *a, **k):
                log("D2H_ASYNC V1 __init__ begin")
                r = _oi1(self, *a, **k)
                st = getattr(self, "sampled_token_ids_cpu", None)
                log(f"D2H_ASYNC V1 __init__ done "
                    f"bytes={_bytes_of(st) if st is not None else '?'}")
                return r

            def _go1(self, *a, **k):
                log("D2H_ASYNC V1 get_output: event.synchronize+tolist")
                return _og1(self, *a, **k)

            _AsyncOutV1.__init__, _AsyncOutV1.get_output = _ai1, _go1
            log("wrapped V1.AsyncGPUModelRunnerOutput")
        except Exception as exc:
            log(f"WARN V1 runner wrap failed: {exc}")

        # ------------------------------------------------------------------
        # Runner V2 (default en v0.23 para Qwen3 denso):
        # vllm.v1.worker.gpu.model_runner.GPUModelRunner
        # ------------------------------------------------------------------
        try:
            from vllm.v1.worker.gpu.model_runner import (
                GPUModelRunner as GMR_V2,)
            instrument_execute_model(GMR_V2, "V2")

            try:
                _orig_st = GMR_V2.sample_tokens

                def sample_tokens(self, *a, **k):
                    t = time.perf_counter()
                    r = _orig_st(self, *a, **k)
                    d = (time.perf_counter() - t) * 1000.0
                    if PROF_START <= _state["step"] <= PROF_END:
                        log(f"  STEP {_state['step']} V2.sample_tokens "
                            f"{d:.2f} ms -> {type(r).__name__}")
                    return r

                GMR_V2.sample_tokens = sample_tokens
                log("wrapped V2.sample_tokens")
            except AttributeError:
                log("MISSING V2.sample_tokens (skip)")
        except Exception as exc:
            log(f"WARN V2 runner wrap failed: {exc}")

        # ------------------------------------------------------------------
        # AsyncOutput V2: la D2H real en copy stream (--async-scheduling)
        # ------------------------------------------------------------------
        try:
            from vllm.v1.worker.gpu.async_utils import AsyncOutput

            _oi2, _og2 = AsyncOutput.__init__, AsyncOutput.get_output

            def _ai2(self, *a, **k):
                log("D2H_ASYNC V2 AsyncOutput.__init__ begin "
                    "(copy stream launch)")
                r = _oi2(self, *a, **k)
                st = getattr(self, "sampled_token_ids", None)
                log(f"D2H_ASYNC V2 sampled_token_ids_np shape="
                    f"{getattr(st, 'shape', '?')} "
                    f"dtype={getattr(st, 'dtype', '?')} bytes={_bytes_of(st)}")
                return r

            def _go2(self, *a, **k):
                log("D2H_ASYNC V2 get_output: copy_event.synchronize "
                    "+ tolist")
                return _og2(self, *a, **k)

            AsyncOutput.__init__, AsyncOutput.get_output = _ai2, _go2
            log("wrapped V2.AsyncOutput.__init__/get_output")
        except Exception as exc:
            log(f"WARN V2 AsyncOutput wrap failed: {exc}")

        # ------------------------------------------------------------------
        # Spec-decode: parse_output del RejectionSampler (@staticmethod)
        # ------------------------------------------------------------------
        try:
            from vllm.v1.sample.rejection_sampler import RejectionSampler

            _desc = RejectionSampler.__dict__["parse_output"]
            _underlying = getattr(_desc, "__func__", _desc)

            def _parse_output(*a, **k):
                log("SPEC parse_output: .cpu().numpy() D2H")
                return _underlying(*a, **k)

            RejectionSampler.parse_output = staticmethod(_parse_output)
            log("wrapped RejectionSampler.parse_output")
        except Exception as exc:
            log(f"WARN RejectionSampler wrap failed: {exc}")

        # ------------------------------------------------------------------
        # H2D genérico: CpuGpuBuffer.copy_to_gpu (runner V1)
        # ------------------------------------------------------------------
        try:
            from vllm.v1.utils import CpuGpuBuffer

            _orig_copy = CpuGpuBuffer.copy_to_gpu

            def copy_to_gpu(self, num_elems=-1, *a, **k):
                try:
                    n = num_elems if num_elems >= 0 else self.cpu.shape[0]
                    sl = self.cpu[:n]
                    log(f"H2D copy_to_gpu elems={n} "
                        f"bytes={_bytes_of(sl)} dtype={sl.dtype}")
                except Exception:
                    log(f"H2D copy_to_gpu elems={num_elems}")
                return _orig_copy(self, num_elems, *a, **k)

            CpuGpuBuffer.copy_to_gpu = copy_to_gpu
            log("wrapped CpuGpuBuffer.copy_to_gpu")
        except Exception as exc:
            log(f"WARN CpuGpuBuffer wrap failed: {exc}")

        log("=== instrumentation complete ===")
    except Exception as _fatal:  # nunca romper el proceso host
        try:
            with open("/lab/results/sitecustomize_fatal.log", "a") as f:
                f.write(f"pid={os.getpid()} fatal={_fatal!r}\n")
        except Exception:
            pass

# ─── CK-1.1: activación de PN108 en el lab (el entrypoint de Genesis no
# corre aquí; el rebind debe instalarse en CADA proceso antes del init) ──
if os.environ.get("LAB_ENABLE_PN108") == "1":
    try:
        from vllm._genesis.wiring.spec_decode import (
            patch_PN108_draft_fp8_lm_head as _pn108,)
        _st, _rs = _pn108.apply()
        print(f"[sitecustomize] PN108 {_st}: {_rs}", flush=True)
    except Exception as _e:
        print(f"[sitecustomize] PN108 error: {_e!r}", flush=True)

# ─── CK-1.2a: activación de PN109 en el lab (buffers persistentes de
# metadatos spec-decode; mismo patrón que PN108) ─────────────────────────
if os.environ.get("LAB_ENABLE_PN109") == "1":
    try:
        from vllm._genesis.wiring.loader import (
            patch_PN109_spec_decode_persistent_metadata as _pn109,)
        _st, _rs = _pn109.apply()
        print(f"[sitecustomize] PN109 {_st}: {_rs}", flush=True)
    except Exception as _e:
        print(f"[sitecustomize] PN109 error: {_e!r}", flush=True)

# ─── CK-2.1/2.2: activación de PN110 en el lab (requant INT8 + despacho
# por fase prefill/decode; mismo patrón) ─────────────────────────────────
if os.environ.get("LAB_ENABLE_PN110") == "1":
    try:
        from vllm._genesis.wiring.quantization import (
            patch_PN110_int8_phase_dispatch as _pn110,)
        _st, _rs = _pn110.apply()
        print(f"[sitecustomize] PN110 {_st}: {_rs}", flush=True)
    except Exception as _e:
        print(f"[sitecustomize] PN110 error: {_e!r}", flush=True)
