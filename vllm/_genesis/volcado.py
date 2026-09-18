# SPDX-License-Identifier: Apache-2.0
"""Volcado de diagnostico: cuando algo revienta, dejar por escrito POR QUE.

El problema que resuelve
------------------------
vLLM tira el traceback y se muere. El traceback dice DONDE fallo, pero no con QUE:

    File "vllm/v1/core/block_pool.py", line 277, in cache_full_blocks
      block_hash = new_block_hashes[i]
    IndexError: list index out of range

Eso no alcanza. Lo que hace falta es `i`, `len(new_block_hashes)`, cuantos bloques habia,
de que request, con que block_size. Todo eso esta vivo en el frame que fallo y se pierde.
Sin ello hay que bisecar a ciegas, un arranque por hipotesis, que es lo que costo horas.

Esto engancha dos caminos, porque vLLM usa los dos:

* ``sys.excepthook`` y ``threading.excepthook`` — la excepcion que mata el proceso.
* un handler en el logger raiz — vLLM muchas veces hace ``logger.exception(...)`` y sale
  ordenado, sin excepcion sin atrapar. El EngineCore es justamente asi
  ("EngineCore encountered a fatal error"), y por ese camino un excepthook NO se entera.

Que escribe
-----------
Un archivo por incidente en ``GENESIS_VOLCADO_DIR`` (por defecto
``/root/.cache/vllm/genesis_volcados``, que esta montado, asi que sobrevive a la muerte del
contenedor). Adentro: la excepcion, el traceback completo y, POR CADA FRAME, sus variables
locales resumidas — los tensores como forma/dtype/dispositivo y un par de estadisticas, las
listas como largo mas los primeros y ultimos elementos, los textos recortados. Nunca el
contenido entero: un volcado que no se puede abrir no sirve de nada.

Ademas: las variables GENESIS_*, la memoria de GPU y el rank, que es lo primero que uno
quiere saber cuando el problema aparece en uno solo de los dos workers.

Uso
---
Se instala solo desde ``plugins_arranque`` en todos los procesos. Apagarlo:
``GENESIS_VOLCADO=0``. A mano, desde cualquier lado::

    from vllm._genesis import volcado
    volcado.volcar("bloques-raros", request_id=rid, bloques=len(b), hashes=len(h))
"""

from __future__ import annotations

import datetime
import logging
import os
import sys
import threading

log = logging.getLogger("genesis.volcado")

_instalado = False
_lock = threading.Lock()
_escritos = 0
MAX_VOLCADOS = int(os.environ.get("GENESIS_VOLCADO_MAX", "20"))


def activo() -> bool:
    return os.environ.get("GENESIS_VOLCADO", "1").strip().lower() not in ("0", "false", "no", "off")


def _dir() -> str:
    d = os.environ.get("GENESIS_VOLCADO_DIR", "/root/.cache/vllm/genesis_volcados")
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        d = "/tmp"
    return d


def _rank() -> str:
    """El rank, probando las fuentes en orden de confiabilidad."""
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            return str(dist.get_rank())
    except Exception:                                        # noqa: BLE001
        pass
    for v in ("RANK", "LOCAL_RANK", "VLLM_DP_RANK"):
        if os.environ.get(v):
            return os.environ[v]
    return "?"


def _resumir(v, prof: int = 0) -> str:
    """Un valor en UNA linea corta. Nunca vuelca el contenido entero.

    La regla es que el volcado se tiene que poder leer. Un tensor de 124160x5120 impreso
    entero no es diagnostico, es ruido que tapa el dato que importa.
    """
    try:
        t = type(v).__name__
        if v is None or isinstance(v, (bool, int, float)):
            return repr(v)
        if isinstance(v, str):
            return repr(v[:200]) + (f" …(len={len(v)})" if len(v) > 200 else "")
        if isinstance(v, (bytes, bytearray)):
            return f"{t}(len={len(v)})"
        # tensores / arrays, sin importarlos si no estan
        if hasattr(v, "shape") and hasattr(v, "dtype"):
            extra = ""
            try:
                if getattr(v, "numel", lambda: 1)() and getattr(v, "numel")() <= 2 ** 22:
                    f = v.detach().float() if hasattr(v, "detach") else v
                    extra = " min=%.4g max=%.4g suma=%.6g" % (float(f.min()), float(f.max()),
                                                              float(f.sum()))
            except Exception:                                # noqa: BLE001
                extra = " (sin estadisticas)"
            return (f"{t} forma={tuple(v.shape)} dtype={v.dtype}"
                    f"{' dev=' + str(v.device) if hasattr(v, 'device') else ''}{extra}")
        if isinstance(v, (list, tuple, set)):
            n = len(v)
            if prof >= 2 or n == 0:
                return f"{t}(len={n})"
            xs = list(v)
            cab = ", ".join(_resumir(x, prof + 1) for x in xs[:4])
            col = ", ".join(_resumir(x, prof + 1) for x in xs[-2:]) if n > 6 else ""
            return f"{t}(len={n}) [{cab}{' … ' + col if col else ''}]"
        if isinstance(v, dict):
            n = len(v)
            if prof >= 2 or n == 0:
                return f"dict(len={n})"
            trozos = [f"{k!r}: {_resumir(x, prof + 1)}" for k, x in list(v.items())[:4]]
            return f"dict(len={n}) {{{', '.join(trozos)}{' …' if n > 4 else ''}}}"
        r = repr(v)
        return r[:200] + (" …" if len(r) > 200 else "")
    except Exception as e:                                   # noqa: BLE001
        return f"(no se pudo resumir: {type(e).__name__}: {e})"


def _frames(tb) -> list[str]:
    """Cada frame con sus locals resumidos. Del mas viejo al que fallo."""
    salida = []
    while tb is not None:
        f = tb.tb_frame
        salida.append("  %s:%d  en %s()" % (f.f_code.co_filename, tb.tb_lineno, f.f_code.co_name))
        # Los frames de terceros ensucian; los locals solo interesan cerca del fallo, pero
        # cuales son "cerca" no se sabe de antemano, asi que se vuelcan todos resumidos.
        for k, v in sorted(f.f_locals.items()):
            if k.startswith("__"):
                continue
            salida.append("        %-28s = %s" % (k, _resumir(v)))
        tb = tb.tb_next
    return salida


def _gpu() -> list[str]:
    try:
        import torch

        if not torch.cuda.is_available():
            return ["  (sin CUDA)"]
        i = torch.cuda.current_device()
        libre, total = torch.cuda.mem_get_info(i)
        return ["  gpu=%d  reservado=%.2f GiB  asignado=%.2f GiB  libre=%.2f de %.2f GiB"
                % (i, torch.cuda.memory_reserved(i) / 2 ** 30,
                   torch.cuda.memory_allocated(i) / 2 ** 30, libre / 2 ** 30, total / 2 ** 30)]
    except Exception as e:                                   # noqa: BLE001
        return [f"  (sin datos de GPU: {type(e).__name__})"]


def volcar(motivo: str, exc: BaseException | None = None, **datos) -> str | None:
    """Escribe un volcado. Devuelve la ruta, o None si no se pudo (nunca levanta)."""
    global _escritos
    if not activo():
        return None
    try:
        with _lock:
            if _escritos >= MAX_VOLCADOS:
                return None
            _escritos += 1
        ahora = datetime.datetime.now()
        seguro = "".join(c if c.isalnum() or c in "-_" else "-" for c in motivo)[:40]
        ruta = os.path.join(_dir(), "%s_pid%d_rank%s_%s.txt"
                            % (ahora.strftime("%Y%m%d-%H%M%S"), os.getpid(), _rank(), seguro))
        L = ["=" * 78,
             "VOLCADO GENESIS  %s" % ahora.isoformat(timespec="seconds"),
             "motivo : %s" % motivo,
             "pid    : %d    rank: %s    hilo: %s" % (os.getpid(), _rank(),
                                                      threading.current_thread().name),
             "=" * 78, ""]
        if datos:
            L.append("-- datos --")
            L += ["  %-28s = %s" % (k, _resumir(v)) for k, v in datos.items()]
            L.append("")
        if exc is not None:
            L.append("-- excepcion --")
            L.append("  %s: %s" % (type(exc).__name__, exc))
            L.append("")
            L.append("-- traceback con los locales de cada frame --")
            L += _frames(exc.__traceback__)
            L.append("")
            if exc.__cause__ is not None or exc.__context__ is not None:
                otra = exc.__cause__ or exc.__context__
                L.append("-- causa previa: %s: %s --" % (type(otra).__name__, otra))
                L += _frames(otra.__traceback__)
                L.append("")
        L.append("-- memoria de GPU --")
        L += _gpu()
        L.append("")
        L.append("-- variables GENESIS_* --")
        L += ["  %-40s = %s" % (k, v) for k, v in sorted(os.environ.items())
              if k.startswith(("GENESIS_", "VLLM_"))]
        with open(ruta, "w", encoding="utf-8", errors="replace") as f:
            f.write("\n".join(L) + "\n")
        # A stderr tambien, porque lo primero que uno mira es `docker logs`.
        print("\n[Genesis] VOLCADO escrito en %s  (motivo: %s)\n" % (ruta, motivo),
              file=sys.stderr, flush=True)
        return ruta
    except Exception:                                        # noqa: BLE001
        return None    # un volcado que falla jamas puede empeorar el problema original


class _HandlerVolcado(logging.Handler):
    """Vuelca cuando alguien loguea una excepcion con ``logger.exception`` / ``exc_info``.

    Este es el camino que importa en la practica: el EngineCore de vLLM atrapa el error, lo
    loguea como "EngineCore encountered a fatal error" y sale ordenado. Por ahi un
    ``sys.excepthook`` no se entera NUNCA.
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if record.exc_info and record.exc_info[1] is not None:
                volcar("log-%s" % record.name.replace(".", "-"), exc=record.exc_info[1],
                       mensaje=record.getMessage())
        except Exception:                                    # noqa: BLE001
            pass


def instalar() -> None:
    """Idempotente. Engancha los dos caminos y no molesta si algo falla."""
    global _instalado
    if _instalado or not activo():
        return
    _instalado = True

    anterior = sys.excepthook

    def _hook(tipo, valor, tb):
        volcar("excepcion-%s" % tipo.__name__, exc=valor)
        anterior(tipo, valor, tb)

    sys.excepthook = _hook

    try:
        anterior_h = threading.excepthook

        def _hook_hilo(args):
            volcar("hilo-%s" % args.exc_type.__name__, exc=args.exc_value)
            anterior_h(args)

        threading.excepthook = _hook_hilo
    except Exception:                                        # noqa: BLE001
        pass

    # En el raiz NO alcanza: la config de logging de vLLM (y la nuestra, ver config_logging)
    # pone propagate=False en el logger "vllm", asi que sus registros nunca suben al raiz y el
    # handler no los ve nunca. Se engancha en el raiz Y en los loggers que cortan la
    # propagacion, que son justamente los que loguean los errores que interesan.
    for nombre in ("", "vllm", "genesis"):
        lg = logging.getLogger(nombre)
        if not any(getattr(x, "name", "") == "genesis_volcado" for x in lg.handlers):
            h = _HandlerVolcado(level=logging.ERROR)
            h.set_name("genesis_volcado")
            lg.addHandler(h)

    log.info("[Genesis] volcado de diagnostico activo -> %s (apagar: GENESIS_VOLCADO=0)", _dir())


# ───────────────────────── logs a archivo, por proceso ─────────────────────────

def config_logging(destino: str = "/root/.cache/vllm/genesis_logs") -> str:
    """Escribe un dictConfig para ``VLLM_LOGGING_CONFIG_PATH`` y devuelve su ruta.

    Se usa la palanca NATIVA de vLLM en vez de inventar uuna: la variable
    ``VLLM_LOGGING_CONFIG_PATH`` acepta un dictConfig de logging estandar.

    Que resuelve: con TP=2 son tres procesos (servidor y dos workers) escribiendo al MISMO
    stdout, asi que `docker logs` los entrevera y hay que adivinar de quien es cada linea. Con
    esto cada proceso deja ademas su propio archivo — el nombre lleva el pid — y los loggers
    de Genesis van en DEBUG mientras los de vLLM se quedan en INFO, que es la unica forma de
    que el detalle propio no se pierda entre miles de lineas ajenas.

    Se mantiene la salida a consola: sirve para mirar rapido, y perderla seria un retroceso.
    """
    import json

    os.makedirs(destino, exist_ok=True)
    # Sin el pid en el NOMBRE: el dictConfig lo escribe un proceso (apply_all) y lo leen los
    # tres, asi que un nombre con pid mandaria todo al pid equivocado. El pid va en cada linea
    # (lo pone el formato), que es lo que hace falta para desenredarlas.
    archivo = os.path.join(destino, "vllm.log")
    cfg = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "genesis": {
                "format": "%(asctime)s %(levelname)-7s [pid %(process)d] %(name)s: %(message)s",
                "datefmt": "%H:%M:%S",
            },
        },
        "handlers": {
            "consola": {
                "class": "logging.StreamHandler",
                "formatter": "genesis",
                "stream": "ext://sys.stdout",
                "level": "INFO",
            },
            "archivo": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "genesis",
                "filename": archivo,
                "maxBytes": 64 * 1024 * 1024,
                "backupCount": 3,
                "level": "DEBUG",
            },
        },
        "loggers": {
            "vllm": {"handlers": ["consola", "archivo"], "level": "INFO", "propagate": False},
            "genesis": {"handlers": ["consola", "archivo"], "level": "DEBUG", "propagate": False},
        },
        "root": {"handlers": ["archivo"], "level": "INFO"},
    }
    ruta = os.path.join(destino, "logging.json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    return ruta


if __name__ == "__main__":
    # `python3 -m vllm._genesis.volcado` escribe el dictConfig y dice como usarlo.
    r = config_logging()
    print("dictConfig escrito en:", r)
    print("usarlo:  VLLM_LOGGING_CONFIG_PATH=%s" % r)
    print("volcados en:", _dir())
