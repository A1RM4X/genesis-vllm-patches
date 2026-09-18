#!/bin/bash
# Diff de dos logs de arranque, normalizando lo que cambia siempre (hora, pid, direcciones,
# tamanios de memoria libre). Lo que quede es diferencia ESTRUCTURAL: una advertencia que
# aparece en uno y no en el otro, o un orden distinto.
# uso: difflog.sh bueno.log malo.log
norm() {
  sed -E \
    -e 's/[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}//g' \
    -e 's/pid=[0-9]+/pid=N/g' \
    -e 's/0x[0-9a-f]+/0xADDR/g' \
    -e 's/[0-9]+\.[0-9]+ (GiB|MiB|GB|MB|s|ms)/N \1/g' \
    -e 's/\b[0-9]{3,}\b/N/g' \
    -e 's/\[[0-9]+\/[0-9]+\]//g' \
    -e 's/[0-9]+%\|[^|]*\|//g' \
    "$1" | grep -vE "it/s|Loading safetensors|Capturing|Compiling a graph|torch.compile|^\s*$" | sort -u
}
diff <(norm "$1") <(norm "$2")
