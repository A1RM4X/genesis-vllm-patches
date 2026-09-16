#!/bin/bash
# Mide el efecto de ACS sobre el ancho de banda P2P entre las dos placas.
#
# POR QUE
# El README del driver P2P (aikitoria/open-gpu-kernel-modules) dice:
#
#   "If P2P transfers are slow, make sure your IOMMU is in passthrough (pt) mode and that ACS
#    is disabled. ACS on root ports forces all GPU-to-GPU traffic through the CPU root complex,
#    killing P2P bandwidth."
#
# En esta maquina `iommu=pt` esta puesto, pero ACS esta ACTIVO en los puertos del switch donde
# cuelgan las dos GPU:
#
#   0e:00.0  ACSCtl: SrcValid+ ReqRedir+ CmpltRedir+ UpstreamFwd+     <- GPU 0 (0f:00.0)
#   0e:10.0  ACSCtl: SrcValid+ ReqRedir+ CmpltRedir+ UpstreamFwd+     <- GPU 1 (11:00.0)
#
# Con ReqRedir+/UpstreamFwd+ el trafico entre las placas sube al root complex en vez de cruzar el
# switch. Medido hoy: 5,5 GB/s con cudaMemcpyPeerAsync contra 11,2 de NCCL, cuando el enlace es
# x8 gen4 (~13 GB/s practicos).
#
# QUE HACE ESTE SCRIPT
# Mide, apaga ACS en esos dos puertos, vuelve a medir, y (salvo que se le pase "dejar") lo
# restaura. El cambio NO es persistente: se deshace solo al reiniciar.
#
# OJO: apagar ACS debilita el aislamiento entre dispositivos PCIe — cualquier dispositivo del
# switch podria leer memoria de otro sin pasar por la IOMMU. En una maquina de un solo dueno y sin
# dispositivos que no sean de confianza es el compromiso normal para tener P2P; decidilo vos.
#
# Uso:  sudo tests/proto/acs_p2p.sh [dejar]
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PUENTES="0e:00.0 0e:10.0"
DEJAR="${1:-no}"

medir() {
  docker run --rm --gpus all --ipc=host --shm-size=2g \
    -v "$REPO/tests/proto:/p:ro" -e HOME=/tmp -e NCCL_DEBUG=WARN \
    --entrypoint torchrun vllm/vllm-openai:v0.27.1 --nproc_per_node=2 /p/p2p_duplex.py 2>&1 \
    | grep -E "unidireccional|a la vez|NCCL all"
}

echo "=== ACS antes:"
for b in $PUENTES; do echo "  $b: $(lspci -vvv -s "$b" | grep ACSCtl | head -1)"; done

echo "=== ancho de banda CON ACS:"
medir

echo "=== apagando ACS..."
declare -A ORIG
for b in $PUENTES; do
  ORIG[$b]=$(setpci -s "$b" ECAP_ACS+6.w)
  setpci -s "$b" ECAP_ACS+6.w=0000 && echo "  $b: ${ORIG[$b]} -> 0000"
done
for b in $PUENTES; do echo "  $b: $(lspci -vvv -s "$b" | grep ACSCtl | head -1)"; done

echo "=== ancho de banda SIN ACS:"
medir

if [ "$DEJAR" = "dejar" ]; then
  echo "=== ACS queda apagado (hasta el proximo reinicio)."
  echo "    Para que sobreviva al reinicio: agregar pcie_acs_override=downstream,multifunction"
  echo "    a GRUB_CMDLINE_LINUX_DEFAULT en /etc/default/grub (si el kernel lo soporta), o"
  echo "    deshabilitar ACS en la BIOS."
else
  echo "=== restaurando ACS:"
  for b in $PUENTES; do
    setpci -s "$b" ECAP_ACS+6.w="${ORIG[$b]}" && echo "  $b: -> ${ORIG[$b]}"
  done
fi
