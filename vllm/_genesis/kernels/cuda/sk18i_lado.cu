// SPDX-License-Identifier: Apache-2.0
//
// SK-18i/lado — mapea los tokens que se estan escribiendo al ESPEJO int8 de la ventana reciente.
//
// La atencion se concentra en los ultimos tokens (39-49% de la masa en las ultimas 832 posiciones
// medido en las capas 3 y 35), asi que las ultimas PAGS paginas de cada secuencia se guardan
// TAMBIEN en int8 en un pool aparte (17 MB para 10 secuencias) y el decode las lee de ahi.
//
// Ranura del espejo para la pagina p de la secuencia b:  b * PAGS + (p % PAGS)
// con p = posicion_absoluta / BS. dueno[ranura] = bloque fisico que la ocupa; el decode solo usa
// el espejo si dueno coincide con su tabla de bloques (si la secuencia cambio de indice en el
// lote, o si el bloque venia de la cache de prefijos, no coincide y se usa el int4).
//
// Grilla: x = token. Un hilo por token.
extern "C" __global__ void __launch_bounds__(128)
sk18i_lado(
    const long long* __restrict__ slot,     // [n] slot en el pool int4
    const int* __restrict__ qsl,            // [B+1] inicio de cada secuencia en los tokens del paso
    const int* __restrict__ seq,            // [B] largo TOTAL de la secuencia (incluye este paso)
    long long* __restrict__ slot2,          // [n] slot en el pool espejo (-1 = no espejar)
    int* __restrict__ dueno,                // [bmax * PAGS]
    int n, int B, int BS, int PAGS)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const long long sl = slot[t];
    if (sl < 0) { slot2[t] = -1; return; }
    int b = 0;
    while (b + 1 < B && qsl[b + 1] <= t) ++b;              // secuencia del token
    const long long pos = (long long)seq[b] - (long long)(qsl[b + 1] - t);   // posicion absoluta
    if (pos < 0) { slot2[t] = -1; return; }
    const int p = (int)(pos / BS);
    // Solo las ULTIMAS PAGS paginas de la secuencia: si no, paginas viejas caen en la misma
    // ranura (p % PAGS) y se mezclan sus datos dentro del mismo lanzamiento.
    const int ult = (seq[b] - 1) / BS;
    if (p < ult - (PAGS - 1)) { slot2[t] = -1; return; }
    const int ranura = b * PAGS + (p % PAGS);
    const int bloque = (int)(sl / BS);
    slot2[t] = (long long)ranura * BS + (sl % BS);
    dueno[ranura] = bloque;
}
