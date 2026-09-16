// SPDX-License-Identifier: Apache-2.0
//
// SK-20/norm_quant — residual + RMSNorm + cuantizacion int8 por token, en UN kernel y sin FPU.
//
// Hoy son tres kernels por sitio (165 + 256 + 208 lanzamientos por paso de decode):
//     h, r = fused_add_rms_norm(h, r, w)     521 us   (lee h, r, w; escribe h, r)
//     h_q, s = per_token_quant_int8(h)       560 us   (lee h; escribe h_q, s)
//     s = s * factor_global                  272 us   (lee y escribe M floats)
//
// La observacion que lo hace barato: **la normalizacion se cancela en los enteros**. Con
//     y_i = r_i * inv_rms * w_i        y        q_i = round(y_i * 127 / max|y|)
// el inv_rms aparece arriba y abajo, asi que
//     q_i = round(r_i * w_i * 127 / max|r*w|)
// y el inv_rms solo hace falta UNA vez por fila, para la escala fp32 de salida: no hay que
// normalizar 5120 valores, alcanza con normalizar un escalar.
//
// Trucos enteros (ninguna instruccion de punto flotante):
//  * r = x + residual: magnitudes en Q32 de 64 bits (exacto) y el fp16 se arma con clz. El
//    redondeo es al par, como IEEE: con truncado o medio-arriba el 10% de los valores se iba
//    un ulp respecto de torch (la suma de dos fp16 cae en empate exacto muy seguido).
//  * maximo de |r*w| sin multiplicar en punto flotante: con mantisas de 11 bits el producto
//    m_r*m_w vive en [2^20, 2^22); NORMALIZADO a [2^20, 2^21) la clave
//        (e_r + e_w) << 21 | (m_r*m_w)
//    es monotona en |r*w| (sin normalizar no lo es: el producto abarca un factor 4 y el
//    exponente deja de mandar) y el maximo de la fila sale con un max entero.
//  * cuantizar: q = (p_i * rec) >> (DESP + e_max - e_i), con rec calculado UNA vez por fila.
//  * suma de cuadrados con exponente compartido de la fila, mantisas REDONDEADAS a 16 bits
//    (truncarlas sesga la suma hacia abajo y eso solo daba 0,3% de error en la escala) y raiz
//    cuadrada entera por Newton.
//
// La segunda pasada relee ``res_out`` (recien escrito, esta en L2) en vez de guardar la fila en
// registros: con 20 elementos por hilo el cache costaba 96 registros y el kernel derramaba.
//
// Grilla: x = fila (token). Un bloque por fila.
#include "sk18h_comun.cuh"

#ifndef HILOS
#define HILOS 256
#endif
// q = (p * rec) >> (DESP + e_max - e). Con p y p_max normalizados a [2^20, 2^21) el cociente
// p/p_max <= 2, asi que p*rec <= 127*2^(DESP+1): DESP = 23 deja el producto justo abajo de 2^31.
#define DESP 23
// bits extra de la mantisa al acumular cuadrados (m << SQB cabe en 16 bits, el cuadrado en 32)
#define SQB 5

// raiz cuadrada de un entero de 32 bits normalizado a [2^30, 2^32), devuelta en 16 bits.
//
// La version digito a digito son 16 iteraciones DEPENDIENTES y la corre un solo hilo con el
// bloque esperando: era 1 us de cola. Aca se parte el rango en [1,2) y [2,4) y se ajusta una
// cuadratica en Q14 (error < 1e-4), con un paso de Newton entero encima para llegar a ~1e-7.
// Newton usa una division de 32 bits, una sola por fila.
__device__ __forceinline__ unsigned raiz16(unsigned v) {
    if (v == 0u) return 0u;
    const unsigned u = v >> 16;                       // Q14 en [2^14, 2^16)
    const int alto = u >= (1u << 15);                 // [2,4) o [1,2)
    const unsigned t = alto ? (u >> 1) : u;           // Q14 en [1,2)
    // sqrt(t) ~ A + B*t - C*t^2 ajustada en [1,2) (Q14)
    const unsigned t2 = (t * t) >> 14;
    unsigned r = 7286u + ((10254u * t) >> 14) - ((1156u * t2) >> 14);   // Q14 ~ sqrt(t)
    if (alto) r = (r * 23170u) >> 14;                 // * sqrt(2) en Q14
    r <<= 1;                                          // sqrt(v) ~ r, en 16 bits
    const unsigned n = r ? (v / r + r) >> 1 : 0u;     // un paso de Newton entero
    return n;
}

// mantisa de 11 bits y exponente de un fp16 (normal o subnormal): |v| = m * 2^(e - 25)
__device__ __forceinline__ unsigned mant16(unsigned b, int* e) {
    const int ee = (int)((b >> 10) & 31u);
    *e = ee ? ee : 1;
    return (b & 1023u) | (ee ? 1024u : 0u);
}

// SUMA fp16 + fp16 -> fp16, exacta y redondeada al par, TODA en enteros de 32 bits.
//
// El camino obvio (magnitudes en Q32 de 64 bits) sale carisimo: el rango del fp16 son 40 bits de
// exponente mas 11 de mantisa, no entra en 32 y cada elemento se lleva ~80 instrucciones de 64
// bits. Este es el sumador clasico: alinear al exponente mayor, guardar un bit "pegajoso" con
// lo que se cae del corrimiento, sumar/restar en 25 bits y renormalizar con clz de 32.
//
// El pegajoso vale para el redondeo exacto: con resto == mitad hay empate SOLO si no quedo nada
// abajo. Y en la resta, A - B - eps = (A - B - 1) + (1 - eps), asi que restar uno y prender el
// pegajoso deja la parte fraccionaria otra vez en (0,1), que es lo que el redondeo necesita.
#ifndef ENTSUMA
// EXCEPCION medida a la regla de "todo entero", acotada a ESTA suma: el sumador fp16 del
// hardware da EXACTAMENTE los mismos bits que el sumador entero de abajo (verificado elemento a
// elemento a M=5/40/8192), porque aquel reproduce el redondeo IEEE al par. O sea que aca el
// entero no compra ni precision ni determinismo: solo cuesta ~40 instrucciones por elemento
// contra 1 (y add.f16x2 hace DOS elementos de una), el 44% del kernel.
// Con -DENTSUMA=1 se compila el camino entero, que queda como referencia y para verificar.
__device__ __forceinline__ unsigned suma_fp16x2(unsigned a, unsigned b) {
    unsigned r;
    asm("add.f16x2 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(b));
    return r;
}
#define HAY_SUMA2 1
#else
__device__ __forceinline__ unsigned suma_fp16(unsigned a, unsigned b) {
    // ordenar por magnitud es gratis: el patron de bits de |fp16| ya esta ordenado
    const unsigned au = a & 0x7fffu, bu = b & 0x7fffu;
    const unsigned hi = au > bu ? au : bu, lo = au > bu ? bu : au;
    const unsigned sr = (au > bu ? a : b) >> 15;          // el signo lo pone el mayor
    const int eh = (int)(hi >> 10), el = (int)(lo >> 10);
    const unsigned mh = (hi & 1023u) | (eh ? 1024u : 0u);
    const unsigned ml = (lo & 1023u) | (el ? 1024u : 0u);
    int d = (eh ? eh : 1) - (el ? el : 1);
    d = d > 25 ? 25 : d;                                  // con 25 ya cae todo al pegajoso
    const unsigned L = ml << 13;
    const unsigned B = L >> d;
    const unsigned peg = (L & ((1u << d) - 1u)) != 0u;
    const int resta = (int)((a ^ b) >> 15);               // signos distintos -> resta
    int S = (int)(mh << 13) + (resta ? -(int)(B + peg) : (int)B);
    if (S == 0) return 0u;                                // cancelacion exacta -> +0
    const int p = 31 - __clz(S);
    int e = p + (eh ? eh : 1) - 23;                       // valor = S * 2^(eh - 38)
    int sh = p - 10;
    if (e <= 0) { sh += 1 - e; e = 1; }                   // subnormal: mantisa corrida
    unsigned mant;
    if (sh > 0) {
        if (sh > 31) return sr << 15;
        const unsigned resto = (unsigned)S & ((1u << sh) - 1u);
        const unsigned mitad = 1u << (sh - 1);
        mant = (unsigned)S >> sh;
        mant += (resto > mitad) | ((resto == mitad) & (peg | (mant & 1u)));
    } else {
        mant = (unsigned)S << (-sh);
    }
    if (mant >= 2048u) { mant >>= 1; ++e; }
    unsigned r;
    if (e >= 31) r = (30u << 10) | 1023u;
    else if (mant < 1024u) r = mant;                      // quedo subnormal (exponente 0)
    else r = ((unsigned)e << 10) | (mant & 1023u);
    return r | (sr << 15);
}

#endif

// clave monotona de |r*w|: producto normalizado a [2^20, 2^21) y exponente arriba
__device__ __forceinline__ unsigned clave_rw(unsigned rb, unsigned wb) {
    int er, ew;
    const unsigned mr = mant16(rb, &er), mw = mant16(wb, &ew);
    unsigned p = mr * mw;
    int E = er + ew;
    if (p >= (1u << 21)) { p >>= 1; ++E; }
    return ((unsigned)E << 21) | p;
}

// acumulador de cuadrados con exponente corrido: guarda sum (|r| * 2^(SQB+25-eref))^2 sin
// conocer de antemano el maximo de la fila. Cuando aparece un exponente mayor, baja lo ya
// sumado. Asi los cuadrados salen en la pasada A, donde la mantisa de r ya esta abierta.
__device__ __forceinline__ void acum_cuad(unsigned mr, int er, unsigned long long* acc, int* eref) {
    if (er > *eref) {
        const int k = 2 * (er - *eref);
        *acc = k >= 64 ? 0ull : (*acc >> k);
        *eref = er;
    }
    const int d = *eref - er;
    const unsigned mn = d >= 16 + SQB ? 0u : (((mr << SQB) + (1u << d >> 1)) >> d);
    *acc += (unsigned long long)mn * mn;
}

// une dos acumuladores con exponentes distintos
__device__ __forceinline__ void unir_cuad(unsigned long long a, int ea, unsigned long long* acc,
                                          int* eref) {
    if (ea > *eref) {
        const int k = 2 * (ea - *eref);
        *acc = k >= 64 ? 0ull : (*acc >> k);
        *eref = ea;
    } else {
        const int k = 2 * (*eref - ea);
        a = k >= 64 ? 0ull : (a >> k);
    }
    *acc += a;
}

// una posicion de la pasada A: r = x + residual, clave del maximo |r*w|, cuadrado acumulado.
// Devuelve, ademas de r, la clave del elemento (con su signo) para que la pasada B no tenga que
// volver a abrir mantisas ni a multiplicar: se la deja en shared.
__device__ __forceinline__ void paso_a(unsigned rb, unsigned wb,
                                       unsigned* clave, unsigned* cel,
                                       unsigned long long* acc, int* eref) {
    int er, ew;
    const unsigned mr = mant16(rb, &er), mw = mant16(wb, &ew);
    acum_cuad(mr, er, acc, eref);
    unsigned p = mr * mw;
    int E = er + ew;
    if (p >= (1u << 21)) { p >>= 1; ++E; }
    const unsigned c = ((unsigned)E << 21) | p;             // monotona en |r*w|
    *clave = *clave > c ? *clave : c;
    *cel = c | (((rb ^ wb) & 0x8000u) << 16);               // el signo, en el bit 31
}

// suma de dos fp16 empaquetados en una palabra de 32 bits
__device__ __forceinline__ unsigned suma_par(unsigned xp, unsigned rp) {
#ifdef HAY_SUMA2
    return suma_fp16x2(xp, rp);
#else
    return suma_fp16(xp & 0xffffu, rp & 0xffffu) | (suma_fp16(xp >> 16, rp >> 16) << 16);
#endif
}

// una posicion de la pasada B: de la clave guardada al byte int8. Sin mantisas ni productos.
__device__ __forceinline__ unsigned paso_b(unsigned cel, int emax, unsigned rec) {
    const unsigned p = cel & 0x1fffffu;
    int sh = DESP + emax - (int)((cel >> 21) & 63u);
    sh = sh < 0 ? 0 : (sh > 31 ? 31 : sh);
    unsigned qq = (p * rec + (1u << sh >> 1)) >> sh;
    qq = qq < 127u ? qq : 127u;
    return ((cel >> 31) ? (unsigned)(-(int)qq) : qq) & 0xffu;
}

extern "C" __global__ void __launch_bounds__(HILOS)
sk20_norm_quant(
    const unsigned short* __restrict__ x,    // [M, K] bits fp16
    const unsigned short* __restrict__ res,  // [M, K] bits fp16 (ignorado si HAY_RES = 0)
    const unsigned short* __restrict__ w,    // [K] bits fp16 (peso del RMSNorm)
    const unsigned* __restrict__ gbits,      // [1] factor global (fp32)
    unsigned short* __restrict__ res_out,    // [M, K] x + residual
    signed char* __restrict__ q,             // [M, K]
    unsigned* __restrict__ esc,              // [M] bits fp32
    int K, int XS, int HAY_RES)
{
    extern __shared__ unsigned s_cel[];       // clave por elemento: la fila NO vuelve a memoria
    __shared__ unsigned s_red[HILOS / 32];
    __shared__ unsigned long long s_acc[HILOS / 32];
    __shared__ int s_ref[HILOS / 32];
    const int fila = blockIdx.x;
    const int tid = threadIdx.x;
    const size_t base = (size_t)fila * XS;
    const size_t bo = (size_t)fila * K;

    // camino vectorizado: 8 fp16 por carga (uint4) y 8 int8 por escritura (uint2)
    const int V8 = ((K & 7) == 0 && (((base * 2) | (size_t)x | (size_t)res | (size_t)w
                                     | (size_t)res_out) & 15u) == 0) ? (K >> 3) : 0;
    const uint4* x4 = (const uint4*)(x + base);
    const uint4* r4 = (const uint4*)(res + base);
    const uint4* w4 = (const uint4*)w;
    uint4* o4 = (uint4*)(res_out + bo);
    uint2* q2 = (uint2*)(q + bo);

    // ── pasada A: suma, clave por elemento y cuadrados ──
    unsigned clave = 0;
    unsigned long long acc = 0;
    int eref = 0;
    if (V8) {
        for (int c = tid; c < V8; c += HILOS) {
            const uint4 xv = x4[c], rv = HAY_RES ? r4[c] : xv, wv = w4[c];
            uint4 ov;
            const unsigned* px = (const unsigned*)&xv;
            const unsigned* pr = (const unsigned*)&rv;
            const unsigned* pw = (const unsigned*)&wv;
            unsigned* po = (unsigned*)&ov;
            unsigned cel[8];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const unsigned rp = HAY_RES ? suma_par(px[j], pr[j]) : px[j];
                po[j] = rp;
                paso_a(rp & 0xffffu, pw[j] & 0xffffu, &clave, &cel[2 * j], &acc, &eref);
                paso_a(rp >> 16, pw[j] >> 16, &clave, &cel[2 * j + 1], &acc, &eref);
            }
            if (HAY_RES) o4[c] = ov;
#pragma unroll
            for (int j = 0; j < 8; ++j) s_cel[c * 8 + j] = cel[j];
        }
    } else {
        for (int i = tid; i < K; i += HILOS) {
            unsigned cel;
            const unsigned rb = HAY_RES ? (suma_par(x[base + i], res[base + i]) & 0xffffu)
                                        : (unsigned)x[base + i];
            paso_a(rb, w[i], &clave, &cel, &acc, &eref);
            if (HAY_RES) res_out[bo + i] = (unsigned short)rb;
            s_cel[i] = cel;
        }
    }
#ifdef ETAPA
    if (ETAPA < 2) { if (tid == 0) esc[fila] = clave + (unsigned)acc; return; }
#endif

    // reduccion: maximo de la clave y suma de cuadrados (con su exponente) en un solo pase
    clave = max_lanes(clave);
    for (int k = 1; k < 32; k <<= 1) {
        const int eo = __shfl_xor_sync(0xffffffffu, eref, k);
        const unsigned long long ao = __shfl_xor_sync(0xffffffffu, acc, k);
        unir_cuad(ao, eo, &acc, &eref);
    }
    if ((tid & 31) == 0) {
        s_red[tid >> 5] = clave;
        s_acc[tid >> 5] = acc;
        s_ref[tid >> 5] = eref;
    }
    __syncthreads();
    if (tid < 32) {                                   // el primer warp junta los HILOS/32 parciales
        const int hay = tid < HILOS / 32;
        clave = hay ? s_red[tid] : 0u;
        acc = hay ? s_acc[tid] : 0ull;
        eref = hay ? s_ref[tid] : 0;
        clave = max_lanes(clave);
        for (int k = 1; k < 32; k <<= 1) {
            const int eo = __shfl_xor_sync(0xffffffffu, eref, k);
            const unsigned long long ao = __shfl_xor_sync(0xffffffffu, acc, k);
            unir_cuad(ao, eo, &acc, &eref);
        }
        if (tid == 0) { s_red[0] = clave; s_acc[0] = acc; s_ref[0] = eref; }
    }
    __syncthreads();
    clave = s_red[0];

    const int emax = (int)(clave >> 21);                  // e_r + e_w del maximo (normalizado)
    const unsigned pmax = clave & 0x1fffffu;              // m_r * m_w del maximo (normalizado)
    const unsigned rec = pmax ? (unsigned)(((127ull << DESP) + (pmax >> 1)) / pmax) : 0u;
#ifdef ETAPA
    if (ETAPA < 3) { if (tid == 0) esc[fila] = rec; return; }
#endif

    // ── pasada B: de la clave al byte, sin volver a tocar r ni w ──
    if (V8) {
        for (int c = tid; c < V8; c += HILOS) {
            unsigned lo = 0, hi = 0;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const unsigned a = paso_b(s_cel[c * 8 + 2 * j], emax, rec);
                const unsigned b = paso_b(s_cel[c * 8 + 2 * j + 1], emax, rec);
                if (j < 2) lo |= (a | (b << 8)) << (16 * j);
                else hi |= (a | (b << 8)) << (16 * (j - 2));
            }
            uint2 qv; qv.x = lo; qv.y = hi;
            q2[c] = qv;
        }
    } else {
        for (int i = tid; i < K; i += HILOS)
            q[bo + i] = (signed char)paso_b(s_cel[i], emax, rec);
    }
#ifdef ETAPA
    if (ETAPA < 4) return;
#endif

    // ── cola: la escala fp32, sin una sola division de 64 bits ──
    if (tid == 0) {
        acc = s_acc[0];
        eref = s_ref[0];
        // acc = sum (|r_i| * 2^(SQB+25-eref))^2. Con sqrt(t/K) = sqrt(t)/sqrt(K) no hace falta
        // dividir: las dos raices salen de isqrt32 y la unica division es la final, de 32 bits.
        int st = 0;
        if (acc) {
            const int alto = 63 - __clzll((long long)acc);
            st = ((30 - alto) >> 1) << 1;             // corrimiento par hasta [2^30, 2^32)
            acc = st >= 0 ? (acc << st) : (acc >> (-st));
        }
        const unsigned rt = raiz16((unsigned)acc);            // sqrt(t) * 2^(st/2)
        unsigned kk = (unsigned)K;
        const int sk = ((30 - (31 - __clz((int)kk))) >> 1) << 1;
        kk <<= sk;
        const unsigned rk = raiz16(kk);                       // sqrt(K) * 2^(sk/2)
        // rms = (rt/rk) * 2^((sk-st)/2 + eref - 25 - SQB)
        // escala = pmax*2^(emax-50) * mg*2^eg * rk / (127 * rt) * 2^(-(sk-st)/2 - eref+25+SQB)
        int eg;
        const unsigned long long mg = fp32_a_mant(gbits[0], &eg);
        unsigned long long num = (unsigned long long)pmax * mg * (unsigned long long)rk;
        unsigned long long den = 127ull * (rt ? rt : 1u);
        int nsh = 0, dsh = 0;
        if (den >= (1ull << 16)) { dsh = (63 - __clzll((long long)den)) - 15; den >>= dsh; }
        if (num >= (1ull << 32)) { nsh = (63 - __clzll((long long)num)) - 31; num >>= nsh; }
        const unsigned cociente = (unsigned)num / (unsigned)(den ? den : 1ull);
        const int e = eg + (emax - 50) + nsh - dsh - ((sk - st) >> 1) - (eref - 25 - SQB);
        esc[fila] = q_a_fp32(cociente, e);
    }
}
