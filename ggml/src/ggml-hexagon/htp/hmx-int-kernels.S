// Copyright (c) 2026 llama.cpp Authors. All rights reserved.

    .file "hmx-int-kernels.S"
    .text

// r0: uint16 output tile, r1: paired u8 activation tiles,
// r2: expanded signed i8 weight tiles, r3: integer bias table, r4: K-tile count.
    .balign 64
    .globl hmx_tile_dot_read_u16_int8_v81
    .type hmx_tile_dot_read_u16_int8_v81, @function
hmx_tile_dot_read_u16_int8_v81:
#if __HEXAGON_ARCH__ >= 73
    {
        p0 = cmp.eq(r4, #0)
        r6 = #31
        r7 = #1023
        r8 = #0
    }
    {
        if (p0) jumpr r31
        loop0(.L_k_tile, r4)
    }
    {
        mxclracc
    }
    {
        bias = mxmem2(r3)
    }
    .balign 32
.L_k_tile:
    {
        activation.ub = mxmem(r1, r6):deep:cm
        weight.b = mxmem(r2, r7):deep
    }
    {
        r1 = add(r1, #2048)
        r2 = add(r2, #1024)
    }:endloop0
    {
        cvt.uh = acc(r8):2x1
    }
    {
        mxmem(r0, r8) = cvt
    }
    {
        jumpr r31
    }
#else
    {
        jumpr r31
    }
#endif
    .size hmx_tile_dot_read_u16_int8_v81, .-hmx_tile_dot_read_u16_int8_v81
