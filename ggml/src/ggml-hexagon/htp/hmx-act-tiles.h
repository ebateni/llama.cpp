#ifndef HMX_ACT_TILES_H
#define HMX_ACT_TILES_H

// fp16 HMX activation tile layout, shared by the producer (RMS_NORM_MUL tile output, unary-ops.c) and the consumer (matmul src1 flagged HTP_TENSOR_ACT_TILE, matmul-ops.c). Tile (r0, c0) is at index r0 * (K/32) + c0 (HTP_MM_HMX_TILE_N_ELMS fp16 per tile). Inside a tile, HVX vector r1/2 holds rows r1 and r1+1, produced by hvx_vec_f32_to_f16_shuff.
// An activation [M x K] is stored as (M/32) x (K/32) fp16 tiles of 32x32 (2048 bytes each). Rows past M in the last tile row are zero. The producer and the consumer MUST use the same conversion (act_tile_row_pair_store) to stay bit-identical with the fp32 -> fp16 activation transfer.

#include "dma-queue.h"
#include "hvx-utils.h"
#include "matmul-ops.h"

// Converts the fp32 row pair (row0, row1) into the tile slots of `vtcm_dst`. `r` is the row index inside the
// tile buffer `vtcm_dst` (tile row r / 32), `k_block` is K (multiple of 32).
static inline void act_tile_row_pair_store(
        __fp16 *restrict vtcm_dst,
        const float *restrict row0,
        const float *restrict row1,
        uint32_t r,
        uint32_t k_block,
        uint32_t k_valid,
        bool row0_valid,
        bool row1_valid) {

    uint32_t r0 = r / HTP_MM_HMX_TILE_N_ROWS;  // tile row index
    uint32_t r1 = r % HTP_MM_HMX_TILE_N_ROWS;  // intra-tile row idx

    uint32_t c = 0;
    for (; c + 32 <= k_valid; c += 32) {
        HVX_Vector v0 = Q6_V_vzero();
        HVX_Vector v1 = Q6_V_vzero();
        if (row0_valid) v0 = *(const HVX_Vector *)(row0 + c);
        if (row1_valid) v1 = *(const HVX_Vector *)(row1 + c);

        HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

        uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;  // tile column index
        uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

        HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
        tile[r1 / 2]     = v_out;
    }
    if (c < k_block) {
        HVX_Vector v0 = Q6_V_vzero();
        HVX_Vector v1 = Q6_V_vzero();
        if (row0_valid) v0 = *(const HVX_Vector *)(row0 + c);
        if (row1_valid) v1 = *(const HVX_Vector *)(row1 + c);

        uint32_t rem = k_valid - c;
        HVX_VectorPred mask = Q6_Q_vsetq2_R(rem > 0 ? rem * sizeof(float) : 0);
        v0 = Q6_V_vmux_QVV(mask, v0, Q6_V_vzero());
        v1 = Q6_V_vmux_QVV(mask, v1, Q6_V_vzero());

        HVX_Vector v_out = hvx_vec_f32_to_f16_shuff(v0, v1);

        uint32_t c0       = c / HTP_MM_HMX_TILE_N_COLS;  // tile column index
        uint32_t tile_idx = r0 * (k_block / HTP_MM_HMX_TILE_N_COLS) + c0;

        HVX_Vector *tile = (HVX_Vector *) (vtcm_dst + tile_idx * HTP_MM_HMX_TILE_N_ELMS);
        tile[r1 / 2]     = v_out;
    }
}

// Queues a contiguous DMA of `bytes` tile bytes from DDR `src` into VTCM `vtcm_dst`. The caller owns the
// queue entry and must pop it (FIFO) before reading `vtcm_dst`.
static inline void act_tiles_dma_push(dma_queue * q, void * vtcm_dst, dma_addr_t src, size_t bytes) {
    dma_queue_push(q, dma_make_data(vtcm_dst, src), bytes, bytes, bytes, 1);
}

#endif /* HMX_ACT_TILES_H */
