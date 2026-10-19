// Shared-memory lifetimes: Q is live for the entire CTA; K then V share one tile.
constexpr int kHeadDim = 256;
constexpr int kQueryHeads = 24;
constexpr int kKvHeads = 2;
constexpr int kPageTokens = 64;
constexpr int kSelectedBlocks = 512;
constexpr float kScale = 0.0625F;

template <typename StorageT>
__device__ __forceinline__ void stage_kv_vector(
    __nv_bfloat16* dst, const StorageT* src, bool valid) {
    if constexpr (std::is_same_v<StorageT, __nv_bfloat16>) {
        ops::cp_async_zfill<16, ops::Cache::cg>(dst, src, valid ? 16 : 0);
    } else {
        if (valid) {
            const unsigned long long raw = *reinterpret_cast<const unsigned long long*>(src);
            const auto* fp8 = reinterpret_cast<const __nv_fp8_e4m3*>(&raw);
            __nv_bfloat16 bf16_vals[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                bf16_vals[i] = __float2bfloat16_rn(static_cast<float>(fp8[i]));
            }
            *reinterpret_cast<float4*>(dst) = *reinterpret_cast<float4*>(bf16_vals);
        } else {
            float4 zero = {0.0f, 0.0f, 0.0f, 0.0f};
            *reinterpret_cast<float4*>(dst) = zero;
        }
    }
}

// GQA tiled MMA: one CTA per (kv_head, query token). The 12 query heads that share a KV head
// stage each K/V tile once and run m16n8k16 over it. Online softmax stays FP32.
constexpr int kMmaHeads     = 16; // 12 real heads, 4 padded
constexpr int kMmaBN        = 64;
constexpr int kMmaWarps     = 4;
constexpr int kMmaThreads   = 32 * kMmaWarps;
constexpr int kMmaNTiles    = kMmaBN / 8;
constexpr int kHeadsPerKv   = 12;

template <typename StorageT>
__global__ __launch_bounds__(kMmaThreads)
void selected_attention_prefill_kernel(
    const __nv_bfloat16* __restrict__ query, const std::int32_t* __restrict__ token_indices,
    int table_row, const std::int32_t* __restrict__ selected_blocks,
    const std::int32_t* __restrict__ selected_counts, const std::int32_t* __restrict__ block_tables,
    int logical_pages, const StorageT* __restrict__ key_pages,
    const StorageT* __restrict__ value_pages, __nv_bfloat16* __restrict__ output) {
    using ops::cp_async_zfill;
    using ops::cp_commit;
    using ops::cp_wait;
    using ops::ldmatrix_x2;
    using ops::ldmatrix_x4;
    using ops::mma_bf16;
    using ops::smem_addr;
    using ops::detail::gemm_swz64;
    using Cache = ops::Cache;

    __shared__ __align__(16) __nv_bfloat16 Qs[kMmaHeads * kHeadDim];
    __shared__ __align__(16) __nv_bfloat16 Ks[kMmaBN * kHeadDim];
    auto* Vs = Ks; // K is dead after QK; V reuses its shared storage.
    __shared__ float s_scores[kMmaHeads * kMmaBN];

    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int kv_head = static_cast<int>(blockIdx.x) % kKvHeads;
    const int token   = static_cast<int>(blockIdx.x) / kKvHeads;
    const int q0      = kv_head * kHeadsPerKv;

    const int token_index     = token_indices[token];
    const int complete_blocks = (token_index + 1) / 4;
    const int tail_count      = (token_index + 1) & 3;
    const int selected_count  = selected_counts[token];
    const int total           = selected_count * 4 + tail_count;
    const auto* selected      = selected_blocks + static_cast<std::int64_t>(token) * kSelectedBlocks;
    const bool is_dense = (complete_blocks <= kSelectedBlocks && selected_count == complete_blocks);
    const auto* block_table_row = block_tables + table_row * logical_pages;

    const auto* q_base =
        query + static_cast<std::int64_t>(token) * kQueryHeads * kHeadDim + q0 * kHeadDim;
    for (int item = tid; item < kMmaHeads * (kHeadDim / 8); item += kMmaThreads) {
        const int row = item / (kHeadDim / 8);
        const int k8  = item - row * (kHeadDim / 8);
        auto* dst     = &Qs[row * kHeadDim + gemm_swz64(row, k8 * 8)];
        const bool valid = row < kHeadsPerKv;
        const __nv_bfloat16* src = q_base + (valid ? row : 0) * kHeadDim + k8 * 8;
        cp_async_zfill<16, Cache::cg>(dst, src, valid ? 16 : 0);
    }
    cp_commit();
    cp_wait<0>();
    __syncthreads();

    float acc[3][8] = {};
    float running_max[3] = {-__int_as_float(0x7F800000), -__int_as_float(0x7F800000),
                            -__int_as_float(0x7F800000)};
    float running_sum[3] = {};

    const int a_mat    = lane >> 3;
    const int a_rin    = lane & 7;
    const int a_rowoff = a_rin + ((a_mat & 1) << 3);
    const int a_coloff = (a_mat >> 1) << 3;
    const int b_rin    = lane & 7;
    const int b_koff   = ((lane >> 3) & 1) << 3;
    const int arow     = a_rowoff;

    for (int start = 0; start < total; start += kMmaBN) {
        const int count = min(kMmaBN, total - start);
        for (int item = tid; item < kMmaBN * (kHeadDim / 8); item += kMmaThreads) {
            const int row = item / (kHeadDim / 8);
            const int k8  = item - row * (kHeadDim / 8);
            auto* kdst    = &Ks[row * kHeadDim + gemm_swz64(row, k8 * 8)];
            const bool valid = row < count;
            int cand = 0;
            if (valid) {
                cand = is_dense ? (start + row)
                                : selected_token(start + row, selected_count, complete_blocks,
                                                 selected);
            }
            const int phys = valid ? block_table_row[cand / kPageTokens] : 0;
            const std::int64_t kv_off =
                ((static_cast<std::int64_t>(phys) * kKvHeads + kv_head) * kPageTokens +
                 (valid ? (cand % kPageTokens) : 0)) *
                    kHeadDim +
                k8 * 8;
            stage_kv_vector(kdst, key_pages + kv_off, valid);
        }
        if constexpr (std::is_same_v<StorageT, __nv_bfloat16>) {
            cp_commit();
            cp_wait<0>();
        }
        __syncthreads();

        float tile_score[kMmaNTiles][4] = {};
        if (warp == 0) {
#pragma unroll
            for (int ki = 0; ki < kHeadDim / 16; ++ki) {
                unsigned af[4];
                ldmatrix_x4(af[0], af[1], af[2], af[3],
                            smem_addr(&Qs[arow * kHeadDim + gemm_swz64(arow, ki * 16 + a_coloff)]));
#pragma unroll
                for (int nt = 0; nt < kMmaNTiles; ++nt) {
                    unsigned bf[2];
                    const int brow = nt * 8 + b_rin;
                    ldmatrix_x2(bf[0], bf[1],
                                smem_addr(&Ks[brow * kHeadDim + gemm_swz64(brow, ki * 16 + b_koff)]));
                    mma_bf16(tile_score[nt][0], tile_score[nt][1], tile_score[nt][2],
                             tile_score[nt][3], af[0], af[1], af[2], af[3], bf[0], bf[1]);
                }
            }
            const int gid = lane >> 2;
            const int lid = lane & 3;
            const int r0  = gid;
            const int r1  = r0 + 8;
#pragma unroll
            for (int nt = 0; nt < kMmaNTiles; ++nt) {
                const int c0 = nt * 8 + 2 * lid;
                const int c1 = c0 + 1;
                auto store = [&](int head, int col, float value) {
                    const float ninf = -__int_as_float(0x7F800000);
                    s_scores[head * kMmaBN + col] =
                        (head < kHeadsPerKv && col < count) ? value * kScale : ninf;
                };
                store(r0, c0, tile_score[nt][0]);
                store(r0, c1, tile_score[nt][1]);
                store(r1, c0, tile_score[nt][2]);
                store(r1, c1, tile_score[nt][3]);
            }
        }
        __syncthreads();

        // Every warp has finished reading K before its storage becomes V.
        for (int item = tid; item < kMmaBN * (kHeadDim / 8); item += kMmaThreads) {
            const int row = item / (kHeadDim / 8);
            const int k8 = item % (kHeadDim / 8);
            const bool valid = row < count;
            const int cand = valid ? (is_dense ? start + row :
                selected_token(start + row, selected_count, complete_blocks, selected)) : 0;
            const int phys = valid ? block_table_row[cand / kPageTokens] : 0;
            const std::int64_t offset =
                ((static_cast<std::int64_t>(phys) * kKvHeads + kv_head) * kPageTokens +
                 (valid ? cand % kPageTokens : 0)) * kHeadDim + k8 * 8;
            stage_kv_vector(&Vs[row * kHeadDim + k8 * 8], value_pages + offset, valid);
        }
        if constexpr (std::is_same_v<StorageT, __nv_bfloat16>) {
            cp_commit();
            cp_wait<0>();
        }
        __syncthreads();

#pragma unroll
        for (int hi = 0; hi < 3; ++hi) {
            const int head = warp + hi * kMmaWarps;
            float hmax     = -__int_as_float(0x7F800000);
            for (int col = lane; col < kMmaBN; col += 32) {
                hmax = fmaxf(hmax, s_scores[head * kMmaBN + col]);
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                hmax = fmaxf(hmax, __shfl_xor_sync(0xFFFFFFFF, hmax, off));
            }
            float hsum = 0.0F;
            for (int col = lane; col < kMmaBN; col += 32) {
                const float p = expf(s_scores[head * kMmaBN + col] - hmax);
                s_scores[head * kMmaBN + col] = p;
                hsum += p;
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                hsum += __shfl_xor_sync(0xFFFFFFFF, hsum, off);
            }
            const float next_max    = fmaxf(running_max[hi], hmax);
            const float prior_scale = running_sum[hi] == 0.0F ? 0.0F : expf(running_max[hi] - next_max);
            const float chunk_scale = expf(hmax - next_max);
#pragma unroll
            for (int i = 0; i < 8; ++i) { acc[hi][i] *= prior_scale; }
            running_sum[hi] = running_sum[hi] * prior_scale + hsum * chunk_scale;
            running_max[hi] = next_max;

            for (int col = 0; col < count; ++col) {
                const float p = s_scores[head * kMmaBN + col] * chunk_scale;
                const auto* v = &Vs[col * kHeadDim + lane * 8];
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc[hi][i] = fmaf(p, __bfloat162float(v[i]), acc[hi][i]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int hi = 0; hi < 3; ++hi) {
        const int head      = warp + hi * kMmaWarps;
        const float inv_sum = running_sum[hi] > 0.0F ? 1.0F / running_sum[hi] : 0.0F;
        auto* out_ptr       = output + static_cast<std::int64_t>(token) * kQueryHeads * kHeadDim +
                        (q0 + head) * kHeadDim + lane * 8;
        __nv_bfloat16 out_bf16[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            out_bf16[i] = __float2bfloat16_rn(acc[hi][i] * inv_sum);
        }
        *reinterpret_cast<float4*>(out_ptr) = *reinterpret_cast<float4*>(out_bf16);
    }
}

