// Fused paged-attention decode kernel for TinyServe (CPU, float32).
//
// Each sequence in the batch contributes one query token. Its keys and values
// are read straight from the shared KV pool through its block table, so there
// is no gather into a temporary tensor and no padding to the longest sequence.
//
// Token j of sequence b lives in pool slot
//     block_tables[b][j / block_size] * block_size + j % block_size.
// A contiguous cache is the special case of one block of max_seq_len tokens.
//
// Work is split over (sequence, kv_head) pairs. The query heads that share a
// KV head (grouped-query attention) are processed together, so each K/V row is
// loaded once per group rather than once per query head. Softmax is computed
// online (flash-attention style) over tiles of tokens in a single pass.

#include <torch/extension.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {

constexpr int64_t kTile = 32;  // tokens scored before each softmax rescale

inline float dot(const float* __restrict a, const float* __restrict b, int64_t n) {
  float acc = 0.0f;
#pragma omp simd reduction(+ : acc)
  for (int64_t d = 0; d < n; ++d) acc += a[d] * b[d];
  return acc;
}

// expf for x <= 0, accurate to a couple of ulp. Unlike std::exp it vectorizes
// without -ffast-math. Cephes-style: x = n*ln2 + r, then a polynomial for e^r.
#pragma omp declare simd
inline float exp_nonpositive(float x) {
  x = std::max(x, -87.0f);  // keeps 2^n a normal float; e^-87 is ~1e-38 anyway
  const float n = std::floor(x * 1.44269504f + 0.5f);
  const float r = x - n * 0.693359375f + n * 2.12194440e-4f;
  float p = 1.9875691500e-4f;
  p = p * r + 1.3981999507e-3f;
  p = p * r + 8.3334519073e-3f;
  p = p * r + 4.1665795894e-2f;
  p = p * r + 1.6666665459e-1f;
  p = p * r + 5.0000001201e-1f;
  p = p * r * r + r + 1.0f;
  const int32_t bits = (static_cast<int32_t>(n) + 127) << 23;  // 2^n
  float scale;
  std::memcpy(&scale, &bits, sizeof(scale));
  return p * scale;
}

// Attention for the `group` query heads of one sequence that share one KV head.
// kHeadDim > 0 fixes head_dim at compile time so the inner loops fully unroll;
// kHeadDim == 0 uses the runtime value.
template <int64_t kHeadDim>
void attend_one(const float* q,        // (group, D)
                const float* k_cache,  // (num_slots, n_kv_heads, D)
                const float* v_cache,  // (num_slots, n_kv_heads, D)
                const int64_t* table,  // this sequence's block table
                int64_t context_len, int64_t block_size, int64_t kv_head, int64_t n_kv_heads,
                int64_t group, int64_t head_dim, float scale,
                float* out) {  // (group, D)
  const int64_t D = kHeadDim > 0 ? kHeadDim : head_dim;
  std::vector<float> acc(group * D, 0.0f);
  std::vector<float> scores(group * kTile);
  std::vector<float> running_max(group, -std::numeric_limits<float>::infinity());
  std::vector<float> denom(group, 0.0f);
  int64_t offsets[kTile];

  for (int64_t start = 0; start < context_len; start += kTile) {
    const int64_t n = std::min(kTile, context_len - start);

    // Resolve slot addresses once per tile; both K and V use them.
    int64_t block = start / block_size, in_block = start % block_size;
    for (int64_t t = 0; t < n; ++t) {
      const int64_t slot = table[block] * block_size + in_block;
      offsets[t] = (slot * n_kv_heads + kv_head) * D;
      if (++in_block == block_size) {
        in_block = 0;
        ++block;
      }
    }

    // Scores for the tile, then rescale the accumulators to the new maximum.
    for (int64_t h = 0; h < group; ++h) {
      float* s = scores.data() + h * kTile;
      float tile_max = -std::numeric_limits<float>::infinity();
      for (int64_t t = 0; t < n; ++t) {
        s[t] = dot(q + h * D, k_cache + offsets[t], D) * scale;
        tile_max = std::max(tile_max, s[t]);
      }
      const float new_max = std::max(running_max[h], tile_max);
      const float correction = std::exp(running_max[h] - new_max);
      running_max[h] = new_max;
      denom[h] *= correction;
      float* a = acc.data() + h * D;
#pragma omp simd
      for (int64_t d = 0; d < D; ++d) a[d] *= correction;
      float tile_sum = 0.0f;
#pragma omp simd reduction(+ : tile_sum)
      for (int64_t t = 0; t < n; ++t) {
        s[t] = exp_nonpositive(s[t] - new_max);
        tile_sum += s[t];
      }
      denom[h] += tile_sum;
    }

    // Accumulate probability-weighted values.
    for (int64_t t = 0; t < n; ++t) {
      const float* v = v_cache + offsets[t];
      for (int64_t h = 0; h < group; ++h) {
        const float p = scores[h * kTile + t];
        float* a = acc.data() + h * D;
#pragma omp simd
        for (int64_t d = 0; d < D; ++d) a[d] += p * v[d];
      }
    }
  }

  for (int64_t h = 0; h < group; ++h) {
    const float inv = denom[h] > 0.0f ? 1.0f / denom[h] : 0.0f;
#pragma omp simd
    for (int64_t d = 0; d < D; ++d) out[h * D + d] = acc[h * D + d] * inv;
  }
}

}  // namespace

// q: (B, H, D). k_cache, v_cache: (num_slots, H_kv, D) for one layer.
// block_tables: (B, W) int64. context_lens: (B,) int64. Returns (B, H, D).
torch::Tensor paged_attention_decode(const torch::Tensor& q, const torch::Tensor& k_cache,
                                     const torch::Tensor& v_cache,
                                     const torch::Tensor& block_tables,
                                     const torch::Tensor& context_lens, int64_t block_size,
                                     double scale) {
  TORCH_CHECK(q.dim() == 3 && k_cache.dim() == 3 && v_cache.dim() == 3,
              "expected q (B, H, D) and caches (num_slots, H_kv, D)");
  TORCH_CHECK(q.scalar_type() == torch::kFloat32 && k_cache.scalar_type() == torch::kFloat32 &&
                  v_cache.scalar_type() == torch::kFloat32,
              "paged_attention_decode supports float32 only");
  TORCH_CHECK(block_tables.scalar_type() == torch::kInt64 &&
                  context_lens.scalar_type() == torch::kInt64,
              "block_tables and context_lens must be int64");
  TORCH_CHECK(k_cache.sizes() == v_cache.sizes(), "k_cache and v_cache shapes differ");
  TORCH_CHECK(q.size(2) == k_cache.size(2), "head_dim mismatch");
  TORCH_CHECK(q.size(1) % k_cache.size(1) == 0, "n_heads must divide by n_kv_heads");
  TORCH_CHECK(block_tables.dim() == 2 && block_tables.size(0) == q.size(0) &&
                  context_lens.numel() == q.size(0),
              "need one block table and one context length per sequence");
  TORCH_CHECK(block_size > 0, "block_size must be positive");
  TORCH_CHECK(k_cache.is_contiguous() && v_cache.is_contiguous(), "caches must be contiguous");

  const auto qc = q.contiguous();
  const auto tables = block_tables.contiguous();
  const auto lens = context_lens.contiguous();
  const int64_t B = qc.size(0), H = qc.size(1), D = qc.size(2);
  const int64_t n_kv_heads = k_cache.size(1);
  const int64_t group = H / n_kv_heads;
  const int64_t width = tables.size(1);

  const int64_t* lens_ptr = lens.data_ptr<int64_t>();
  for (int64_t b = 0; b < B; ++b) {
    TORCH_CHECK(lens_ptr[b] >= 0 && lens_ptr[b] <= width * block_size,
                "context length exceeds the block table");
  }

  auto out = torch::empty_like(qc);
  const float* q_ptr = qc.data_ptr<float>();
  const float* k_ptr = k_cache.data_ptr<float>();
  const float* v_ptr = v_cache.data_ptr<float>();
  const int64_t* table_ptr = tables.data_ptr<int64_t>();
  float* out_ptr = out.data_ptr<float>();
  const float scale_f = static_cast<float>(scale);

  auto attend = attend_one<0>;
  if (D == 64) attend = attend_one<64>;
  if (D == 128) attend = attend_one<128>;

  // Dynamic scheduling balances sequences of very different lengths.
#pragma omp parallel for collapse(2) schedule(dynamic, 1) num_threads(at::get_num_threads())
  for (int64_t b = 0; b < B; ++b) {
    for (int64_t g = 0; g < n_kv_heads; ++g) {
      const int64_t head0 = g * group;  // query heads [head0, head0 + group) share KV head g
      attend(q_ptr + (b * H + head0) * D, k_ptr, v_ptr, table_ptr + b * width, lens_ptr[b],
             block_size, g, n_kv_heads, group, D, scale_f, out_ptr + (b * H + head0) * D);
    }
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("paged_attention_decode", &paged_attention_decode,
        "Single-token attention reading K/V from a paged pool via block tables");
}
