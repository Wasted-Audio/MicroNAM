#pragma once
#include <cstdint>
#include <cstring>
#include <cassert>
#include <cmath>

namespace MicroNAM {

// ─────────────────────────────────────────────────────────────────────────────
// A2 Lite WaveNet inference engine
//
// A2 Lite architecture (SlimmableContainer submodel at max_value 0.5):
//   23 layers × 3-channel
//   Kernel sizes: 6 everywhere, except layers 14 and 15 which use 15
//   Dilations:    [1,3,7,17,41,101,239] twice, then [1,13], then
//                 [1,3,7,17,41,101,239]
//   Every layer:  conv(K taps, dilation d) + conv_b + mixin_w * input
//                 -> LeakyReLU(0.01)
//                 -> head_sum[ch] += activated[ch]
//                 -> x = x + b1x1 + activated @ w1x1
//   Head:         16-tap causal FIR over the 3-channel head_sum across time,
//                 + head_b, then * head_scale
//
// Each layer keeps its recent inputs in a ring of (k-1)*d + 1 columns, plus
// NAM_BLOCK extra columns that mirror the start of the ring. The mirror means
// a tap read never wraps, so tap addresses are fixed pointers for a block and
// simply advance by one frame.
//
// Layers run in tiles of kTile frames: the nine weights of a tap are loaded
// once and used for every frame in the tile.
//
// Weight matrices live in a read-only array separate from the mutable state,
// so the compiler knows a state store cannot overwrite a weight.
//
// The head FIR reads a linear history (15 frames back plus the current block).
// head_scale is multiplied into the head weights at load time.
//
// load_weights expects exactly 1871 floats in the NAM file weight order.
// forward processes up to NAM_BLOCK samples per call.
// ─────────────────────────────────────────────────────────────────────────────

// Frames per tile. Four keeps the accumulators, the tap weights and the tap
// pointers in registers on x86 and on the M7. Larger tiles gain little and
// start spilling; smaller ones lose the weight reuse.
#ifndef A2LITE_TILE
  #define A2LITE_TILE 4
#endif

// Widest block the state buffers are sized for, in samples. Lower it when the
// block size is known to be small: every layer reserves two extra rings of
// this width, at 12 bytes per column. Must not exceed NAM_BLOCK.
#ifndef A2LITE_MAX_N
  #define A2LITE_MAX_N 1024
#endif

static constexpr int kTile = A2LITE_TILE;
static constexpr int kHeadTaps = 16;

template<size_t NAM_BLOCK>
class A2LiteNet
{
public:
  // ── API ───────────────────────────────────────────────────────────────────
  void reset() { reset_impl(); }
  void load_weights(const float* model_weights) { load_weights_impl(model_weights); }

  // Process up to NAM_BLOCK samples. Short blocks are fine: the ring mirror
  // covers the whole block width, so a partial block cannot overrun a tap.
  void forward(const float* input, float* output) noexcept
  {
    static_assert((int)NAM_BLOCK <= kMaxN, "NAM_BLOCK exceeds A2LITE_MAX_N");
    forward_impl(input, output, (int)NAM_BLOCK);
  }
  void forward(const float* input, float* output, int n) noexcept
  {
    assert(n >= 0 && n <= kMaxN);
    forward_impl(input, output, n);
  }
  // ──────────────────────────────────────────────────────────────────────────

private:
  static constexpr int NB = (int)NAM_BLOCK;
  // Mirror width, capped at the block size.
  static constexpr int kMaxN = (A2LITE_MAX_N < NB) ? A2LITE_MAX_N : NB;

  static inline float nam_leaky_relu(float x) noexcept
  {
    // Branchless: one compare-free form for both signs.
    return std::fmax(x, 0.0f) + 0.01f * std::fmin(x, 0.0f);
  }

  // ── Read-only per-layer weights, unchanged after load ────────────────────
  struct alignas(16) Layer3Weights
  {
    float wt[15 * 3 * 3];  // transposed conv: [tap][in][out]
    float conv_b[3];
    float mixin_w[3];
    float w1x1[9];         // [out][in]
    float b1x1[3];
  };

  // ── Mutable per-layer state ──────────────────────────────────────────────
  struct Layer3
  {
    int    k;                      // kernel size
    int    d;                      // dilation
    int    cols;                   // ring columns: ss + kMaxN
    int    ptr;                    // ring write column at block start
    float* state;                  // (cols + NB) * 3 floats
  };

  Layer3Weights       weights_[23];   // ~11 KB
  Layer3              layers[23];

  // State: per layer, (ss + kMaxN) ring columns followed by a kMaxN mirror.
  static constexpr int kStateFloats()
  {
    const int ks[23] = { 6,6,6,6,6,6,6, 6,6,6,6,6,6,6, 15,15, 6,6,6,6,6,6,6 };
    const int ds[23] = { 1,3,7,17,41,101,239, 1,3,7,17,41,101,239, 1,13, 1,3,7,17,41,101,239 };
    int n = 0;
    for (int l = 0; l < 23; ++l) n += ((ks[l] - 1) * ds[l] + 1 + 2 * kMaxN) * 3;
    return n;
  }

  alignas(16) float state_pool[kStateFloats()];

  // Head FIR history: 15 past frames + this block's frames.
  alignas(16) float head_hist[kHeadTaps - 1 + kMaxN][3];

  // ── Block working buffers ────────────────────────────────────────────────
  alignas(16) float buf_a[kMaxN][3];
  alignas(16) float buf_b[kMaxN][3];

  void init_state_ptrs()
  {
    static constexpr int ks[23] = {
      6, 6, 6, 6, 6, 6, 6,  6, 6, 6, 6, 6, 6, 6,
      15, 15,
      6, 6, 6, 6, 6, 6, 6
    };
    static constexpr int ds[23] = {
      1, 3, 7, 17, 41, 101, 239,
      1, 3, 7, 17, 41, 101, 239,
      1, 13,
      1, 3, 7, 17, 41, 101, 239
    };

    int soff = 0;
    for (int l = 0; l < 23; ++l)
    {
      Layer3&   L  = layers[l];
      const int k  = ks[l];
      const int d  = ds[l];
      const int ss = (k - 1) * d + 1;

      L.k          = k;
      L.d          = d;
      L.cols       = ss + kMaxN;
      L.ptr        = 0;
      L.state      = state_pool + soff;
      soff        += (L.cols + kMaxN) * 3;
    }
  }

  void reset_impl()
  {
    for (int l = 0; l < 23; ++l)
    {
      Layer3& L = layers[l];
      std::memset(L.state, 0, sizeof(float) * (L.cols + kMaxN) * 3);
      L.ptr = 0;
    }
    std::memset(head_hist, 0, sizeof(head_hist));
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // load_weights — float[1871]
  //
  // File weight order:
  //   rechannel: [3]
  //   per layer: conv[out][in][k], conv_b[3], mixin_w[3], w1x1[out][in], b1x1[3]
  //   head:      head_w[channel][tap] (3 channels outer, 16 taps inner)
  //              head_b
  //              head_scale
  // ─────────────────────────────────────────────────────────────────────────────
  void load_weights_impl(const float* model_weights)
  {
    init_state_ptrs();
    reset_impl();

    const float* w = model_weights;

    for (int c = 0; c < 3; ++c) rechannel_w[c] = *w++;

    for (int l = 0; l < 23; ++l)
    {
      Layer3Weights& W = weights_[l];
      const int K = layers[l].k;

      // Source is PyTorch [out][in][k]; transpose to [tap][in][out]
      for (int o = 0; o < 3; ++o)
        for (int i = 0; i < 3; ++i)
          for (int k = 0; k < K; ++k)
            W.wt[(k * 3 + i) * 3 + o] = *w++;

      for (int o = 0; o < 3; ++o) W.conv_b[o]  = *w++;
      for (int o = 0; o < 3; ++o) W.mixin_w[o] = *w++;
      for (int o = 0; o < 3; ++o)
        for (int i = 0; i < 3; ++i)
          W.w1x1[o * 3 + i] = *w++;
      for (int o = 0; o < 3; ++o) W.b1x1[o] = *w++;
    }

    // Head FIR + bias + scale. File stream is [channel][tap].
    for (int c = 0; c < 3; ++c)
      for (int t = 0; t < kHeadTaps; ++t)
        head_w[t][c] = *w++;

    head_b     = *w++;
    head_scale = *w++;

    assert((int)(w - model_weights) == 1871);

    // Fold the output gain into the head weights so the head pass costs no
    // extra multiply per sample.
    for (int t = 0; t < kHeadTaps; ++t)
      for (int c = 0; c < 3; ++c) head_w[t][c] *= head_scale;
    head_b *= head_scale;
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // Per-layer block processing
  // ─────────────────────────────────────────────────────────────────────────────
  template<int K, int T>
  inline void conv_tile(const float* const* tap, const Layer3Weights& __restrict W,
                        const float* __restrict cond, float (*z)[3]) const noexcept
  {
    const float* __restrict cb = W.conv_b;
    const float* __restrict mx = W.mixin_w;
    const float* __restrict wt = W.wt;

    float z0[T], z1[T], z2[T];
    for (int f = 0; f < T; ++f)
    {
      z0[f] = cb[0] + mx[0] * cond[f];
      z1[f] = cb[1] + mx[1] * cond[f];
      z2[f] = cb[2] + mx[2] * cond[f];
    }

    for (int t = 0; t < K; ++t)
    {
      const float* __restrict wk = wt + t * 9;
      const float w00 = wk[0], w01 = wk[1], w02 = wk[2];
      const float w10 = wk[3], w11 = wk[4], w12 = wk[5];
      const float w20 = wk[6], w21 = wk[7], w22 = wk[8];
      const float* __restrict s = tap[t];
      for (int f = 0; f < T; ++f)
      {
        const float s0 = s[0], s1 = s[1], s2 = s[2];
        z0[f] += s0 * w00 + s1 * w10 + s2 * w20;
        z1[f] += s0 * w01 + s1 * w11 + s2 * w21;
        z2[f] += s0 * w02 + s1 * w12 + s2 * w22;
        s += 3;
      }
    }
    for (int f = 0; f < T; ++f) { z[f][0] = z0[f]; z[f][1] = z1[f]; z[f][2] = z2[f]; }
  }

  // Activation + head accumulation + 1×1 residual for one tile.
  template<int T>
  inline void finish_tile(const Layer3Weights& __restrict W,
                          const float (* __restrict ins)[3],
                          float (* __restrict outs)[3],
                          float (* __restrict head)[3],
                          const float (* __restrict z)[3]) const noexcept
  {
    const float* __restrict w1 = W.w1x1;
    const float* __restrict b1 = W.b1x1;

    for (int f = 0; f < T; ++f)
    {
      const float a0 = nam_leaky_relu(z[f][0]);
      const float a1 = nam_leaky_relu(z[f][1]);
      const float a2 = nam_leaky_relu(z[f][2]);

      head[f][0] += a0;
      head[f][1] += a1;
      head[f][2] += a2;

      outs[f][0] = ins[f][0] + b1[0] + (a0 * w1[0] + a1 * w1[1] + a2 * w1[2]);
      outs[f][1] = ins[f][1] + b1[1] + (a0 * w1[3] + a1 * w1[4] + a2 * w1[5]);
      outs[f][2] = ins[f][2] + b1[2] + (a0 * w1[6] + a1 * w1[7] + a2 * w1[8]);
    }
  }

  template<int K>
  void layer3(int li, const float (* __restrict ins)[3], float (* __restrict outs)[3],
              const float* __restrict cond, float (* __restrict head)[3], int nf) noexcept
  {
    Layer3& L = layers[li];
    const Layer3Weights& __restrict W = weights_[li];
    float* const __restrict st = L.state;
    const int cols = L.cols;

    // ── Tap pointers, resolved once per block ─────────────────────────────
    // The ring mirror guarantees these reads never wrap.
    const float* tap[15];
    for (int t = 0; t < K; ++t)
    {
      int c = L.ptr - (K - 1 - t) * L.d;
      if (c < 0) c += cols;
      tap[t] = st + c * 3;
    }

    // ── Ring write cursor ─────────────────────────────────────────────────
    int    col = L.ptr;
    float* wp  = st + col * 3;

    int f0 = 0;
    for (; f0 + kTile <= nf; f0 += kTile)
    {
      const float* is = &ins[f0][0];
      for (int f = 0; f < kTile; ++f)
      {
        const float v0 = is[0], v1 = is[1], v2 = is[2];
        is += 3;
        wp[0] = v0; wp[1] = v1; wp[2] = v2;
        // Keep the mirror tail in step with the ring head.
        if (col < kMaxN) { float* q = st + (col + cols) * 3; q[0] = v0; q[1] = v1; q[2] = v2; }
        wp += 3;
        if (++col == cols) { col = 0; wp = st; }
      }

      alignas(16) float z[kTile][3];
      conv_tile<K, kTile>(tap, W, cond + f0, z);
      for (int t = 0; t < K; ++t) tap[t] += kTile * 3;

      finish_tile<kTile>(W, ins + f0, outs + f0, head + f0, z);
    }

    for (; f0 < nf; ++f0)
    {
      const float v0 = ins[f0][0], v1 = ins[f0][1], v2 = ins[f0][2];
      wp[0] = v0; wp[1] = v1; wp[2] = v2;
      if (col < kMaxN) { float* q = st + (col + cols) * 3; q[0] = v0; q[1] = v1; q[2] = v2; }
      wp += 3;
      if (++col == cols) { col = 0; wp = st; }

      alignas(16) float z[1][3];
      conv_tile<K, 1>(tap, W, cond + f0, z);
      for (int t = 0; t < K; ++t) tap[t] += 3;

      finish_tile<1>(W, ins + f0, outs + f0, head + f0, z);
    }

    L.ptr = col;
  }

  // ─────────────────────────────────────────────────────────────────────────────
  // forward — process n samples
  // ─────────────────────────────────────────────────────────────────────────────
  void forward_impl(const float* input, float* output, int n) noexcept
  {
    // ── Rechannel: 1→3 straight into buf_a, head history cleared ──────────
    float (* __restrict hist)[3] = head_hist + (kHeadTaps - 1);
    for (int f = 0; f < n; ++f)
    {
      const float x = input[f];
      buf_a[f][0] = rechannel_w[0] * x;
      buf_a[f][1] = rechannel_w[1] * x;
      buf_a[f][2] = rechannel_w[2] * x;
      hist[f][0] = 0.0f; hist[f][1] = 0.0f; hist[f][2] = 0.0f;
    }
    std::memset(buf_a + n, 0, sizeof(float) * 3 * (kMaxN - n));

    // ── 23 layers; layer-outer, sample-inner ──────────────────────────────
    const float (*ping)[3] = buf_a;
    float       (*pong)[3] = buf_b;

    for (int l = 0; l < 23; ++l)
    {
      const int K = layers[l].k;
      if (K == 6) layer3<6>(l, ping, pong, input, hist, n);
      else        layer3<15>(l, ping, pong, input, hist, n);

      float (*tmp)[3] = pong;
      pong = const_cast<float (*)[3]>(ping);
      ping = tmp;
    }

    // ── Head: 16-tap causal FIR over the linear history ───────────────────
    head_fir(output, n);

    // ── Shift history: keep the last 15 frames ───────────────────────────
    std::memmove(head_hist, head_hist + n, sizeof(float) * 3 * (kHeadTaps - 1));
  }

  void head_fir(float* output, int n) noexcept
  {
    for (int f = 0; f < n; ++f)
    {
      const float (* __restrict h)[3] = head_hist + f;
      float y0 = 0.0f, y1 = 0.0f, y2 = 0.0f;
      for (int t = 0; t < kHeadTaps; ++t)
      {
        const float* __restrict hw = head_w[t];
        y0 += h[t][0] * hw[0];
        y1 += h[t][1] * hw[1];
        y2 += h[t][2] * hw[2];
      }
      output[f] = head_b + y0 + y1 + y2;
    }
  }

  // ── Persistent weights: rechannel + head ────────────────────────────────
  float rechannel_w[3];
  float head_w[kHeadTaps][3];
  float head_b;
  float head_scale;   // already folded into head_w and head_b at load
};

} // namespace MicroNAM
