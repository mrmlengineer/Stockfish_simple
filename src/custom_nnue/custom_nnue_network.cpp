/*
  Custom dual-head NNUEX runtime (full-recompute reference path).
*/

#include "custom_nnue_network.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(USE_AVX2) && (defined(__AVX2__) || defined(_M_AVX2))
  #include <immintrin.h>
  #define CUSTOM_NNUE_HAS_AVX2_INTRINSICS 1
#else
  #define CUSTOM_NNUE_HAS_AVX2_INTRINSICS 0
#endif

#include "../nnue/nnue_common.h"
#include "../position.h"

namespace Stockfish::CustomNNUE {

namespace {

using SteadyClock = std::chrono::steady_clock;

constexpr std::uint32_t kExpectedInputSize   = 737;
constexpr std::uint32_t kExpectedBucketCount = 16;
constexpr std::uint32_t kExpectedH1Total     = 512;
constexpr std::uint32_t kExpectedH1Psqt      = 128;
constexpr std::uint32_t kExpectedH1Pos       = 384;
constexpr std::uint32_t kExpectedPsqtH2      = 16;
constexpr std::uint32_t kExpectedPosH2       = 16;
constexpr std::uint32_t kExpectedPsqtH3      = 32;
constexpr std::uint32_t kExpectedPosH3       = 32;
constexpr std::uint32_t kExpectedOutputs     = 2;
constexpr std::size_t   kMaxActiveFeatures   = 33;  // 32 pieces + side-to-move bit

constexpr std::uint32_t kExpectedFeatureVariantId = 2;
constexpr std::uint32_t kExpectedBucketSchemeId   = 1;
constexpr std::uint32_t kExpectedOutputUnitType   = 1;
constexpr std::size_t   kNnuexHeaderSize          = 256;
constexpr std::uint32_t kNnuexVersion             = 1;
constexpr char          kNnuexMagic[8]            = {'C', 'D', 'N', 'N', 'U', 'E', 'X', '1'};
constexpr int           kDefaultEngineValueOutputScale = 1;

thread_local RuntimeMetrics* gRuntimeMetricsSink = nullptr;

struct ScopedRuntimeMetricTimer {
    explicit ScopedRuntimeMetricTimer(std::uint64_t* dst_) : dst(dst_) {
        if (dst)
            start = SteadyClock::now();
    }

    ~ScopedRuntimeMetricTimer() {
        if (dst)
            *dst += std::uint64_t(
              std::chrono::duration_cast<std::chrono::nanoseconds>(SteadyClock::now() - start).count());
    }

    std::uint64_t*          dst   = nullptr;
    SteadyClock::time_point start = {};
};

}  // namespace

ScopedRuntimeMetricsBinding::ScopedRuntimeMetricsBinding(RuntimeMetrics* sink) noexcept :
    prev_(gRuntimeMetricsSink) {
    gRuntimeMetricsSink = sink;
}

ScopedRuntimeMetricsBinding::~ScopedRuntimeMetricsBinding() noexcept { gRuntimeMetricsSink = prev_; }

namespace {

inline float clip01(float x) {
    return x < 0.0f ? 0.0f : (x > 1.0f ? 1.0f : x);
}

long long round_half_away_from_zero(double x) {
    return x >= 0.0 ? static_cast<long long>(std::floor(x + 0.5))
                    : static_cast<long long>(std::ceil(x - 0.5));
}

std::int64_t round_div_symmetric_i64(std::int64_t x, std::int32_t denom) {
    if (denom <= 0)
        return 0;
    const std::int64_t absX = x >= 0 ? x : -x;
    const std::int64_t q    = (absX + (denom / 2)) / denom;
    return x >= 0 ? q : -q;
}

std::uint8_t clip_to_hidden_q(std::int64_t x, std::int32_t hiddenQuantizedOne) {
    if (x < 0)
        return 0;
    if (x > hiddenQuantizedOne)
        return static_cast<std::uint8_t>(hiddenQuantizedOne);
    return static_cast<std::uint8_t>(x);
}

#if CUSTOM_NNUE_HAS_AVX2_INTRINSICS
void dense_layer_clip_q_avx2_out16(const std::uint8_t*              input,
                                   std::size_t                      inputDim,
                                   const std::vector<std::int8_t>&  weight,
                                   const std::vector<std::int32_t>& bias,
                                   std::int32_t                     weightScaleHidden,
                                   std::int32_t                     hiddenQuantizedOne,
                                   std::uint8_t*                    out) {
    __m256i acc0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias.data()));
    __m256i acc1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias.data() + 8));

    for (std::size_t i = 0; i < inputDim; ++i)
    {
        const std::uint8_t in = input[i];
        if (!in)
            continue;

        const __m256i vin16 = _mm256_set1_epi16(static_cast<std::int16_t>(in));
        const auto*   row   = weight.data() + i * 16;

        const __m128i w8    = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row));
        const __m256i w16   = _mm256_cvtepi8_epi16(w8);
        const __m256i prod16 = _mm256_mullo_epi16(vin16, w16);

        const __m128i prodLo16 = _mm256_castsi256_si128(prod16);
        const __m128i prodHi16 = _mm256_extracti128_si256(prod16, 1);
        acc0                   = _mm256_add_epi32(acc0, _mm256_cvtepi16_epi32(prodLo16));
        acc1                   = _mm256_add_epi32(acc1, _mm256_cvtepi16_epi32(prodHi16));
    }

    alignas(32) std::int32_t acc[16];
    _mm256_store_si256(reinterpret_cast<__m256i*>(acc), acc0);
    _mm256_store_si256(reinterpret_cast<__m256i*>(acc + 8), acc1);

    for (std::size_t j = 0; j < 16; ++j)
        out[j] =
          clip_to_hidden_q(round_div_symmetric_i64(std::int64_t(acc[j]), weightScaleHidden),
                           hiddenQuantizedOne);
}

void dense_layer_clip_q_avx2_out32(const std::uint8_t*              input,
                                   std::size_t                      inputDim,
                                   const std::vector<std::int8_t>&  weight,
                                   const std::vector<std::int32_t>& bias,
                                   std::int32_t                     weightScaleHidden,
                                   std::int32_t                     hiddenQuantizedOne,
                                   std::uint8_t*                    out) {
    __m256i acc0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias.data()));
    __m256i acc1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias.data() + 8));
    __m256i acc2 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias.data() + 16));
    __m256i acc3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bias.data() + 24));

    for (std::size_t i = 0; i < inputDim; ++i)
    {
        const std::uint8_t in = input[i];
        if (!in)
            continue;

        const __m256i vin16 = _mm256_set1_epi16(static_cast<std::int16_t>(in));
        const auto*   row   = weight.data() + i * 32;

        const __m128i w8a    = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row));
        const __m256i w16a   = _mm256_cvtepi8_epi16(w8a);
        const __m256i prod16a = _mm256_mullo_epi16(vin16, w16a);
        acc0                  = _mm256_add_epi32(acc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod16a)));
        acc1 = _mm256_add_epi32(acc1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod16a, 1)));

        const __m128i w8b    = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + 16));
        const __m256i w16b   = _mm256_cvtepi8_epi16(w8b);
        const __m256i prod16b = _mm256_mullo_epi16(vin16, w16b);
        acc2                  = _mm256_add_epi32(acc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod16b)));
        acc3 = _mm256_add_epi32(acc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod16b, 1)));
    }

    alignas(32) std::int32_t acc[32];
    _mm256_store_si256(reinterpret_cast<__m256i*>(acc), acc0);
    _mm256_store_si256(reinterpret_cast<__m256i*>(acc + 8), acc1);
    _mm256_store_si256(reinterpret_cast<__m256i*>(acc + 16), acc2);
    _mm256_store_si256(reinterpret_cast<__m256i*>(acc + 24), acc3);

    for (std::size_t j = 0; j < 32; ++j)
        out[j] =
          clip_to_hidden_q(round_div_symmetric_i64(std::int64_t(acc[j]), weightScaleHidden),
                           hiddenQuantizedOne);
}
#endif

bool parse_positive_int_after_colon(std::string_view s, std::size_t colonPos, int& out) {
    std::size_t i = colonPos + 1;
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i])))
        ++i;
    if (i >= s.size())
        return false;

    bool negative = false;
    if (s[i] == '+' || s[i] == '-')
    {
        negative = (s[i] == '-');
        ++i;
    }
    if (i >= s.size() || !std::isdigit(static_cast<unsigned char>(s[i])))
        return false;

    long long value = 0;
    while (i < s.size() && std::isdigit(static_cast<unsigned char>(s[i])))
    {
        value = value * 10 + (s[i] - '0');
        if (value > 1'000'000)
            return false;
        ++i;
    }
    if (negative)
        value = -value;
    if (value <= 0)
        return false;
    out = static_cast<int>(value);
    return true;
}

bool try_parse_engine_value_output_scale_from_metadata(std::string_view metadataJson, int& outScale) {
    constexpr std::string_view key = "\"engine_value_output_scale\"";
    const std::size_t          keyPos = metadataJson.find(key);
    if (keyPos == std::string_view::npos)
        return false;

    const std::size_t colonPos = metadataJson.find(':', keyPos + key.size());
    if (colonPos == std::string_view::npos)
        return false;

    int parsed = 0;
    if (!parse_positive_int_after_colon(metadataJson, colonPos, parsed))
        return false;
    outScale = parsed;
    return true;
}

struct Header;

bool load_engine_value_output_scale_from_metadata(const std::vector<std::uint8_t>& fileBytes,
                                                  const Header&                    h,
                                                  int&                             outScale,
                                                  std::string&                     err);

struct Header {
    std::array<char, 8> magic{};
    std::uint32_t       version = 0, headerSize = 0, flags = 0, featureVariantId = 0,
                  bucketSchemeId = 0, outputUnitType = 0, inputSize = 0, bucketCount = 0,
                  hidden1Total = 0, hidden1Psqt = 0, hidden1Positional = 0, psqtH2 = 0,
                  positionalH2 = 0, psqtH3 = 0, positionalH3 = 0, outputs = 0;
    std::int32_t  mixPsqt = 0, mixPositional = 0, mixDenominator = 0;
    float         labelScalePsqt = 0.0f, labelScalePositional = 0.0f;
    float         ftQuantizedOne = 0.0f, hiddenQuantizedOne = 0.0f, weightScaleHidden = 0.0f,
          weightScaleOut = 0.0f, nnue2score = 0.0f;
    std::uint64_t tensorPayloadBytes = 0, metadataJsonBytes = 0;
    std::array<std::uint8_t, 32> tensorPayloadSha256{};
    std::array<std::uint8_t, 32> metadataJsonSha256{};
    std::uint32_t manifestFormatVersion = 0, npzSchemaVersion = 0;
};

bool load_engine_value_output_scale_from_metadata(const std::vector<std::uint8_t>& fileBytes,
                                                  const Header&                    h,
                                                  int&                             outScale,
                                                  std::string&                     err) {
    outScale = kDefaultEngineValueOutputScale;
    if ((h.flags & 1u) == 0 || h.metadataJsonBytes == 0)
        return true;

    const std::uint64_t metaOffset = std::uint64_t(h.headerSize) + h.tensorPayloadBytes;
    const std::uint64_t metaEnd    = metaOffset + h.metadataJsonBytes;
    if (metaEnd > fileBytes.size())
    {
        err = "Truncated metadata JSON blob";
        return false;
    }

    const auto* data = reinterpret_cast<const char*>(fileBytes.data() + std::size_t(metaOffset));
    std::string_view metadataJson(data, std::size_t(h.metadataJsonBytes));

    // Metadata parsing is best-effort for backward compatibility. If the key is absent, fall back to legacy scale=1.
    int parsedScale = 0;
    if (try_parse_engine_value_output_scale_from_metadata(metadataJson, parsedScale))
        outScale = parsedScale;
    return true;
}

class ByteReader {
   public:
    ByteReader(const std::uint8_t* begin, const std::uint8_t* end) : p_(begin), end_(end) {}
    std::size_t remaining() const { return static_cast<std::size_t>(end_ - p_); }

    bool read_bytes(void* out, std::size_t n) {
        if (remaining() < n)
            return false;
        std::memcpy(out, p_, n);
        p_ += n;
        return true;
    }
    bool read_u32(std::uint32_t& out) {
        if (remaining() < 4)
            return false;
        out = (std::uint32_t(p_[0]) << 0) | (std::uint32_t(p_[1]) << 8) | (std::uint32_t(p_[2]) << 16)
            | (std::uint32_t(p_[3]) << 24);
        p_ += 4;
        return true;
    }
    bool read_i32(std::int32_t& out) {
        std::uint32_t u{};
        if (!read_u32(u))
            return false;
        out = static_cast<std::int32_t>(u);
        return true;
    }
    bool read_u64(std::uint64_t& out) {
        if (remaining() < 8)
            return false;
        out = 0;
        for (int i = 0; i < 8; ++i)
            out |= std::uint64_t(p_[i]) << (8 * i);
        p_ += 8;
        return true;
    }
    bool read_f32(float& out) {
        std::uint32_t bits{};
        if (!read_u32(bits))
            return false;
        std::memcpy(&out, &bits, sizeof(float));
        return true;
    }
    bool read_i8(std::int8_t& out) {
        if (remaining() < 1)
            return false;
        out = static_cast<std::int8_t>(*p_++);
        return true;
    }
    bool read_i16(std::int16_t& out) {
        if (remaining() < 2)
            return false;
        const std::uint16_t u = (std::uint16_t(p_[0]) << 0) | (std::uint16_t(p_[1]) << 8);
        out                  = static_cast<std::int16_t>(u);
        p_ += 2;
        return true;
    }

   private:
    const std::uint8_t* p_;
    const std::uint8_t* end_;
};

bool parse_header(const std::vector<std::uint8_t>& fileBytes, Header& h, std::string& err) {
    if (fileBytes.size() < kNnuexHeaderSize)
    {
        err = "File too small for nnuex header";
        return false;
    }

    ByteReader rd(fileBytes.data(), fileBytes.data() + kNnuexHeaderSize);
    if (!rd.read_bytes(h.magic.data(), h.magic.size())
        || !rd.read_u32(h.version) || !rd.read_u32(h.headerSize) || !rd.read_u32(h.flags)
        || !rd.read_u32(h.featureVariantId) || !rd.read_u32(h.bucketSchemeId)
        || !rd.read_u32(h.outputUnitType) || !rd.read_u32(h.inputSize) || !rd.read_u32(h.bucketCount)
        || !rd.read_u32(h.hidden1Total) || !rd.read_u32(h.hidden1Psqt) || !rd.read_u32(h.hidden1Positional)
        || !rd.read_u32(h.psqtH2) || !rd.read_u32(h.positionalH2) || !rd.read_u32(h.psqtH3)
        || !rd.read_u32(h.positionalH3) || !rd.read_u32(h.outputs) || !rd.read_i32(h.mixPsqt)
        || !rd.read_i32(h.mixPositional) || !rd.read_i32(h.mixDenominator)
        || !rd.read_f32(h.labelScalePsqt) || !rd.read_f32(h.labelScalePositional)
        || !rd.read_f32(h.ftQuantizedOne) || !rd.read_f32(h.hiddenQuantizedOne)
        || !rd.read_f32(h.weightScaleHidden) || !rd.read_f32(h.weightScaleOut) || !rd.read_f32(h.nnue2score)
        || !rd.read_u64(h.tensorPayloadBytes) || !rd.read_u64(h.metadataJsonBytes)
        || !rd.read_bytes(h.tensorPayloadSha256.data(), h.tensorPayloadSha256.size())
        || !rd.read_bytes(h.metadataJsonSha256.data(), h.metadataJsonSha256.size())
        || !rd.read_u32(h.manifestFormatVersion) || !rd.read_u32(h.npzSchemaVersion))
    {
        err = "Failed to parse nnuex header";
        return false;
    }

    const std::array<char, 8> expected = {kNnuexMagic[0], kNnuexMagic[1], kNnuexMagic[2], kNnuexMagic[3],
                                          kNnuexMagic[4], kNnuexMagic[5], kNnuexMagic[6], kNnuexMagic[7]};
    if (h.magic != expected)
    {
        err = "Invalid nnuex magic";
        return false;
    }
    if (h.version != kNnuexVersion || h.headerSize != kNnuexHeaderSize)
    {
        err = "Unsupported nnuex version/header size";
        return false;
    }

    if (h.featureVariantId != kExpectedFeatureVariantId || h.bucketSchemeId != kExpectedBucketSchemeId
        || h.outputUnitType != kExpectedOutputUnitType)
    {
        err = "Unsupported nnuex feature/bucket/unit identifiers";
        return false;
    }
    if (h.inputSize != kExpectedInputSize || h.bucketCount != kExpectedBucketCount
        || h.hidden1Total != kExpectedH1Total || h.hidden1Psqt != kExpectedH1Psqt
        || h.hidden1Positional != kExpectedH1Pos || h.psqtH2 != kExpectedPsqtH2
        || h.positionalH2 != kExpectedPosH2 || h.psqtH3 != kExpectedPsqtH3
        || h.positionalH3 != kExpectedPosH3 || h.outputs != kExpectedOutputs)
    {
        err = "NNUEX dimensions do not match frozen custom contract";
        return false;
    }
    if (h.mixDenominator == 0 || h.ftQuantizedOne == 0.0f || h.hiddenQuantizedOne == 0.0f
        || h.weightScaleHidden == 0.0f || h.weightScaleOut == 0.0f || h.nnue2score == 0.0f)
    {
        err = "NNUEX header contains invalid scaling constants";
        return false;
    }

    const std::uint64_t totalNeed = std::uint64_t(h.headerSize) + h.tensorPayloadBytes + h.metadataJsonBytes;
    if (totalNeed > fileBytes.size())
    {
        err = "Truncated nnuex file";
        return false;
    }
    return true;
}

inline std::uint32_t rotr32(std::uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

std::array<std::uint8_t, 32> sha256_bytes(const std::uint8_t* data, std::size_t size) {
    static constexpr std::uint32_t K[64] = {
      0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
      0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
      0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
      0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
      0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
      0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
      0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
      0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
      0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
      0xc67178f2u};

    std::uint32_t h0 = 0x6a09e667u, h1 = 0xbb67ae85u, h2 = 0x3c6ef372u, h3 = 0xa54ff53au;
    std::uint32_t h4 = 0x510e527fu, h5 = 0x9b05688cu, h6 = 0x1f83d9abu, h7 = 0x5be0cd19u;

    auto process_block = [&](const std::uint8_t* block) {
        std::uint32_t w[64]{};
        for (int i = 0; i < 16; ++i)
        {
            const int j = i * 4;
            w[i] = (std::uint32_t(block[j]) << 24) | (std::uint32_t(block[j + 1]) << 16)
                 | (std::uint32_t(block[j + 2]) << 8) | (std::uint32_t(block[j + 3]) << 0);
        }
        for (int i = 16; i < 64; ++i)
        {
            const std::uint32_t s0 = rotr32(w[i - 15], 7) ^ rotr32(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = rotr32(w[i - 2], 17) ^ rotr32(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i]                   = w[i - 16] + s0 + w[i - 7] + s1;
        }

        std::uint32_t a = h0, b = h1, c = h2, d = h3, e = h4, f = h5, g = h6, h = h7;
        for (int i = 0; i < 64; ++i)
        {
            const std::uint32_t S1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
            const std::uint32_t ch = (e & f) ^ (~e & g);
            const std::uint32_t t1 = h + S1 + ch + K[i] + w[i];
            const std::uint32_t S0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
            const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t t2  = S0 + maj;

            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }

        h0 += a;
        h1 += b;
        h2 += c;
        h3 += d;
        h4 += e;
        h5 += f;
        h6 += g;
        h7 += h;
    };

    const std::size_t fullBlocks = size / 64;
    for (std::size_t i = 0; i < fullBlocks; ++i)
        process_block(data + i * 64);

    std::array<std::uint8_t, 128> tail{};
    const std::size_t             rem = size % 64;
    if (rem)
        std::memcpy(tail.data(), data + fullBlocks * 64, rem);
    tail[rem] = 0x80;

    const std::uint64_t bitLen = std::uint64_t(size) * 8u;
    const std::size_t   padded = (rem + 1 + 8 <= 64) ? 64 : 128;
    for (int i = 0; i < 8; ++i)
        tail[padded - 1 - i] = static_cast<std::uint8_t>(bitLen >> (i * 8));

    process_block(tail.data());
    if (padded == 128)
        process_block(tail.data() + 64);

    std::array<std::uint8_t, 32> out{};
    const std::uint32_t          words[8] = {h0, h1, h2, h3, h4, h5, h6, h7};
    for (int i = 0; i < 8; ++i)
    {
        out[i * 4 + 0] = static_cast<std::uint8_t>(words[i] >> 24);
        out[i * 4 + 1] = static_cast<std::uint8_t>(words[i] >> 16);
        out[i * 4 + 2] = static_cast<std::uint8_t>(words[i] >> 8);
        out[i * 4 + 3] = static_cast<std::uint8_t>(words[i] >> 0);
    }
    return out;
}

std::string hex_bytes(const std::uint8_t* data, std::size_t size) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string           s(size * 2, '\0');
    for (std::size_t i = 0; i < size; ++i)
    {
        s[2 * i]     = hex[(data[i] >> 4) & 0x0F];
        s[2 * i + 1] = hex[data[i] & 0x0F];
    }
    return s;
}

template<std::size_t N>
std::string hex_bytes(const std::array<std::uint8_t, N>& a) {
    return hex_bytes(a.data(), a.size());
}

bool verify_file_checksums(const std::vector<std::uint8_t>& fileBytes, const Header& h, std::string& err) {
    const std::size_t payloadOffset = std::size_t(h.headerSize);
    const std::size_t payloadBytes  = std::size_t(h.tensorPayloadBytes);
    const std::size_t metaOffset    = payloadOffset + payloadBytes;
    const std::size_t metaBytes     = std::size_t(h.metadataJsonBytes);

    const auto payloadDigest = sha256_bytes(fileBytes.data() + payloadOffset, payloadBytes);
    if (payloadDigest != h.tensorPayloadSha256)
    {
        err = "Tensor payload SHA256 mismatch (expected " + hex_bytes(h.tensorPayloadSha256) + ", got "
            + hex_bytes(payloadDigest) + ")";
        return false;
    }

    if (metaBytes > 0)
    {
        const auto metadataDigest = sha256_bytes(fileBytes.data() + metaOffset, metaBytes);
        if (metadataDigest != h.metadataJsonSha256)
        {
            err = "Metadata JSON SHA256 mismatch (expected " + hex_bytes(h.metadataJsonSha256)
                + ", got " + hex_bytes(metadataDigest) + ")";
            return false;
        }
    }

    return true;
}

std::vector<std::uint8_t> read_file_bytes(const std::string& path, std::string& err) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        err = "Failed to open file: " + path;
        return {};
    }
    in.seekg(0, std::ios::end);
    const auto endPos = in.tellg();
    if (endPos < 0)
    {
        err = "Failed to get file size";
        return {};
    }
    const std::size_t size = static_cast<std::size_t>(endPos);
    in.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(size);
    if (size && !in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(size)))
    {
        err = "Failed to read file";
        return {};
    }
    return bytes;
}

template<typename IntT>
bool read_scaled_array(ByteReader& rd, std::size_t count, float scale, std::vector<float>& out) {
    if (scale == 0.0f)
        return false;
    out.resize(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        IntT v{};
        if constexpr (std::is_same_v<IntT, std::int8_t>)
        {
            if (!rd.read_i8(v))
                return false;
        }
        else if constexpr (std::is_same_v<IntT, std::int16_t>)
        {
            if (!rd.read_i16(v))
                return false;
        }
        else
        {
            static_assert(std::is_same_v<IntT, std::int32_t>);
            std::int32_t t{};
            if (!rd.read_i32(t))
                return false;
            v = t;
        }
        out[i] = static_cast<float>(v) / scale;
    }
    return true;
}

[[maybe_unused]] bool read_scaled_scalar_i32(ByteReader& rd, float scale, float& out) {
    if (scale == 0.0f)
        return false;
    std::int32_t v{};
    if (!rd.read_i32(v))
        return false;
    out = static_cast<float>(v) / scale;
    return true;
}

template<typename IntT>
bool read_raw_array(ByteReader& rd, std::size_t count, std::vector<IntT>& out) {
    out.resize(count);
    for (std::size_t i = 0; i < count; ++i)
    {
        if constexpr (std::is_same_v<IntT, std::int8_t>)
        {
            std::int8_t v{};
            if (!rd.read_i8(v))
                return false;
            out[i] = v;
        }
        else if constexpr (std::is_same_v<IntT, std::int16_t>)
        {
            std::int16_t v{};
            if (!rd.read_i16(v))
                return false;
            out[i] = v;
        }
        else
        {
            static_assert(std::is_same_v<IntT, std::int32_t>);
            std::int32_t v{};
            if (!rd.read_i32(v))
                return false;
            out[i] = v;
        }
    }
    return true;
}

bool read_raw_scalar_i32(ByteReader& rd, std::int32_t& out) { return rd.read_i32(out); }

struct BucketLayers {
    std::vector<std::int8_t>  h2Weight, h3Weight;
    std::vector<std::int32_t> h2Bias, h3Bias;
};

struct EncodedFen {
    std::array<float, kExpectedInputSize> features{};
    std::array<std::uint16_t, kMaxActiveFeatures> activeFeatureIndices{};
    std::uint16_t                                 activeFeatureCount = 0;
    int                                           pieceCount         = 0;
    int                                           stmBlack           = 0;
    int                                           bucket8            = 0;
    int                                           bucket16           = 0;
};

int piece_index_no_pawn(char p) {
    switch (p)
    {
    case 'r':
        return 0;
    case 'n':
        return 1;
    case 'b':
        return 2;
    case 'q':
        return 3;
    case 'k':
        return 4;
    default:
        return -1;
    }
}

int piece_index_all(char p) {
    switch (p)
    {
    case 'p':
        return 0;
    case 'r':
        return 1;
    case 'n':
        return 2;
    case 'b':
        return 3;
    case 'q':
        return 4;
    case 'k':
        return 5;
    default:
        return -1;
    }
}

bool add_active_feature(EncodedFen& enc, int featureIndex);
[[maybe_unused]] bool encode_fen_v2(std::string_view fen, EncodedFen& out, std::string& err);
bool encode_position_v2(const Position& pos, EncodedFen& out);
bool load_payload_quantized(const std::vector<std::uint8_t>& fileBytes, Network::Impl& impl, std::string& err);
std::optional<Evaluation> evaluate_encoded(const Network::Impl& impl, const EncodedFen& enc);
bool build_incremental_state_from_encoded(const Network::Impl&  impl,
                                          const EncodedFen&     enc,
                                          Key                   key,
                                          IncrementalState&     out);
bool advance_incremental_state_impl(const Network::Impl&   impl,
                                    const Position&        posAfterMove,
                                    Move                   move,
                                    const DirtyPiece&      dirtyPiece,
                                    const IncrementalState& prev,
                                    IncrementalState&      next);
bool advance_incremental_state_null_impl(const Network::Impl&   impl,
                                         const Position&        posAfterNull,
                                         const IncrementalState& prev,
                                         IncrementalState&      next);
std::optional<Evaluation> evaluate_incremental_state_impl(const Network::Impl& impl,
                                                          const IncrementalState& state);

}  // namespace

struct Network::Impl {
    Header header{};
    bool   metadataJsonPresent = false;
    int    engineValueOutputScale = kDefaultEngineValueOutputScale;

    std::vector<std::int16_t> hidden1Weight;
    std::vector<std::int16_t> hidden1Bias;
    std::array<BucketLayers, kExpectedBucketCount> psqtBuckets{};
    std::array<BucketLayers, kExpectedBucketCount> positionalBuckets{};
    std::vector<std::int8_t> psqtOutputWeight;
    std::int32_t             psqtOutputBias = 0;
    std::vector<std::int8_t> positionalOutputWeight;
    std::int32_t             positionalOutputBias = 0;
};

Network::Network(const Network& other) :
    initialized_(other.initialized_),
    requestedPath_(other.requestedPath_),
    loadedPath_(other.loadedPath_),
    error_(other.error_),
    impl_(other.impl_ ? new Impl(*other.impl_) : nullptr) {}

Network::Network(Network&& other) noexcept :
    initialized_(other.initialized_),
    requestedPath_(std::move(other.requestedPath_)),
    loadedPath_(std::move(other.loadedPath_)),
    error_(std::move(other.error_)),
    impl_(std::exchange(other.impl_, nullptr)) {
    other.initialized_ = false;
}

Network& Network::operator=(const Network& other) {
    if (this == &other)
        return *this;
    initialized_   = other.initialized_;
    requestedPath_ = other.requestedPath_;
    loadedPath_    = other.loadedPath_;
    error_         = other.error_;
    delete impl_;
    impl_ = other.impl_ ? new Impl(*other.impl_) : nullptr;
    return *this;
}

Network& Network::operator=(Network&& other) noexcept {
    if (this == &other)
        return *this;
    initialized_   = other.initialized_;
    requestedPath_ = std::move(other.requestedPath_);
    loadedPath_    = std::move(other.loadedPath_);
    error_         = std::move(other.error_);
    delete impl_;
    impl_ = std::exchange(other.impl_, nullptr);
    other.initialized_ = false;
    return *this;
}

Network::~Network() { delete impl_; }

std::string Network::resolve_evalfile_path(const std::string& rootDirectory,
                                           const std::string& requestedPath) const {
    namespace fs = std::filesystem;

    if (requestedPath.empty())
        return {};

    const fs::path requested(requestedPath);
    std::vector<fs::path> candidates;

    if (requested.is_absolute())
        candidates.push_back(requested);
    else
    {
        candidates.push_back(requested);

        if (!rootDirectory.empty())
        {
            const fs::path root(rootDirectory);
            candidates.push_back(root / requested);
            candidates.push_back(root / ".." / requested);
            candidates.push_back(root / ".." / ".." / requested);
            // Convenience for this workspace layout when EvalFile is just the basename.
            candidates.push_back(root / ".." / ".." / "export" / requested);
        }
        candidates.push_back(fs::path("export") / requested);
        candidates.push_back(fs::path("..") / "export" / requested);
        candidates.push_back(fs::path("..") / ".." / "export" / requested);
    }

    for (const auto& c : candidates)
    {
        std::error_code ec;
        if (fs::exists(c, ec) && !ec)
            return c.lexically_normal().string();
    }

    return requestedPath;
}

bool Network::load_from_file(const std::string& path, std::string& err) {
    auto bytes = read_file_bytes(path, err);
    if (bytes.empty() && !err.empty())
        return false;

    auto newImpl = std::make_unique<Impl>();
    if (!parse_header(bytes, newImpl->header, err))
        return false;
    if (!verify_file_checksums(bytes, newImpl->header, err))
        return false;
    if (!load_engine_value_output_scale_from_metadata(bytes, newImpl->header, newImpl->engineValueOutputScale, err))
        return false;
    if (!load_payload_quantized(bytes, *newImpl, err))
        return false;

    newImpl->metadataJsonPresent =
      (newImpl->header.flags & 1u) != 0 && newImpl->header.metadataJsonBytes > 0;

    delete impl_;
    impl_        = newImpl.release();
    initialized_ = true;
    return true;
}

void Network::load(const std::string& rootDirectory, std::string evalfilePath) {
    initialized_   = false;
    requestedPath_ = std::move(evalfilePath);
    loadedPath_.clear();
    error_.clear();
    delete impl_;
    impl_ = nullptr;

    if (requestedPath_.empty())
    {
        error_ = "EvalFile is empty (expected .nnuex path)";
        return;
    }

    const std::string resolved = resolve_evalfile_path(rootDirectory, requestedPath_);
    std::string       err;
    if (!load_from_file(resolved, err))
    {
        error_ = "Failed to load custom nnuex '" + requestedPath_ + "'";
        if (resolved != requestedPath_)
            error_ += " (resolved to '" + resolved + "')";
        if (!err.empty())
            error_ += ": " + err;
        return;
    }

    loadedPath_ = resolved;
}

void Network::verify(std::string evalfilePath,
                     const std::function<void(std::string_view)>& out) const {
    if (!out)
        return;

    std::ostringstream ss;
    ss << "Custom NNUEX ";
    if (!initialized_ || impl_ == nullptr)
    {
        ss << "not loaded";
        if (!evalfilePath.empty())
            ss << " (EvalFile='" << evalfilePath << "')";
        if (!error_.empty())
            ss << ". " << error_;
        out(ss.str());
        return;
    }

    const auto& h = impl_->header;
    ss << "loaded: " << loadedPath_ << " | v" << h.version << " | input=" << h.inputSize
       << " buckets=" << h.bucketCount << " h1=" << h.hidden1Total << " (" << h.hidden1Psqt << "+"
       << h.hidden1Positional << ")"
       << " | outputs=" << h.outputs << " | label_scales=" << h.labelScalePsqt << ","
       << h.labelScalePositional << " | engine_output_scale=" << impl_->engineValueOutputScale;
    if (impl_->metadataJsonPresent)
        ss << " | metadata-json";
    ss << " | checksum=ok(payload";
    if (impl_->metadataJsonPresent)
        ss << "+metadata";
    ss << ")";
    out(ss.str());
}

std::optional<Evaluation> Network::evaluate(const Position& pos) const {
    if (!initialized_ || impl_ == nullptr)
        return std::nullopt;

    EncodedFen enc;
    if (!encode_position_v2(pos, enc))
        return std::nullopt;
    return evaluate_encoded(*impl_, enc);
}

std::optional<Evaluation> Network::evaluate(const IncrementalState& state) const {
    if (!initialized_ || impl_ == nullptr || !state.valid)
        return std::nullopt;
    return evaluate_incremental_state_impl(*impl_, state);
}

bool Network::build_incremental_state(const Position& pos, IncrementalState& out) const {
    out = IncrementalState{};
    if (!initialized_ || impl_ == nullptr)
        return false;

    EncodedFen enc;
    if (!encode_position_v2(pos, enc))
        return false;
    return build_incremental_state_from_encoded(*impl_, enc, pos.key(), out);
}

bool Network::advance_incremental_state(const Position&        posAfterMove,
                                        Move                   move,
                                        const DirtyPiece&      dirtyPiece,
                                        const IncrementalState& prev,
                                        IncrementalState&      next) const {
    next = IncrementalState{};
    if (!initialized_ || impl_ == nullptr || !prev.valid)
        return false;
    return advance_incremental_state_impl(*impl_, posAfterMove, move, dirtyPiece, prev, next);
}

bool Network::advance_incremental_state_null(const Position&        posAfterNull,
                                             const IncrementalState& prev,
                                             IncrementalState&      next) const {
    next = IncrementalState{};
    if (!initialized_ || impl_ == nullptr || !prev.valid)
        return false;
    return advance_incremental_state_null_impl(*impl_, posAfterNull, prev, next);
}

namespace {

[[maybe_unused]] bool encode_fen_v2(std::string_view fen, EncodedFen& out, std::string& err) {
    out = EncodedFen{};

    const auto space1 = fen.find(' ');
    if (space1 == std::string_view::npos || space1 == 0)
    {
        err = "Invalid FEN: missing board/side fields";
        return false;
    }
    const auto sideStart = space1 + 1;
    const auto space2    = fen.find(' ', sideStart);
    const auto board     = fen.substr(0, space1);
    const auto side      = (space2 == std::string_view::npos) ? fen.substr(sideStart)
                                                              : fen.substr(sideStart, space2 - sideStart);
    if (side.empty() || (side[0] != 'w' && side[0] != 'b'))
    {
        err = "Invalid FEN: side-to-move must be w/b";
        return false;
    }

    std::size_t offset = 0;
    std::size_t cursor = 0;
    int         rankIdx = 0;

    while (cursor <= board.size())
    {
        const auto slash = board.find('/', cursor);
        const auto end   = (slash == std::string_view::npos) ? board.size() : slash;
        const auto rank  = board.substr(cursor, end - cursor);

        if (rankIdx >= 8)
        {
            err = "Invalid FEN board: too many ranks";
            return false;
        }

        const int  actualRank    = 8 - rankIdx;
        const bool isEdgeRank    = actualRank == 1 || actualRank == 8;
        const int  nTypes        = isEdgeRank ? 5 : 6;
        const int  featuresPerSq = nTypes * 2;
        int        fileIdx       = 0;

        for (char ch : rank)
        {
            const auto uch = static_cast<unsigned char>(ch);
            if (std::isdigit(uch))
            {
                const int skip = ch - '0';
                if (skip < 1 || skip > 8 || fileIdx + skip > 8)
                {
                    err = "Invalid FEN board digit span";
                    return false;
                }
                fileIdx += skip;
                offset += std::size_t(skip * featuresPerSq);
                continue;
            }

            const char p = static_cast<char>(std::tolower(uch));
            if (piece_index_all(p) < 0)
            {
                err = "Invalid FEN board piece";
                return false;
            }

            const bool isBlack = std::islower(uch) != 0;
            const int  idx     = isEdgeRank ? piece_index_no_pawn(p) : piece_index_all(p);
            if (idx >= 0)
            {
                const int featureIndex = static_cast<int>(offset) + (isBlack ? nTypes : 0) + idx;
                if (!add_active_feature(out, featureIndex))
                {
                    err = "Invalid FEN feature encoding";
                    return false;
                }
            }

            ++fileIdx;
            ++out.pieceCount;
            offset += std::size_t(featuresPerSq);
            if (fileIdx > 8)
            {
                err = "Invalid FEN board rank width";
                return false;
            }
        }

        if (fileIdx != 8)
        {
            err = "Invalid FEN board rank width";
            return false;
        }

        ++rankIdx;
        if (slash == std::string_view::npos)
            break;
        cursor = slash + 1;
    }

    if (rankIdx != 8 || offset != 736)
    {
        err = "Invalid FEN board shape";
        return false;
    }

    out.stmBlack = side[0] == 'b' ? 1 : 0;
    if (out.stmBlack && !add_active_feature(out, 736))
    {
        err = "Invalid FEN stm feature encoding";
        return false;
    }
    out.bucket8       = std::clamp((std::max(out.pieceCount, 1) - 1) / 4, 0, 7);
    out.bucket16      = out.bucket8 * 2 + out.stmBlack;
    return true;
}

bool load_payload_quantized(const std::vector<std::uint8_t>& fileBytes, Network::Impl& impl, std::string& err) {
    const Header& h = impl.header;
    const auto*   p = fileBytes.data() + h.headerSize;
    ByteReader    rd(p, p + std::size_t(h.tensorPayloadBytes));

    if (!read_raw_array<std::int16_t>(rd, std::size_t(kExpectedInputSize) * kExpectedH1Total, impl.hidden1Weight)
        || !read_raw_array<std::int16_t>(rd, kExpectedH1Total, impl.hidden1Bias))
    {
        err = "Failed reading hidden_1 tensors";
        return false;
    }

    for (auto& b : impl.psqtBuckets)
    {
        if (!read_raw_array<std::int8_t>(rd, std::size_t(kExpectedH1Psqt) * kExpectedPsqtH2, b.h2Weight)
            || !read_raw_array<std::int32_t>(rd, kExpectedPsqtH2, b.h2Bias)
            || !read_raw_array<std::int8_t>(rd, std::size_t(kExpectedPsqtH2) * kExpectedPsqtH3, b.h3Weight)
            || !read_raw_array<std::int32_t>(rd, kExpectedPsqtH3, b.h3Bias))
        {
            err = "Failed reading PSQT bucket tensors";
            return false;
        }
    }

    for (auto& b : impl.positionalBuckets)
    {
        if (!read_raw_array<std::int8_t>(rd, std::size_t(kExpectedH1Pos) * kExpectedPosH2, b.h2Weight)
            || !read_raw_array<std::int32_t>(rd, kExpectedPosH2, b.h2Bias)
            || !read_raw_array<std::int8_t>(rd, std::size_t(kExpectedPosH2) * kExpectedPosH3, b.h3Weight)
            || !read_raw_array<std::int32_t>(rd, kExpectedPosH3, b.h3Bias))
        {
            err = "Failed reading positional bucket tensors";
            return false;
        }
    }

    if (!read_raw_array<std::int8_t>(rd, kExpectedPsqtH3, impl.psqtOutputWeight)
        || !read_raw_scalar_i32(rd, impl.psqtOutputBias)
        || !read_raw_array<std::int8_t>(rd, kExpectedPosH3, impl.positionalOutputWeight)
        || !read_raw_scalar_i32(rd, impl.positionalOutputBias))
    {
        err = "Failed reading output tensors";
        return false;
    }

    if (rd.remaining() != 0)
    {
        err = "Tensor payload trailing bytes remain after parse";
        return false;
    }
    return true;
}

void fill_bucket_fields(int pieceCount, int stmBlack, int& bucket8, int& bucket16) {
    bucket8  = std::clamp((std::max(pieceCount, 1) - 1) / 4, 0, 7);
    bucket16 = bucket8 * 2 + stmBlack;
}

bool add_active_feature(EncodedFen& enc, int featureIndex) {
    if (featureIndex < 0 || featureIndex >= int(kExpectedInputSize))
        return false;

    enc.features[std::size_t(featureIndex)] = 1.0f;
    if (enc.activeFeatureCount >= enc.activeFeatureIndices.size())
        return false;

    enc.activeFeatureIndices[enc.activeFeatureCount++] = static_cast<std::uint16_t>(featureIndex);
    return true;
}

struct SquareFeatureGeom {
    int base   = 0;
    int nTypes = 0;
};

const std::array<SquareFeatureGeom, SQUARE_NB>& square_feature_geoms() {
    static const std::array<SquareFeatureGeom, SQUARE_NB> geoms = [] {
        std::array<SquareFeatureGeom, SQUARE_NB> out{};
        int                                      offset = 0;
        for (int rank = 8; rank >= 1; --rank)
        {
            const bool edge         = rank == 1 || rank == 8;
            const int  nTypes       = edge ? 5 : 6;
            const int  featuresPerSq = nTypes * 2;
            for (int file = 0; file < 8; ++file)
            {
                const Square sq = make_square(File(file), Rank(rank - 1));
                out[std::size_t(sq)] = {offset + file * featuresPerSq, nTypes};
            }
            offset += 8 * featuresPerSq;
        }
        return out;
    }();
    return geoms;
}

int piece_index_no_pawn(PieceType pt) {
    switch (pt)
    {
    case ROOK:
        return 0;
    case KNIGHT:
        return 1;
    case BISHOP:
        return 2;
    case QUEEN:
        return 3;
    case KING:
        return 4;
    default:
        return -1;
    }
}

int piece_index_all(PieceType pt) {
    switch (pt)
    {
    case PAWN:
        return 0;
    case ROOK:
        return 1;
    case KNIGHT:
        return 2;
    case BISHOP:
        return 3;
    case QUEEN:
        return 4;
    case KING:
        return 5;
    default:
        return -1;
    }
}

bool feature_index_for_piece_square(Piece pc, Square sq, int& featureIndex) {
    if (pc == NO_PIECE || sq == SQ_NONE)
        return false;
    if (!is_ok(sq))
        return false;

    const auto& geom = square_feature_geoms()[std::size_t(sq)];
    const int   idx  = geom.nTypes == 5 ? piece_index_no_pawn(type_of(pc)) : piece_index_all(type_of(pc));
    if (idx < 0)
        return false;

    const int colorOffset = color_of(pc) == BLACK ? geom.nTypes : 0;
    featureIndex          = geom.base + colorOffset + idx;
    return featureIndex >= 0 && featureIndex < 736;
}

bool encode_position_v2(const Position& pos, EncodedFen& out) {
    out = EncodedFen{};

    const auto& board = pos.piece_array();
    for (int rank = 8; rank >= 1; --rank)
    {
        for (int file = 0; file < 8; ++file)
        {
            const Square sq = make_square(File(file), Rank(rank - 1));
            const Piece  pc = board[std::size_t(sq)];
            if (pc == NO_PIECE)
                continue;

            ++out.pieceCount;

            int featureIndex = -1;
            if (feature_index_for_piece_square(pc, sq, featureIndex))
            {
                if (!add_active_feature(out, featureIndex))
                    return false;
                continue;
            }

            // Match the FEN v2 encoder contract: pawns on edge ranks are ignored in features
            // but still counted in piece_count/bucketing.
            if (type_of(pc) == PAWN && (rank == 1 || rank == 8))
                continue;

            return false;
        }
    }

    out.stmBlack = pos.side_to_move() == BLACK ? 1 : 0;
    if (out.stmBlack && !add_active_feature(out, 736))
        return false;
    fill_bucket_fields(out.pieceCount, out.stmBlack, out.bucket8, out.bucket16);
    return true;
}

void apply_h1_feature_delta(const Network::Impl& impl,
                            int                  featureIndex,
                            int                  sign,
                            std::array<std::int32_t, kExpectedH1Total>& h1Pre) {
    const auto* row = impl.hidden1Weight.data() + std::size_t(featureIndex) * kExpectedH1Total;
    for (std::size_t j = 0; j < kExpectedH1Total; ++j)
        h1Pre[j] += sign * std::int32_t(row[j]);
}

void refresh_h1_clip(const Network::Impl& impl,
                     const std::array<std::int32_t, kExpectedH1Total>& h1Pre,
                     std::array<std::uint8_t, kExpectedH1Total>&       h1Clip) {
    RuntimeMetrics* metrics = gRuntimeMetricsSink;
    if (metrics)
        ++metrics->h1ClipCalls;
    ScopedRuntimeMetricTimer metricTimer(metrics ? &metrics->h1ClipNs : nullptr);

    const std::int32_t ftQuantizedOne     = static_cast<std::int32_t>(std::llround(impl.header.ftQuantizedOne));
    const std::int32_t hiddenQuantizedOne = static_cast<std::int32_t>(std::llround(impl.header.hiddenQuantizedOne));
    for (std::size_t j = 0; j < kExpectedH1Total; ++j)
    {
        const std::int64_t scaled =
          round_div_symmetric_i64(std::int64_t(h1Pre[j]) * hiddenQuantizedOne, ftQuantizedOne);
        h1Clip[j] = clip_to_hidden_q(scaled, hiddenQuantizedOne);
    }
}

bool apply_piece_toggle(const Network::Impl& impl,
                        Piece                pc,
                        Square               sq,
                        int                  sign,
                        std::array<std::int32_t, kExpectedH1Total>& h1Pre) {
    int idx = -1;
    if (!feature_index_for_piece_square(pc, sq, idx))
        return false;
    apply_h1_feature_delta(impl, idx, sign, h1Pre);
    return true;
}

std::optional<Evaluation> evaluate_from_h1_clipped(const Network::Impl& impl,
                                                   const IncrementalState& state) {
    const auto& h = impl.header;
    if (state.bucket16 < 0 || state.bucket16 >= int(kExpectedBucketCount))
        return std::nullopt;

    const std::int32_t hiddenQuantizedOne = static_cast<std::int32_t>(std::llround(h.hiddenQuantizedOne));
    const std::int32_t weightScaleHidden   = static_cast<std::int32_t>(std::llround(h.weightScaleHidden));
    const std::int64_t outScale =
      static_cast<std::int64_t>(std::llround(double(h.nnue2score) * double(h.weightScaleOut)));

    std::array<std::uint8_t, kExpectedPsqtH2> psqtH2{};
    std::array<std::uint8_t, kExpectedPsqtH3> psqtH3{};
    std::array<std::uint8_t, kExpectedPosH2>  posH2{};
    std::array<std::uint8_t, kExpectedPosH3>  posH3{};

    const auto& psqtB = impl.psqtBuckets[std::size_t(state.bucket16)];
    const auto& posB  = impl.positionalBuckets[std::size_t(state.bucket16)];

    auto dense_layer_clip_q = [weightScaleHidden, hiddenQuantizedOne](const std::uint8_t* input,
                                                                       std::size_t inputDim,
                                                                       std::size_t outputDim,
                                                                       const std::vector<std::int8_t>& weight,
                                                                       const std::vector<std::int32_t>& bias,
                                                                       std::uint8_t* out) {
#if CUSTOM_NNUE_HAS_AVX2_INTRINSICS
        if (outputDim == 16 && bias.size() >= 16 && weight.size() >= inputDim * 16)
        {
            dense_layer_clip_q_avx2_out16(input, inputDim, weight, bias, weightScaleHidden,
                                          hiddenQuantizedOne, out);
            return;
        }

        if (outputDim == 32 && bias.size() >= 32 && weight.size() >= inputDim * 32)
        {
            dense_layer_clip_q_avx2_out32(input, inputDim, weight, bias, weightScaleHidden,
                                          hiddenQuantizedOne, out);
            return;
        }
#endif
        // Row-major-friendly accumulation: walk each input row once and update all outputs.
        std::array<std::int64_t, kExpectedPosH3> acc{};
        for (std::size_t j = 0; j < outputDim; ++j)
            acc[j] = bias[j];

        for (std::size_t i = 0; i < inputDim; ++i)
        {
            const std::uint8_t in = input[i];
            if (!in)
                continue;

            const auto* row = weight.data() + i * outputDim;
            for (std::size_t j = 0; j < outputDim; ++j)
                acc[j] += std::int64_t(in) * std::int64_t(row[j]);
        }

        for (std::size_t j = 0; j < outputDim; ++j)
            out[j] =
              clip_to_hidden_q(round_div_symmetric_i64(acc[j], weightScaleHidden), hiddenQuantizedOne);
    };

    auto dense_output_acc_q = [hiddenQuantizedOne](const std::uint8_t* input,
                                                   std::size_t inputDim,
                                                   const std::vector<std::int8_t>& weight,
                                                   std::int32_t bias) {
        std::int64_t acc = bias;
#if CUSTOM_NNUE_HAS_AVX2_INTRINSICS
        if (hiddenQuantizedOne <= 127 && inputDim == 32 && weight.size() >= 32)
        {
            const __m256i inVec =
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input));
            const __m256i wVec =
              _mm256_loadu_si256(reinterpret_cast<const __m256i*>(weight.data()));
            const __m256i pairSums16 = _mm256_maddubs_epi16(inVec, wVec);
            const __m256i laneSums32 = _mm256_madd_epi16(pairSums16, _mm256_set1_epi16(1));

            std::int32_t partial[8];
            _mm256_storeu_si256(reinterpret_cast<__m256i*>(partial), laneSums32);
            for (int i = 0; i < 8; ++i)
                acc += partial[i];
            return acc;
        }
#endif
        for (std::size_t i = 0; i < inputDim; ++i)
            acc += std::int64_t(input[i]) * std::int64_t(weight[i]);
        return acc;
    };

    RuntimeMetrics* metrics = gRuntimeMetricsSink;
    std::int64_t    psqtOutAccQ = 0;
    std::int64_t    posOutAccQ  = 0;

    if (metrics)
        ++metrics->postH1ForwardCalls;
    {
        ScopedRuntimeMetricTimer postH1Timer(metrics ? &metrics->postH1ForwardNs : nullptr);
        dense_layer_clip_q(state.h1Clip.data(), kExpectedH1Psqt, kExpectedPsqtH2, psqtB.h2Weight, psqtB.h2Bias,
                           psqtH2.data());
        dense_layer_clip_q(psqtH2.data(), kExpectedPsqtH2, kExpectedPsqtH3, psqtB.h3Weight, psqtB.h3Bias,
                           psqtH3.data());
        dense_layer_clip_q(state.h1Clip.data() + kExpectedH1Psqt, kExpectedH1Pos, kExpectedPosH2, posB.h2Weight,
                           posB.h2Bias, posH2.data());
        dense_layer_clip_q(posH2.data(), kExpectedPosH2, kExpectedPosH3, posB.h3Weight, posB.h3Bias, posH3.data());

        psqtOutAccQ = dense_output_acc_q(psqtH3.data(), kExpectedPsqtH3, impl.psqtOutputWeight, impl.psqtOutputBias);
        posOutAccQ =
          dense_output_acc_q(posH3.data(), kExpectedPosH3, impl.positionalOutputWeight, impl.positionalOutputBias);
    }

    if (metrics)
        ++metrics->outputConvertCalls;
    ScopedRuntimeMetricTimer outputConvertTimer(metrics ? &metrics->outputConvertNs : nullptr);

    const float psqtNorm = static_cast<float>(double(psqtOutAccQ) / double(outScale));
    const float posNorm  = static_cast<float>(double(posOutAccQ) / double(outScale));

    const auto psqtHeadCpRaw =
      round_half_away_from_zero(double(psqtOutAccQ) * double(h.labelScalePsqt) / double(outScale));
    const auto posHeadCpRaw =
      round_half_away_from_zero(double(posOutAccQ) * double(h.labelScalePositional) / double(outScale));

    Evaluation out;
    // Optional metadata-driven engine-facing output compression (legacy nets omit it => scale=1).
    out.psqt       = Value(psqtHeadCpRaw / impl.engineValueOutputScale);
    out.positional = Value(posHeadCpRaw / impl.engineValueOutputScale);
    out.pieceCount = state.pieceCount;
    out.bucket8    = state.bucket8;
    out.stmBlack   = state.stmBlack;
    out.bucket16   = state.bucket16;
    out.psqtNorm   = psqtNorm;
    out.posNorm    = posNorm;
    return out;
}

bool build_incremental_state_from_encoded(const Network::Impl& impl,
                                          const EncodedFen&    enc,
                                          Key                  key,
                                          IncrementalState&    out) {
    out = IncrementalState{};
    if (enc.bucket16 < 0 || enc.bucket16 >= int(kExpectedBucketCount))
        return false;

    RuntimeMetrics* metrics = gRuntimeMetricsSink;
    if (metrics)
        ++metrics->buildPreClipCalls;
    {
        ScopedRuntimeMetricTimer metricTimer(metrics ? &metrics->buildPreClipNs : nullptr);
        std::copy(impl.hidden1Bias.begin(), impl.hidden1Bias.end(), out.h1Pre.begin());
        for (std::size_t i = 0; i < enc.activeFeatureCount; ++i)
            apply_h1_feature_delta(impl, enc.activeFeatureIndices[i], +1, out.h1Pre);
    }
    refresh_h1_clip(impl, out.h1Pre, out.h1Clip);

    out.valid     = true;
    out.key       = key;
    out.pieceCount = enc.pieceCount;
    out.bucket8    = enc.bucket8;
    out.stmBlack   = enc.stmBlack;
    out.bucket16   = enc.bucket16;
    return true;
}

bool advance_incremental_state_impl(const Network::Impl&   impl,
                                    const Position&        posAfterMove,
                                    Move                   move,
                                    const DirtyPiece&      dirtyPiece,
                                    const IncrementalState& prev,
                                    IncrementalState&      next) {
    if (!prev.valid)
        return false;

    next = prev;
    next.valid = false;

    auto apply_stm_toggle = [&](int newStmBlack) {
        if (newStmBlack == next.stmBlack)
            return true;
        const int sign = newStmBlack ? +1 : -1;
        apply_h1_feature_delta(impl, 736, sign, next.h1Pre);
        next.stmBlack = newStmBlack;
        return true;
    };

    if (dirtyPiece.pc == NO_PIECE || dirtyPiece.from == SQ_NONE)
        return false;

    const bool castling = move.type_of() == CASTLING;
    const bool capture  = dirtyPiece.remove_sq != SQ_NONE && !castling;

    RuntimeMetrics* metrics = gRuntimeMetricsSink;
    if (metrics)
        ++metrics->advanceMovePreClipCalls;
    {
        ScopedRuntimeMetricTimer metricTimer(metrics ? &metrics->advanceMovePreClipNs : nullptr);
        if (!apply_piece_toggle(impl, dirtyPiece.pc, dirtyPiece.from, -1, next.h1Pre))
            return false;

        if (dirtyPiece.to != SQ_NONE)
        {
            if (!apply_piece_toggle(impl, dirtyPiece.pc, dirtyPiece.to, +1, next.h1Pre))
                return false;
        }

        if (castling)
        {
            if (dirtyPiece.remove_sq == SQ_NONE || dirtyPiece.add_sq == SQ_NONE)
                return false;
            if (!apply_piece_toggle(impl, dirtyPiece.remove_pc, dirtyPiece.remove_sq, -1, next.h1Pre))
                return false;
            if (!apply_piece_toggle(impl, dirtyPiece.add_pc, dirtyPiece.add_sq, +1, next.h1Pre))
                return false;
        }
        else
        {
            if (capture)
            {
                if (!apply_piece_toggle(impl, dirtyPiece.remove_pc, dirtyPiece.remove_sq, -1, next.h1Pre))
                    return false;
            }
            if (dirtyPiece.add_sq != SQ_NONE)
            {
                if (!apply_piece_toggle(impl, dirtyPiece.add_pc, dirtyPiece.add_sq, +1, next.h1Pre))
                    return false;
            }
        }

        if (!apply_stm_toggle(posAfterMove.side_to_move() == BLACK ? 1 : 0))
            return false;
    }

    next.pieceCount = prev.pieceCount - (capture ? 1 : 0);
    fill_bucket_fields(next.pieceCount, next.stmBlack, next.bucket8, next.bucket16);
    next.key = posAfterMove.key();
    refresh_h1_clip(impl, next.h1Pre, next.h1Clip);
    next.valid = true;
    return true;
}

bool advance_incremental_state_null_impl(const Network::Impl&   impl,
                                         const Position&        posAfterNull,
                                         const IncrementalState& prev,
                                         IncrementalState&      next) {
    if (!prev.valid)
        return false;
    next = prev;
    next.valid = false;

    RuntimeMetrics* metrics = gRuntimeMetricsSink;
    if (metrics)
        ++metrics->advanceNullPreClipCalls;
    {
        ScopedRuntimeMetricTimer metricTimer(metrics ? &metrics->advanceNullPreClipNs : nullptr);
        const int newStmBlack = posAfterNull.side_to_move() == BLACK ? 1 : 0;
        if (newStmBlack != next.stmBlack)
        {
            const int sign = newStmBlack ? +1 : -1;
            apply_h1_feature_delta(impl, 736, sign, next.h1Pre);
            next.stmBlack = newStmBlack;
        }
    }

    fill_bucket_fields(next.pieceCount, next.stmBlack, next.bucket8, next.bucket16);
    next.key = posAfterNull.key();
    refresh_h1_clip(impl, next.h1Pre, next.h1Clip);
    next.valid = true;
    return true;
}

std::optional<Evaluation> evaluate_incremental_state_impl(const Network::Impl& impl,
                                                          const IncrementalState& state) {
    if (!state.valid)
        return std::nullopt;
    return evaluate_from_h1_clipped(impl, state);
}

std::optional<Evaluation> evaluate_encoded(const Network::Impl& impl, const EncodedFen& enc) {
    IncrementalState tmp;
    if (!build_incremental_state_from_encoded(impl, enc, 0, tmp))
        return std::nullopt;
    return evaluate_from_h1_clipped(impl, tmp);
}

}  // namespace

}  // namespace Stockfish::CustomNNUE
