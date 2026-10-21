#include "jf-knn.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace jf_knn {

namespace {

constexpr char     LIB_MAGIC[8]    = { 'J', 'F', 'K', 'N', 'N', 'L', 'B', '2' };
constexpr uint32_t LIB_VERSION     = 2;
constexpr size_t   LIB_HEADER_SIZE = 32;
constexpr uint32_t LIB_MM_DIM_MAX  = 1u << 16;

std::string key_of(int slot, const llama_token * ctx, size_t n_ctx) {
    std::string k(1 + n_ctx * sizeof(llama_token), '\0');
    k[0] = (char) (uint8_t) slot;
    if (n_ctx > 0) {
        std::memcpy(&k[1], ctx, n_ctx * sizeof(llama_token));
    }
    return k;
}

uint32_t le32(const uint8_t * p) {
    return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}

// bounds-checked little-endian reader over the whole library file
struct reader {
    const std::vector<uint8_t> & buf;
    size_t pos = 0;

    void need(size_t n) const {
        if (n > buf.size() - pos) {
            throw std::runtime_error("jf_knn: truncated library file");
        }
    }

    uint32_t u32() {
        need(4);
        const uint32_t v = le32(buf.data() + pos);
        pos += 4;
        return v;
    }
};

} // namespace

std::string library_path_env() {
    const char * e = std::getenv("LLAMA_JF_KNN");
    return e != nullptr ? std::string(e) : std::string();
}

bool mmproj_key_env() {
    const char * e = std::getenv("LLAMA_JF_KNN_MMPROJ");
    return e != nullptr && std::strcmp(e, "1") == 0;
}

std::vector<uint8_t> thumb32(const unsigned char * rgb, uint32_t nx, uint32_t ny) {
    if (rgb == nullptr || nx < (uint32_t) THUMB_SIDE || ny < (uint32_t) THUMB_SIDE) {
        throw std::invalid_argument("jf_knn::thumb32: frame is missing or smaller than 32x32");
    }

    std::vector<uint8_t> out(THUMB_BYTES);
    for (uint64_t j = 0; j < (uint64_t) THUMB_SIDE; ++j) {
        const uint64_t y0 = j * ny / THUMB_SIDE;
        const uint64_t y1 = (j + 1) * ny / THUMB_SIDE;
        for (uint64_t i = 0; i < (uint64_t) THUMB_SIDE; ++i) {
            const uint64_t x0 = i * nx / THUMB_SIDE;
            const uint64_t x1 = (i + 1) * nx / THUMB_SIDE;

            uint64_t sum[3] = { 0, 0, 0 };
            for (uint64_t y = y0; y < y1; ++y) {
                const unsigned char * p = rgb + (y * nx + x0) * 3;
                for (uint64_t x = x0; x < x1; ++x, p += 3) {
                    sum[0] += p[0];
                    sum[1] += p[1];
                    sum[2] += p[2];
                }
            }

            const uint64_t n = (y1 - y0) * (x1 - x0);
            for (int c = 0; c < 3; ++c) {
                out[((size_t) j * THUMB_SIDE + (size_t) i) * 3 + (size_t) c] = (uint8_t) ((2 * sum[c] + n) / (2 * n));
            }
        }
    }
    return out;
}

void counts::add(int slot, const llama_token * ctx, size_t n_ctx, llama_token next) {
    auto & cells = map_[key_of(slot, ctx, n_ctx)];
    for (auto & c : cells) {
        if (c.tok == next) {
            c.n++;
            return;
        }
    }
    cells.push_back({ next, 1 });
}

llama_token counts::top(int slot, const llama_token * ctx, size_t n_ctx) const {
    const auto it = map_.find(key_of(slot, ctx, n_ctx));
    if (it == map_.end()) {
        return LLAMA_TOKEN_NULL;
    }
    const cell * best = nullptr;
    for (const auto & c : it->second) {
        if (best == nullptr || c.n > best->n) {
            best = &c; // strict: the entry inserted first wins ties
        }
    }
    return best != nullptr ? best->tok : LLAMA_TOKEN_NULL;
}

std::shared_ptr<const library> library_load(const std::string & path, int32_t n_vocab, bool with_mm) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error("jf_knn: cannot open " + path);
    }
    const std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<uint8_t> buf((size_t) size);
    if (size <= 0 || !f.read((char *) buf.data(), size)) {
        throw std::runtime_error("jf_knn: cannot read " + path);
    }

    reader r { buf };
    r.need(sizeof(LIB_MAGIC));
    if (std::memcmp(buf.data(), LIB_MAGIC, sizeof(LIB_MAGIC)) != 0) {
        throw std::runtime_error("jf_knn: bad magic in " + path);
    }
    r.pos = sizeof(LIB_MAGIC);
    if (r.u32() != LIB_VERSION) {
        throw std::runtime_error("jf_knn: unsupported library version");
    }
    const uint32_t n_items = r.u32();
    if (r.u32() != THUMB_BYTES || r.u32() != (uint32_t) N_SLOTS || r.u32() != (uint32_t) THUMB_SIDE) {
        throw std::runtime_error("jf_knn: library geometry does not match 32x32 RGB, 14 slots");
    }
    const uint32_t mm_dim = r.u32();
    if (r.pos != LIB_HEADER_SIZE || n_items == 0) {
        throw std::runtime_error("jf_knn: empty or malformed library header");
    }
    if (mm_dim > LIB_MM_DIM_MAX) {
        throw std::runtime_error("jf_knn: library mmproj width out of range");
    }
    if (with_mm && mm_dim == 0) {
        throw std::runtime_error("jf_knn: LLAMA_JF_KNN_MMPROJ=1 but the library has no mmproj block: " + path);
    }

    auto lib = std::make_shared<library>();

    r.need(THUMB_BYTES * 4);
    lib->mean.resize(THUMB_BYTES);
    for (size_t k = 0; k < THUMB_BYTES; ++k) {
        const uint32_t u = le32(buf.data() + r.pos);
        std::memcpy(&lib->mean[k], &u, 4);
        r.pos += 4;
    }

    r.need((size_t) n_items * THUMB_BYTES);
    const uint8_t * thumbs = buf.data() + r.pos;
    lib->feat.resize((size_t) n_items * THUMB_BYTES);
    for (size_t i = 0; i < n_items; ++i) {
        const uint8_t * t = thumbs + i * THUMB_BYTES;
        float *         o = lib->feat.data() + i * THUMB_BYTES;
        double          n2 = 0.0;
        for (size_t k = 0; k < THUMB_BYTES; ++k) {
            o[k] = (float) t[k] - lib->mean[k];
            n2 += (double) o[k] * o[k];
        }
        const float inv = n2 > 0.0 ? (float) (1.0 / std::sqrt(n2)) : 0.0f;
        for (size_t k = 0; k < THUMB_BYTES; ++k) {
            o[k] *= inv;
        }
    }
    r.pos += (size_t) n_items * THUMB_BYTES;

    lib->mm_dim = (int) mm_dim;
    if (mm_dim > 0) {
        const size_t D = mm_dim;
        r.need(D * 4);
        lib->mm_mean.resize(D);
        for (size_t k = 0; k < D; ++k) {
            const uint32_t u = le32(buf.data() + r.pos);
            std::memcpy(&lib->mm_mean[k], &u, 4);
            r.pos += 4;
        }

        r.need((size_t) n_items * D * 4);
        const uint8_t * raw = buf.data() + r.pos;
        if (with_mm) {
            lib->mm_feat.resize((size_t) n_items * D);
            for (size_t i = 0; i < n_items; ++i) {
                const uint8_t * p = raw + i * D * 4;
                float *         o = lib->mm_feat.data() + i * D;
                double          n2 = 0.0;
                for (size_t k = 0; k < D; ++k) {
                    const uint32_t u = le32(p + 4 * k);
                    float          v;
                    std::memcpy(&v, &u, 4);
                    o[k] = v - lib->mm_mean[k];
                    n2 += (double) o[k] * o[k];
                }
                const float inv = n2 > 0.0 ? (float) (1.0 / std::sqrt(n2)) : 0.0f;
                for (size_t k = 0; k < D; ++k) {
                    o[k] *= inv;
                }
            }
        }
        r.pos += (size_t) n_items * D * 4;
    }

    lib->answers.resize(n_items);
    for (auto & a : lib->answers) {
        a.image_index = r.u32();
        const uint32_t n_free = r.u32();
        r.need((size_t) n_free * 5); // one int8 slot and one int32 token per value token
        const uint8_t * slots  = buf.data() + r.pos;
        const uint8_t * tokens = slots + n_free;
        for (uint32_t j = 0; j < n_free; ++j) {
            const int s = (int8_t) slots[j];
            if (s < 0 || s >= N_SLOTS) {
                throw std::runtime_error("jf_knn: slot out of range in library");
            }
            const llama_token t = (llama_token) le32(tokens + 4 * (size_t) j);
            if (t < 0 || (n_vocab > 0 && t >= n_vocab)) {
                throw std::runtime_error("jf_knn: token out of vocabulary range in library");
            }
            a.runs[s].push_back(t);
        }
        r.pos += (size_t) n_free * 5;
    }
    if (r.pos != buf.size()) {
        throw std::runtime_error("jf_knn: trailing bytes in library file");
    }

    for (const auto & a : lib->answers) {
        for (int s = 0; s < N_SLOTS; ++s) {
            const auto & run = a.runs[s];
            for (size_t L = 0; L < run.size(); ++L) {
                lib->global.add(s, run.data(), L, run[L]);
            }
        }
    }

    return lib;
}

request_index::request_index(std::shared_ptr<const library> lib, const std::vector<uint8_t> & thumb,
                             const std::vector<float> & mm) : lib_(std::move(lib)) {
    if (thumb.size() != THUMB_BYTES) {
        throw std::invalid_argument("jf_knn: thumbnail has the wrong size");
    }

    // query feature: same transform as the library features
    std::vector<float> z(THUMB_BYTES);
    double n2 = 0.0;
    for (size_t k = 0; k < THUMB_BYTES; ++k) {
        z[k] = (float) thumb[k] - lib_->mean[k];
        n2 += (double) z[k] * z[k];
    }
    if (n2 > 0.0) {
        const float inv = (float) (1.0 / std::sqrt(n2));
        for (auto & v : z) {
            v *= inv;
        }
    }

    // mmproj block of the key (LLAMA_JF_KNN_MMPROJ=1): same transform; its cosine is added with weight 1
    std::vector<float> zm;
    if (!mm.empty()) {
        if (lib_->mm_feat.empty() || mm.size() != (size_t) lib_->mm_dim) {
            throw std::invalid_argument("jf_knn: mmproj feature does not match the library");
        }
        const size_t D = mm.size();
        zm.resize(D);
        double m2 = 0.0;
        for (size_t k = 0; k < D; ++k) {
            zm[k] = mm[k] - lib_->mm_mean[k];
            m2 += (double) zm[k] * zm[k];
        }
        if (m2 > 0.0) {
            const float inv = (float) (1.0 / std::sqrt(m2));
            for (auto & v : zm) {
                v *= inv;
            }
        }
    }

    // similarity to every library answer: sum of the block cosines (library features are unit norm)
    const size_t n = lib_->answers.size();
    std::vector<std::pair<float, uint32_t>> cand(n);
    for (size_t i = 0; i < n; ++i) {
        const float * f = lib_->feat.data() + i * THUMB_BYTES;
        float s = 0.0f;
        for (size_t k = 0; k < THUMB_BYTES; ++k) {
            s += f[k] * z[k];
        }
        if (!zm.empty()) {
            const float * g = lib_->mm_feat.data() + i * zm.size();
            for (size_t k = 0; k < zm.size(); ++k) {
                s += g[k] * zm[k];
            }
        }
        cand[i] = { s, (uint32_t) i };
    }
    const size_t K = std::min(K_NEIGHBOURS, n);
    std::partial_sort(cand.begin(), cand.begin() + (std::ptrdiff_t) K, cand.end(),
            [](const std::pair<float, uint32_t> & a, const std::pair<float, uint32_t> & b) {
                return a.first > b.first || (a.first == b.first && a.second < b.second);
            });
    nb_.resize(K);
    sim_.resize(K);
    for (size_t j = 0; j < K; ++j) {
        sim_[j] = cand[j].first;
        nb_[j]  = cand[j].second;
    }

    // kNN prefix counts and suffix counts over the neighbours, most similar first
    for (const uint32_t i : nb_) {
        const library::answer & a = lib_->answers[i];
        for (int s = 0; s < N_SLOTS; ++s) {
            const auto & run = a.runs[s];
            for (size_t L = 0; L < run.size(); ++L) {
                knn_.add(s, run.data(), L, run[L]);
                const size_t m = std::min<size_t>(L, 2);
                suffix_.add(s, run.data() + (L - m), m, run[L]);
            }
        }
    }
}

llama_token request_index::predict(int slot, const llama_token * prefix, size_t n_prefix, src * from) const {
    llama_token t = knn_.top(slot, prefix, n_prefix);
    if (t != LLAMA_TOKEN_NULL) {
        if (from) *from = src::KNN;
        return t;
    }
    t = lib_->global.top(slot, prefix, n_prefix);
    if (t != LLAMA_TOKEN_NULL) {
        if (from) *from = src::GLOBAL;
        return t;
    }
    const size_t m = std::min<size_t>(n_prefix, 2);
    t = suffix_.top(slot, prefix + (n_prefix - m), m);
    if (t != LLAMA_TOKEN_NULL) {
        if (from) *from = src::SUFFIX;
        return t;
    }
    if (from) *from = src::NONE;
    return LLAMA_TOKEN_NULL;
}

} // namespace jf_knn
