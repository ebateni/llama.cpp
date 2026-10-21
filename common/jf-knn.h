#pragma once

// jf-knn (LLAMA_JF_KNN=<library file>; unset = off): image-thumbnail nearest-neighbour value predictor for the cabin template drafter (common/speculative.cpp, draft_template_one). The K library answers whose images are closest in cosine (thumbnails centred on the library mean) supply their value runs; a free value token is predicted by kNN prefix counts -> global prefix counts over all library answers -> kNN suffix counts (last two value tokens). Replay: analysis/speed150/research/draftv2 (thumb32+mmproj).
// Each block is centred on the library mean and normalised; the similarity is the sum of the two block cosines (weights 1.0). The library then needs an mmproj block (format v2). Design: analysis/speed150/impl/ImplKnnDraft.md. Design: analysis/prefill4k/decode200/D2Tau.md. The request image is reduced to a 32x32 RGB thumbnail of the decoded frame. When none has a candidate the caller falls back to the ngram cache. LLAMA_JF_KNN_MMPROJ=1 (default off): the key adds a second block, the model's own mmproj embedding of the same image (mean over its image tokens, 2560 floats).

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace jf_knn {

inline constexpr int    THUMB_SIDE   = 32;
inline constexpr size_t THUMB_BYTES  = (size_t) THUMB_SIDE * THUMB_SIDE * 3; // uint8 RGB, layout [y][x][c]
inline constexpr int    N_SLOTS      = 14;  // value slots of the cabin answer (jf_template segment order)
inline constexpr size_t K_NEIGHBOURS = 40;

enum class src : int8_t { NONE = -1, KNN = 0, GLOBAL = 1, SUFFIX = 2 };

// LLAMA_JF_KNN library path; empty when unset
std::string library_path_env();

// LLAMA_JF_KNN_MMPROJ=1: the key includes the mmproj block; false when unset
bool mmproj_key_env();

// 32x32 RGB thumbnail of an 8-bit RGB frame (RGBRGB..., nx*ny*3 bytes, nx and ny >= THUMB_SIDE). Pixel (i, j) is the
// per-channel integer mean of the source block [i*nx/32, (i+1)*nx/32) x [j*ny/32, (j+1)*ny/32), rounded half up.
std::vector<uint8_t> thumb32(const unsigned char * rgb, uint32_t nx, uint32_t ny);

// (slot, token context) -> next-token counts; ties resolve to the entry that was inserted first
class counts {
public:
    void add(int slot, const llama_token * ctx, size_t n_ctx, llama_token next);
    llama_token top(int slot, const llama_token * ctx, size_t n_ctx) const; // LLAMA_TOKEN_NULL if no entry

private:
    struct cell {
        llama_token tok;
        uint32_t    n;
    };
    std::unordered_map<std::string, std::vector<cell>> map_;
};

// immutable library
struct library {
    struct answer {
        uint32_t                 image_index;
        std::vector<llama_token> runs[N_SLOTS]; // value tokens of each slot, in answer order
    };

    std::vector<answer> answers;
    std::vector<float>  mean; // THUMB_BYTES: mean thumbnail over the library
    std::vector<float>  feat; // answers.size() x THUMB_BYTES: (thumb - mean) / |thumb - mean|
    int                 mm_dim = 0;  // mmproj block width; 0 when the library has no mmproj block
    std::vector<float>  mm_mean;     // mm_dim: mean mmproj embedding over the library (loaded with the mmproj key only)
    std::vector<float>  mm_feat;     // answers.size() x mm_dim: (mmproj - mm_mean) / |.| (loaded with the mmproj key only)
    counts              global; // prefix counts over all answers, library order
};

// file format v2 (little endian): magic "JFKNNLB2", u32 version 2, u32 n_items, u32 THUMB_BYTES, u32 N_SLOTS, u32 THUMB_SIDE, u32 mm_dim (0 = no mmproj block); f32 thumbnail mean; n_items x THUMB_BYTES u8 thumbnails; if mm_dim: f32 mmproj mean (mm_dim) and n_items x mm_dim f32 raw mmproj embeddings; then per answer: u32 image_index, u32 n_free, int8 slot[n_free], int32 token[n_free].
// with_mm: also load the mmproj block (throws if the library has none). Throws std::runtime_error.
std::shared_ptr<const library> library_load(const std::string & path, int32_t n_vocab, bool with_mm);

// per-request index: the K library answers closest to the image, with their kNN prefix and suffix counts
class request_index {
public:
    // mm: mmproj embedding of the request's first image (mm_dim floats); empty = thumbnail key only
    request_index(std::shared_ptr<const library> lib, const std::vector<uint8_t> & thumb, const std::vector<float> & mm);

    // next value token of `slot` after the slot's value tokens so far (`prefix`); LLAMA_TOKEN_NULL if no source has a
    // candidate. `from` (optional) receives the source that answered.
    llama_token predict(int slot, const llama_token * prefix, size_t n_prefix, src * from = nullptr) const;

    const std::vector<uint32_t> & neighbours() const { return nb_; } // library indices, most similar first
    const std::vector<float> &    sims()       const { return sim_; } // their similarities (sum of the key blocks' cosines)

private:
    std::shared_ptr<const library> lib_;
    std::vector<uint32_t>          nb_;
    std::vector<float>             sim_;
    counts                         knn_;
    counts                         suffix_;
};

} // namespace jf_knn
