// Dump MUL_MAT src[1] (f32, row-major [n_tokens, k]) per blk.L weight to $DUMP_DIR/CLS_<class>_<L>.f32 (default .) for smooth_calib.py.
// Usage: act_dump <model.gguf> <prompt.txt>. Build with c++ -std=c++17, link -lllama -lggml from build/bin.

#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static bool cb(struct ggml_tensor * t, bool ask, void *) {
    if (ask) {
        return true;
    }

    static const char * suffix[] = { "attn_q.weight", "attn_output.weight", "ffn_gate.weight", "ffn_down.weight",
                                     "per_layer_inp_gate.weight", "per_layer_proj.weight", NULL };
    static const char * sname[]  = { "qkv", "o", "gate", "down", "pgate", "pproj" };

    char key[64] = "";
    int layer = 99;
    const char * weight = t->src[0] ? t->src[0]->name : "";

    if (strcmp(weight, "output.weight") == 0) {
        snprintf(key, sizeof key, "lmhead");
    } else if (strcmp(weight, "per_layer_model_proj.weight") == 0) {
        snprintf(key, sizeof key, "mproj");
    } else if (strncmp(weight, "blk.", 4) == 0) {
        char tail[64] = "";
        if (sscanf(weight + 4, "%d.%63s", &layer, tail) == 2) {
            for (int i = 0; suffix[i]; ++i) {
                if (strcmp(tail, suffix[i]) == 0) {
                    snprintf(key, sizeof key, "%s", sname[i]);
                    break;
                }
            }
        }
    }
    if (!key[0]) {
        return true;
    }

    const struct ggml_tensor * src = t->src[1];
    const char * dir = getenv("DUMP_DIR") ? getenv("DUMP_DIR") : ".";

    char fn[512];
    snprintf(fn, sizeof fn, "%s/CLS_%s_%d.f32", dir, key, layer);

    FILE * f = fopen(fn, "wb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", fn);
        return true;
    }

    std::vector<float> buf(ggml_nelements(src));
    if (src->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(src, buf.data(), 0, ggml_nbytes(src));
    } else if (src->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h(ggml_nelements(src));
        ggml_backend_tensor_get(src, h.data(), 0, ggml_nbytes(src));
        for (size_t i = 0; i < h.size(); ++i) {
            buf[i] = ggml_fp16_to_fp32(h[i]);
        }
    } else {
        fclose(f);
        return true;
    }

    fwrite(buf.data(), sizeof(float), buf.size(), f);
    fclose(f);
    fprintf(stderr, "dumped %s act of %s w=%s ne=%ld,%ld\n", key, src->name, weight, (long) src->ne[0], (long) src->ne[1]);
    return true;
}

int main(int argc, char ** argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s <model.gguf> <prompt.txt>\n", argv[0]);
        return 2;
    }

    llama_backend_init();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.use_extra_bufts = false;

    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) {
        fprintf(stderr, "load failed: %s\n", argv[1]);
        return 1;
    }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 1024;
    cp.n_batch = 1024;
    cp.n_ubatch = 1024;
    cp.cb_eval = cb;
    cp.cb_eval_user_data = nullptr;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "context creation failed\n");
        return 1;
    }

    FILE * f = fopen(argv[2], "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", argv[2]);
        return 1;
    }
    std::string txt(1 << 16, 0);
    txt.resize(fread(&txt[0], 1, txt.size(), f));
    fclose(f);

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(4096);
    int n = llama_tokenize(vocab, txt.data(), (int) txt.size(), toks.data(), (int) toks.size(), true, true);
    if (n > 512) {
        n = 512;
    }
    fprintf(stderr, "ntok %d\n", n);

    llama_batch batch = llama_batch_init(n, 0, 1);
    batch.n_tokens = n;
    for (int i = 0; i < n; ++i) {
        batch.token[i] = toks[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }

    int rc = llama_decode(ctx, batch);
    fprintf(stderr, "rc %d\n", rc);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return rc;
}
