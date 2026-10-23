#!/usr/bin/env python3
"""Generate the Hexagon int8 activation-smoothing file (smooth.bin).

The packed int8 path uses W' = W * s_k per input channel and divides the
activation by s_k on the DSP. The scale comes from a static calibration:

    s_k = max_t|x_k|^a / max_n|w_nk|^(1-a), normalized to geometric mean 1

Weight groups share one activation: q/k/v use blk.L.attn_q.weight as the
record key, gate/up use blk.L.ffn_gate.weight, o uses blk.L.attn_output.weight.
The record format is:

    [u32 name_len][name][u32 k][f32 s[k]]   (repeat)

Calibration needs the class-input activations. Build the dumper from
scripts/snapdragon/act_dump.cpp and link it against this tree, for example:

    c++ -O2 -std=c++17 -I include -I ggml/include \
        scripts/snapdragon/act_dump.cpp -o act_dump \
        -L build/bin -Wl,-rpath,$PWD/build/bin -lllama -lggml

Then run:

    python3 scripts/snapdragon/smooth_calib.py \
        --model gemma-4-E2B-it-Q4_0-imatembd.gguf \
        --calib-dir calib --dumper ./act_dump --out smooth.bin

The calibration texts are capped at 512 tokens each by the dumper. Use
--reuse-dir to score an existing CLS_*.f32 set without running the model, and
--verify to compare against a reference smooth.bin.
"""

import argparse
import glob
import hashlib
import os
import re
import struct
import subprocess
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "gguf-py"))
import gguf  # noqa: E402

# class -> (weight suffixes, activation file prefix, record key suffix, alpha)
GROUPS = {
    "qkv":  (["attn_q", "attn_k", "attn_v"], "qkv", "attn_q",      0.5),
    "gate": (["ffn_gate", "ffn_up"],          "gate", "ffn_gate",  0.7),
    "o":    (["attn_output"],                 "o",    "attn_output", 0.7),
}
EPS = 1e-6
DEFAULT_CLIP = 64.0


def deq_rows(t, n_rows):
    """Dequantize the first n_rows output rows of a Q4_0/Q4_1 tensor."""
    k = int(t.shape[0])
    nb = k // 32
    qtype = t.tensor_type.name
    if qtype == "Q4_0":
        raw = np.asarray(t.data).reshape(int(t.shape[1]), nb, 18)[:n_rows]
        d = np.ascontiguousarray(raw[:, :, 0:2]).view(np.float16).astype(np.float32)[:, :, 0]
        qs = raw[:, :, 2:]
        q = np.concatenate([(qs & 0xF).astype(np.int8) - 8,
                            (qs >> 4).astype(np.int8) - 8], axis=2).astype(np.float32)
        return q, d, None
    if qtype == "Q4_1":
        raw = np.asarray(t.data).reshape(int(t.shape[1]), nb, 20)[:n_rows]
        d = np.ascontiguousarray(raw[:, :, 0:2]).view(np.float16).astype(np.float32)[:, :, 0]
        m = np.ascontiguousarray(raw[:, :, 2:4]).view(np.float16).astype(np.float32)[:, :, 0]
        qs = raw[:, :, 4:]
        q = np.concatenate([(qs & 0xF), (qs >> 4)], axis=2).astype(np.float32)
        return q, d, m
    raise SystemExit("unsupported tensor type %s (expected Q4_0 or Q4_1)" % qtype)


def w_fp32(q, d, m):
    w = q * d[:, :, None]
    if m is not None:
        w = w + m[:, :, None]
    return w.reshape(q.shape[0], -1)


def weight_absmax(tensors, layer, names):
    """max_n|w_nk| per input channel, combined over the group's weights."""
    out = None
    for name in names:
        key = "blk.%d.%s.weight" % (layer, name)
        if key not in tensors:
            raise SystemExit("missing tensor " + key)
        t = tensors[key]
        q, d, m = deq_rows(t, int(t.shape[1]))
        w = np.abs(w_fp32(q, d, m)).max(axis=0)
        out = w if out is None else np.maximum(out, w)
    return out + EPS


def act_absmax(path, k):
    a = np.fromfile(path, dtype=np.float32)
    if a.size == 0 or a.size % k != 0:
        raise SystemExit("%s: size %d is not a multiple of k=%d" % (path, a.size, k))
    return np.abs(a.reshape(-1, k)).max(axis=0)


def run_dumper(dumper, model, text, dump_dir, verbose):
    env = dict(os.environ)
    env["DUMP_DIR"] = dump_dir
    stderr = None if verbose else subprocess.DEVNULL
    subprocess.run([dumper, model, text], env=env, stderr=stderr, check=True)


def write_smooth(path, records):
    with open(path, "wb") as f:
        for key in sorted(records):
            name = records[key][0]
            s = records[key][1].astype("<f4")
            f.write(struct.pack("<I", len(name)))
            f.write(name)
            f.write(struct.pack("<I", len(s)))
            f.write(s.tobytes())


def read_smooth(path):
    out = {}
    with open(path, "rb") as f:
        data = f.read()
    off = 0
    while off < len(data):
        n = struct.unpack_from("<I", data, off)[0]
        off += 4
        name = data[off:off + n].decode()
        off += n
        k = struct.unpack_from("<I", data, off)[0]
        off += 4
        s = np.frombuffer(data, dtype="<f4", count=k, offset=off).copy()
        off += 4 * k
        out[name] = s
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="GGUF model used for calibration")
    ap.add_argument("--out", default="smooth.bin", help="output file (default: smooth.bin)")
    ap.add_argument("--calib-dir", default="calib", help="directory with calibration texts")
    ap.add_argument("--calib-glob", default="t*.txt", help="text glob inside --calib-dir (default: t*.txt)")
    ap.add_argument("--dump-dir", default=None, help="activation dump directory (default: calib dir)")
    ap.add_argument("--dumper", default=None, help="act_dump executable; omit with --reuse-dir")
    ap.add_argument("--reuse-dir", default=None, help="score existing CLS_*.f32 files, no model run")
    ap.add_argument("--verify", default=None, help="compare the result with a reference smooth.bin")
    ap.add_argument("--clip", type=float, default=DEFAULT_CLIP, help="clip s to [1/clip, clip]")
    ap.add_argument("--alpha-qkv", type=float, default=GROUPS["qkv"][3])
    ap.add_argument("--alpha-gate", type=float, default=GROUPS["gate"][3])
    ap.add_argument("--alpha-o", type=float, default=GROUPS["o"][3])
    ap.add_argument("--verbose", action="store_true", help="show dumper output")
    args = ap.parse_args()

    alphas = {"qkv": args.alpha_qkv, "gate": args.alpha_gate, "o": args.alpha_o}
    if args.reuse_dir:
        if args.dumper:
            raise SystemExit("use either --reuse-dir or --dumper")
        dump_dir = args.reuse_dir
        texts = []
    else:
        if not args.dumper:
            raise SystemExit("--dumper is required unless --reuse-dir is used")
        dump_dir = args.dump_dir or args.calib_dir
        texts = sorted(glob.glob(os.path.join(args.calib_dir, args.calib_glob)),
                       key=lambda p: int(re.search(r"t(\d+)", os.path.basename(p)).group(1)))
        if not texts:
            raise SystemExit("no t*.txt files in " + args.calib_dir)
        os.makedirs(dump_dir, exist_ok=True)

    reader = gguf.GGUFReader(args.model)
    tensors = {t.name: t for t in reader.tensors}
    arch = reader.get_field("general.architecture").contents()
    n_layer = int(reader.get_field(arch + ".block_count").contents())
    print("model: %s, arch %s, %d layers" % (args.model, arch, n_layer))

    ks = {}
    for cls, (_, _, key, _) in GROUPS.items():
        for layer in range(n_layer):
            name = "blk.%d.%s.weight" % (layer, key)
            if name not in tensors:
                raise SystemExit("missing tensor " + name)
            ks[(cls, layer)] = int(tensors[name].shape[0])

    ax = {}
    if texts:
        for i, text in enumerate(texts):
            print("dump %d/%d %s" % (i + 1, len(texts), os.path.basename(text)), flush=True)
            run_dumper(args.dumper, args.model, text, dump_dir, args.verbose)
            for cls in GROUPS:
                for layer in range(n_layer):
                    path = os.path.join(dump_dir, "CLS_%s_%d.f32" % (cls, layer))
                    if not os.path.exists(path):
                        raise SystemExit("missing activation " + path)
                    a = act_absmax(path, ks[(cls, layer)])
                    key = (cls, layer)
                    ax[key] = a if key not in ax else np.maximum(ax[key], a)
    else:
        for cls in GROUPS:
            for layer in range(n_layer):
                path = os.path.join(dump_dir, "CLS_%s_%d.f32" % (cls, layer))
                if not os.path.exists(path):
                    raise SystemExit("missing activation " + path)
                ax[(cls, layer)] = act_absmax(path, ks[(cls, layer)])

    records = {}
    for cls, (names, _, key, _) in GROUPS.items():
        alpha = alphas[cls]
        for layer in range(n_layer):
            aw = weight_absmax(tensors, layer, names)
            s = (ax[(cls, layer)] + EPS) ** alpha / (aw ** (1.0 - alpha))
            s = s / np.exp(np.log(s).mean())
            s = np.clip(s, 1.0 / args.clip, args.clip).astype(np.float32)
            name = ("blk.%d.%s.weight" % (layer, key)).encode()
            records[(cls, layer)] = (name, s)

    write_smooth(args.out, records)
    data = open(args.out, "rb").read()
    print("wrote %s: %d bytes, %d vectors, sha256 %s" % (
        args.out, len(data), len(records), hashlib.sha256(data).hexdigest()))

    if args.verify:
        ref = read_smooth(args.verify)
        got = read_smooth(args.out)
        if set(ref) != set(got):
            print("verify: key mismatch (ref %d, got %d)" % (len(ref), len(got)))
            return 1
        worst = 0.0
        for name in ref:
            worst = max(worst, float(np.abs(ref[name] - got[name]).max()))
        print("verify: %d vectors, max |diff| %.3e" % (len(ref), worst))
        if open(args.verify, "rb").read() == data:
            print("verify: byte-identical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
