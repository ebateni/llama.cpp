#!/usr/bin/env python3
"""Per-op mdev breakdown from GGML_HEXAGON_PROFILE=3 logs (1c and Nc).

For the last graph (opbatch) of each log, every op on every NSP is split into
  flush  : L2FLUSH on thread 0 before the barrier (htp_tensor_flush_all)
  fence  : FENCE   on thread 0 (htp_mdev_group_barrier wait)
  exec   : the rest of the op's cycles
and aggregated per op class (opcode + dst name with the layer index removed).

Fence at op i is the barrier latency plus the imbalance of op i-1 (+ flush i):
  base(i)   = min_n fence_n(i)            -> charged to op i as "barrier"
  caused(i) = mean_n fence_n(i+1) - base(i+1) -> charged to op i as "imbalance"

Usage: mdev-profile-analyze.py <one-core.log[.gz|.part*]> <multi-core.log...> [--top N]
Input can be pre-filtered with: grep -E 'profile-op|FENCE|L2FLUSH'
"""
import gzip, io, re, sys, glob, statistics as st
from collections import defaultdict, OrderedDict

WRAP = 1 << 32

def open_any(path):
    parts = sorted(glob.glob(path + '.part*')) if not glob.glob(path) else [path]
    if not parts:
        sys.exit(f'no such file: {path}')
    raw = b''.join(open(p, 'rb').read() for p in parts) if len(parts) > 1 else None
    if raw is not None:
        return io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(raw)), errors='replace')
    p = parts[0]
    if p.endswith('.gz'):
        return io.TextIOWrapper(gzip.open(p, 'rb'), errors='replace')
    return open(p, errors='replace')

TAIL = re.compile(r'usec (\d+) cycles (\d+) start (\d+)')

def parse(path):
    """-> {sess: [batch]} ; batch = {'ops': [...], 'ev': [...]}"""
    sess = defaultdict(list)
    with open_any(path) as f:
        for line in f:
            if 'profile-op' in line:
                tok = line.split(None, 3)
                s = tok[1]
                body = tok[3]
                if body.startswith('OPBATCH'):
                    sess[s].append({'ops': [], 'ev': []})
                    continue
                if not sess[s]:
                    continue
                fl = body.split('|')
                m = TAIL.search(fl[-1])
                if not m:
                    continue
                names = fl[1]
                dst = names.split(' -> ')[-1].strip() if ' -> ' in names else names
                sess[s][-1]['ops'].append({
                    'op': fl[0], 'names': names, 'dst': dst, 'dims': fl[2], 'types': fl[3],
                    'kp': fl[5] if len(fl) > 6 else '',
                    'usec': int(m.group(1)), 'cyc': int(m.group(2)), 'start': int(m.group(3)) % WRAP})
            elif 'trace-evt' in line:
                tok = line.split()
                # ggml-hex: SESS trace-evt NAME: thread T info I start|stop CYC
                name = tok[3].rstrip(':')
                if name not in ('FENCE', 'L2FLUSH') or tok[5] != '0':
                    continue
                s = tok[1]
                if sess[s]:
                    sess[s][-1]['ev'].append((name, tok[8] == 'start', int(tok[9]) % WRAP))
    return sess

def rel(v, base):
    r = (v - base) % WRAP
    return r - WRAP if r > WRAP // 2 else r

def op_components(batch):
    """per op: flush, fence, exec cycles (thread 0)."""
    ops = batch['ops']
    if not ops:
        return []
    base = ops[0]['start']
    starts = [rel(o['start'], base) for o in ops]
    comp = [{'flush': 0, 'fence': 0} for _ in ops]
    open_ = {}
    import bisect
    # FENCE/L2FLUSH never nest on thread 0, so pair them by alternation: the
    # start/stop bit (0x8000) collides with high fence sequence numbers in
    # 'info', making later FENCE events all print as "stop".
    for name, _, c in batch['ev']:
        r = rel(c, base)
        if name not in open_:
            open_[name] = r
        elif name in open_:
            t0 = open_.pop(name)
            i = bisect.bisect_right(starts, t0) - 1
            if 0 <= i < len(ops) and t0 < starts[i] + ops[i]['cyc']:
                comp[i]['flush' if name == 'L2FLUSH' else 'fence'] += r - t0
    out = []
    for o, c in zip(ops, comp):
        k = o['usec'] / o['cyc'] if o['cyc'] else 0.0
        fl, fe = c['flush'] * k, c['fence'] * k
        out.append({**o, 'flush_us': fl, 'fence_us': fe, 'exec_us': max(0.0, o['usec'] - fl - fe)})
    return out

def cls(o):
    d = re.sub(r' \((view|permuted|copy[^)]*)\)', '', o['dst'])
    d = re.sub(r'-\d+$', '', d)
    d = re.sub(r'_l\d+$', '', d)
    d = re.sub(r'^node_\d+$', 'node', d)
    return f"{o['op']}:{d}"

def last_graph(sess):
    names = sorted(sess, key=lambda s: (s != 'HTP0', s))
    nb = min(len(sess[s]) for s in names)
    return names, [op_components(sess[s][nb - 1]) for s in names]

def fa_kv(ops):
    for o in ops:
        if o['op'].startswith('FLASH_ATTN'):
            dims = o['dims'].split(' x ')
            if len(dims) > 1:
                return dims[1].split(':')[1]
    return '?'

def analyze(p1, pn, top):
    s1 = parse(p1); _, g1 = last_graph(s1); g1 = g1[0]
    sn = parse(pn); names, gn = last_graph(sn)
    N = len(gn)
    n_ops = min(len(g) for g in gn)
    if len(g1) != n_ops:
        print(f'  note: 1c graph has {len(g1)} ops, {N}c has {n_ops}; matching by class')

    T1 = sum(o['usec'] for o in g1)
    Tn = st.mean(sum(o['usec'] for o in g[:n_ops]) for g in gn)
    print(f'graph: {N}c, {n_ops} ops, FA n_kv {fa_kv(gn[0])};  1c {T1/1e3:.1f} ms  {N}c {Tn/1e3:.1f} ms  speedup {T1/Tn:.2f}x')

    agg = OrderedDict()
    tot = defaultdict(float)
    for i in range(n_ops):
        ex = [g[i]['exec_us'] for g in gn]
        fl = [g[i]['flush_us'] for g in gn]
        fe = [g[i]['fence_us'] for g in gn]
        base = min(fe)
        caused = 0.0
        if i + 1 < n_ops:
            fe2 = [g[i + 1]['fence_us'] for g in gn]
            caused = st.mean(fe2) - min(fe2)
        o = gn[0][i]
        a = agg.setdefault(cls(o), defaultdict(float))
        a['n'] += 1
        a['exec'] += st.mean(ex); a['exec_max'] += max(ex)
        a['flush'] += st.mean(fl); a['barrier'] += base; a['imb'] += caused
        a['solo'] += 1 if (max(ex) > 20 and sorted(ex)[-2] < 0.1 * max(ex)) else 0
        a['ex0'] += ex[0]
        tot['exec'] += st.mean(ex); tot['flush'] += st.mean(fl); tot['barrier'] += base; tot['imb'] += caused
    c1 = defaultdict(float)
    for o in g1:
        c1[cls(o)] += o['exec_us'] + o['flush_us'] + o['fence_us']

    acc = sum(tot.values())
    print(f'  {N}c time split: exec {tot["exec"]/1e3:.1f} ms ({100*tot["exec"]/acc:.0f}%)  '
          f'flush {tot["flush"]/1e3:.1f} ms ({100*tot["flush"]/acc:.0f}%)  '
          f'barrier {tot["barrier"]/1e3:.1f} ms ({100*tot["barrier"]/acc:.0f}%)  '
          f'imbalance {tot["imb"]/1e3:.1f} ms ({100*tot["imb"]/acc:.0f}%)  '
          f'[per-op barrier {1e3*tot["barrier"]/n_ops/1e3:.1f} us]')
    print(f'  ideal (1c/{N}) {T1/N/1e3:.1f} ms; lost {Tn/1e3 - T1/N/1e3:.1f} ms')
    hdr = f'  {"class":42s} {"n":>3s} {"1c ms":>7s} {N}c ms  {"x":>5s} {"exec":>6s} {"max/mean":>8s} {"flush":>6s} {"barr":>6s} {"imb":>6s} {"loss ms":>7s} solo'
    print(hdr)
    rows = []
    for k, a in agg.items():
        wall = a['exec'] + a['flush'] + a['barrier'] + a['imb']
        one = c1.get(k, 0.0)
        loss = wall - one / N
        rows.append((loss, k, a, wall, one))
    for loss, k, a, wall, one in sorted(rows, key=lambda r: -r[0])[:top]:
        mm = a['exec_max'] / a['exec'] if a['exec'] else 0
        sp = one / wall if wall else 0
        print(f'  {k[:42]:42s} {int(a["n"]):3d} {one/1e3:7.2f} {wall/1e3:6.2f}  {sp:5.2f} {a["exec"]/1e3:6.2f} {mm:8.2f} '
              f'{a["flush"]/1e3:6.2f} {a["barrier"]/1e3:6.2f} {a["imb"]/1e3:6.2f} {loss/1e3:7.2f} {int(a["solo"]) or ""}')

if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    top = 25
    if '--top' in sys.argv:
        top = int(sys.argv[sys.argv.index('--top') + 1]); args.remove(str(top))
    analyze(args[0], args[1], top)
