# -*- coding: utf-8 -*-
"""上游 vs 我方：开关、环境变量、能力关键词的差集（判据可复算）。"""
import os, re, sys, hashlib
sys.stdout.reconfigure(encoding='utf-8')
OURS = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src-tree', 'fusion-engine-src')
UP   = r'E:\infer-build\refs\infer-all-full\ninfer-all-master'
TAN  = r'E:\infer-build\refs\community\tancau-ninfer-kvmem-ring'

def walk_text(root):
    for dp, dn, fn in os.walk(root):
        if '__pycache__' in dp or '\\.git' in dp: continue
        for f in fn:
            if f.endswith(('.pyc', '.png', '.jpg', '.ninfer', '.exe', '.dll', '.zip')): continue
            p = os.path.join(dp, f)
            try:
                t = open(p, encoding='utf-8', errors='ignore').read()
            except OSError:
                continue
            yield os.path.relpath(p, root), t

OPT = re.compile(r'--[a-z][a-z0-9][a-z0-9-]{2,}')
ENV = re.compile(r'NINFER_[A-Z0-9_]{2,}')

def collect(root):
    opts, envs, allt = set(), set(), {}
    for rel, t in walk_text(root):
        allt[rel] = t
        opts |= set(OPT.findall(t))
        envs |= set(ENV.findall(t))
    return opts, envs, allt

o_opts, o_env, o_files = collect(OURS)
u_opts, u_env, u_files = collect(UP)
t_opts, t_env, t_files = collect(TAN)

print('files scanned: ours=%d upstream=%d tancau=%d' % (len(o_files), len(u_files), len(t_files)))
print()
print('== 命令行开关：我方有、上游没有（%d）==' % len(o_opts - u_opts))
for x in sorted(o_opts - u_opts): print('   +', x)
print('== 命令行开关：上游有、我方没有（%d）==' % len(u_opts - o_opts))
for x in sorted(u_opts - o_opts): print('   -', x)
print()
print('== 环境变量：我方有、上游没有（%d）==' % len(o_env - u_env))
for x in sorted(o_env - u_env): print('   +', x)
print('== 环境变量：上游有、我方没有（%d）==' % len(u_env - o_env))
for x in sorted(u_env - o_env): print('   -', x)
print()
print('== 能力关键词：三棵树命中文件数（我方 / 上游 / tancau）==')
for kw in ['kvmem', 'raw_k_shadow', 'mean_k_index', 'kvmem_window', 'kvmem_retrieve', 'kvmem_select',
           'NINFER_KV_RING', 'NINFER_KV_WINDOW', 'NINFER_KV_RETRIEVE', 'NINFER_HOST_PAGEABLE',
           'NINFER_KV_REUSE_HOSTBACKED', 'NINFER_TERNARY_KVMEM', 'host_pageable', 'host_kv_store',
           't2_ptq1', 'device_profiles', 'no_cuda_graph', 'no-cuda-graph', 'kvmem_resident_pages',
           'KVMI-012', 'kv_capacity', 'kv-capacity', 'host-kv-mib', 'lm-head-draft', 'dflash2']:
    def cnt(d): return sum(1 for t in d.values() if kw in t)
    print('   %-28s ours=%-4d upstream=%-4d tancau=%d' % (kw, cnt(o_files), cnt(u_files), cnt(t_files)))
print()
print('== 我方新增文件按目录归类（上游无此路径）==')
added = sorted(set(o_files) - set(u_files))
from collections import Counter
c = Counter(os.path.dirname(k) for k in added)
for d, n in c.most_common(20): print('   %-52s %d' % (d.replace('/', '\\'), n))
