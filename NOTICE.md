# NOTICE · 署名、许可与我方改动声明

本仓按 **Apache License 2.0** 发布（原文见 [LICENSE](LICENSE)）。
本文件是 Apache-2.0 **§4** 要求的署名与改动声明，**只覆盖本仓的内容，不覆盖任何模型权重**（本仓不含权重）。

---

## 1. 基座与来源（Third-party）

**上游出处（2026-10-04 已核实，GitHub API 读 `license.spdx_id`）**

| 仓 | 角色 | 默认分支 | 许可 |
|---|---|---|---|
| [`Neroued/ninfer`](https://github.com/Neroued/ninfer) | NInfer 原始仓 | `master` | **Apache-2.0** |
| [`ashalliants/ninfer-3090`](https://github.com/ashalliants/ninfer-3090) | 引擎树 README 自称的基线（`master`） | `master` | **Apache-2.0** |
| [`iamwavecut/ninfer-3090`](https://github.com/iamwavecut/ninfer-3090) | 本机存档目录名对应的仓（`VERSION = 0.11.0-rtx3090`） | `master` | **Apache-2.0** |
| [`tancau/ninfer-kvmem-ring`](https://github.com/tancau/ninfer-kvmem-ring) | 环的原始线（第三方，见 §2.1） | `main` | **Apache-2.0** |
| [`kvmem/kvmem-qw3`](https://github.com/kvmem/kvmem-qw3) | 策略层（第三方，作者 Di Chai，源码随树） | `main` | **Apache-2.0** |

> **关于逐文件版权头**：上游 NInfer 的源文件**本身不带逐文件 `Copyright` 行**（实测：整棵树仅 1 个文件含该行），署名以**树根 `LICENSE`** 与各 `third_party/*/LICENSE` 为准 —— 这不是我们抹掉了版权头，特此说明以免误会。


| 组件 | 在本仓里的形态 | 许可 | 证据 / 出处 |
|---|---|---|---|
| **NInfer**（引擎基座） | `patches/changed-files/` 是针对它写的改动 | **Apache-2.0** | 上游树内 `src/LICENSE`（11,357 B，与 [LICENSE](LICENSE) 同源）|
| **`tancau/ninfer-kvmem-ring`** | **代码已并入**我们的引擎（KVMem 环：打分 → 降级到主机 → 映射 → 按需搬回 → 装掩码）| **Apache-2.0** | 该仓 `LICENSE`（本机逐字核过；作者 README 的 License 段声明）|
| **`kvmem-qw3`**（作者 Di Chai） | **含其源码改造版（8 件，`src/ops/kvmem/qw3/`，头部带 `PORTED` 声明）**，语义沿用其常驻窗口 / 检索预算 / 差量计划 | **Apache-2.0** | 其 `LICENSE` + `THIRD_PARTY_NOTICES` |
| **NInfer-all / `iamwavecut`** | 基线：`VERSION = 0.11.0-rtx3090` | Apache-2.0 | 见 `patches/README-改动说明.md` §1 |

⚠️ **本仓不含**：模型权重、`.ninfer` 制品、引擎二进制（`.exe` / `.dll`）、CUDA 运行库、FFmpeg 等第三方二进制。
（我们发布的**分发包**里含这些二进制。⚠️ **2026-10-02 复核更正**：旧文档称"许可文本随那个包走"，但**包内实际没有 `licenses\` 目录** —— 多数许可正文**未随包**；逐条见 §7 与 §6 第 12 条。）

---

## 2. 我方改动声明（Apache-2.0 §4(b)：修改过的文件必须带显著声明）

**改动的逐文件清单、基线与可复现判据** = [`patches/README-改动说明.md`](patches/README-改动说明.md)。
**判据（2026-10-03 逐文件 SHA256 重算）**：本树 2,463 件 vs 上游基线 NInfer-all 2,419 件（两边均排除 `__pycache__`）——
**2,327 件逐字节相同**、**92 件为我方修改**、**44 件为我方新增**、**0 件缺失**。
**`patches/changed-files/` 收录的是其中 38 件的改后完整文件**（KVMem 环与启动期修复那一批）；
其余改动与新增**随源码树整体发布**（`src-tree/fusion-engine-src/`），不再单列。
（早前此处记的"2,424 件里 2,385 件相同 / 34 件改动"为过时口径，已按重算读数改写，复算脚本：`verify/reconcile-vs-upstream.ps1`；清单：`patches/changed-files.txt`。）

类别级声明（每条都指向 `patches/changed-files/` 里的具体文件）：

| # | 类别 | 性质 | 归属 |
|---|---|---|---|
| a | **KVMem 环 / 存储解耦** | **我方移植与改造**（设计目标：让设备池可以小于逻辑上下文；新增 `src/ops/kvmem/` 主体 44 件 + 相关改动） | 我方；含与 `tancau/ninfer-kvmem-ring` 逐字节相同的 4 件（见 **§2.1**）|
| b | **host-backed 复用 + 惰性领用** | **我方实现** | 我方 |
| c | **按工作集定池** | **我方实现**（定池做法与 CraneBW / laamaafung 的 `kvmem_resident_pages` 同向，出处注释随源码，见 **§2.1**；**未并入其代码**） | 我方 |
| d | **启动守卫**（装不下 ⇒ 启动期拒绝并给出所需页数） | **我方实现** | 我方 |
| e | **按卡自适应档位**（本机 4080 SUPER 标定档并进内置表） | **我方标定数据** | 我方 |
| f | **内核族与启动参数的配套改动** | **我方实现** | 我方 |

**逐行差异没有随本仓发布**（本仓给的是"改后的完整文件"）。要行级 diff：

```bat
git diff --no-index --stat "<上游树>" "<你的树>"
```

---

## 2.1 本仓包含哪些第三方代码（**逐文件实测，2026-10-03**）

判据：与三个上游副本逐文件 SHA256 比对（`tancau` 仓 `refs\community\tancau-ninfer-kvmem-ring`、
`kvmem-qw3` 仓 `refs\kvmem\kvmem-qw3-main`、基线 `refs\infer-all-full\infer-all-master`）。

| 来源 | 许可 | 本仓落点 | 实测判据 |
|---|---|---|---|
| 上游 **NInfer-all**（引擎基座） | Apache-2.0（`LICENSE`，11,357 B） | 整棵树 | 2,327 件与基线逐字节相同 |
| **`kvmem/kvmem-qw3`**（作者 Di Chai） | Apache-2.0（`LICENSE` 11,358 B + `THIRD_PARTY_NOTICES.md` 逐字节随树） | `src/ops/kvmem/qw3/` **8 件代码** + 许可 + notices | 许可与 notices **逐字节相同**；8 件代码为**我方改造版**（与原文逐字节不同，如 `kvmem_store.cpp` 31,213 B vs 原文 30,482 B）。每件头部带 `PORTED from kvmem-qw3 … Original: src\kvmem_store.cpp … Retrieved 2026-09-19, ported into the ninfer ternary tree on 2026-09-21` 声明 |
| **`tancau/ninfer-kvmem-ring`** | Apache-2.0（`LICENSE` 11,357 B，与本仓根 `LICENSE` 同一文本，sha256 `c71d239d…`） | `src/ops/kernel/paged_kv_address.cuh`、`src/ops/softmax_attention/dense/causal_cache/prompt_nvfp4.{cu,cuh}`、`…/small_t_nvfp4.cu` —— **4 件与本仓外的 tancau 仓逐字节相同、与上游基线不同** | 另有 **65 件**双方都偏离上游的执行文件（`core/paged_kv_cache.*`、`program/storage/context.cpp`、`program/planning/startup.*`、`program/program{,_impl}.*`、`transactions/*` 等），内容两边各自不同；`prompt_fp8.cuh`、`small_t_fp8.cuh` 注有 `ported from tancau`。tancau 仓**未附 NOTICE 文件**（实测：仓内无 `NOTICE`/`COPYING`/`AUTHORS`） |
| **`CraneBW` / `laamaafung` 线**（**仅语义参照，未并入其代码**） | 不适用：无代码交换 ⇒ **不触发许可义务**（其自述 Apache-2.0） | `src/models/qwen3_5/program/planning/startup.{cpp,h}` 的 `kvmem_resident_pages` 语义（自动定池 + 启动守卫） | 源码注释自带出处：`// LOCAL PORT (CraneBW / laamaafung 'kvmem_resident_pages'; source of the copied SEMANTICS: …exp\cranebw\diffwork\theirs_layouts_impl.h:673-697)`（`startup.cpp:119`、`:197`、`:1509`、`startup.h:100`）；原参照文件**已不在本机**，无法逐字节比对 |

**我方工作在其中的位置**：上表是"用到的他人代码"，逐条有出处与许可证据；
在它之上做的是**移植、接入、改造、定容、实验、修崩溃、打包**（逐条见 `README.md` §0），
按 Apache-2.0 §4(b) 在文件头声明改动（`qw3/` 8 件、`startup.*` 注释、`patches/changed-files/` 38 件）。
**两件事要分开说**：我方工作是真的（92 件修改 + 44 件新增，见 §2），他人代码也是真的（上表），不互相抵消。

## 3. 本仓的性质

- **研究与工程记录**：所有读数都带口径（见 `docs/00`），**未验项逐条列出**（见 `README.md` §5 与 `docs/05`），不隐藏。
- 所有"来源"标注都指向**可复核的凭据**（许可原文或文件路径）；拿不出凭据的，一律写"**未核**"，不含糊。
- **尊重原作**：模型权重版权归其各自作者与上游（Qwen 体系等）所有，按其各自许可发布；本仓只做**引擎侧适配与工具**。
- **逐文件版权头**：上游源文件本身不带 `Copyright` 行（实测全树仅 1 个文件含该行），署名以树根 `LICENSE` 与各 `third_party/*/LICENSE` 为准；本仓未删改任何上游版权信息。

---

## 4. 未核 / 待办（诚实清单）

1. ✅ **上游仓的公开 URL 已于 2026-10-04 核实**（`Neroued/ninfer` · `ashalliants/ninfer-3090` · `iamwavecut/ninfer-3090`，分支见 §2 的上游出处表；三者均为 Apache-2.0）。
2. **`CraneBW` / `laamaafung` 线：仅语义参照，未并入其代码** —— `kvmem_resident_pages` 的定池做法与其同向（出处注释随源码，见 §2.1），两边**没有代码交换** ⇒ 不构成再分发，**不触发其许可义务**（不需要取得其许可原文）。将来若并入其代码，才需先取得并随仓附许可。
3. ⚠️ **本仓未含构建产物**，因此"照本仓能编出可用二进制"这件事**未在别人的工具链上验过**。

---

## 5. 组件与许可明细（逐条带证据；**2026-10-02 复核补入**）

> 证据一律写成**相对路径**（树根 = 上游引擎源码树；本仓内文件直接写仓内路径）。
> 本节只写**实际打开过的文件**里读到的内容；拿不到原文的一律写"未核"，缺什么写在 §6。

### 5.1 基座与算法来源

| 组件 | 在本仓/引擎里的形态 | 许可 | 证据 |
|---|---|---|---|
| **NInfer**（引擎基座） | `patches/changed-files/` 是针对它写的改动；本仓**不随仓发整树** | **Apache-2.0** | 树内 `LICENSE:1-3`（与 [LICENSE](LICENSE) 同哈希）；版本 `VERSION:1` = `0.11.0-rtx3090` |
| **`tancau/ninfer-kvmem-ring`** | **代码已并入**引擎（KVMem 环五拍：打分 → 降级到主机 → 映射 → 按需搬回 → 装掩码） | **Apache-2.0** | 该仓 `LICENSE:1-3`（与引擎 `LICENSE` **逐字节相同**，sha256 `C71D239D…0AB4`）；其 `README.md` 许可段原文 `Apache License 2.0. See LICENSE.`；其 `README-RING.md` 描述的机制与本仓"五拍"一致 |
| **`kvmem-qw3`**（作者 **Di Chai**） | **含其源码改造版（8 件，`src/ops/kvmem/qw3/`，头部带 `PORTED` 声明）**，语义沿用其常驻窗口 / 检索预算 / 差量计划 | **Apache-2.0** | 树内 `src/ops/kvmem/qw3/LICENSE-kvmem-qw3.txt:2-3`（11,358 B，sha256 `CFC7749B…3D30`）；同目录 `THIRD_PARTY_NOTICES-kvmem-qw3.md` |

**上游署名（引用上游 README 原文，供署名对齐）**：

> `The base is the master of ashalliants/ninfer-3090: v0.11.0 and the multi-GPU pipeline stages, most of both written by Warlax, on the line Don-Chad/ninfer-3090 started from Neroued's NInfer.`
> `Each change keeps its author.`　·　许可段原文：`Apache-2.0, as upstream.`
> （出处：上游 `README.md` 开头贡献者段与许可段）

### 5.2 引擎树内 vendored 第三方（**本仓不分发**这些目录；列出是因为它们会编进二进制）

| 组件 | 许可 | 证据（树内路径） | 在引擎里的用途 |
|---|---|---|---|
| `third_party/cpp-httplib` | **MIT**（© 2017 yhirose） | `third_party/cpp-httplib/LICENSE:1-3` | HTTP/SSE 服务端（`src/serve/http_server.*`） |
| `third_party/ggml-quants` | **MIT**（© 2023-2026 The ggml authors） | `third_party/ggml-quants/LICENSE:1-3`；来源 commit 记于 `README.ninfer.md` | GGUF 块量化 kernel（`src/ops/linear/gguf/*`） |
| `third_party/llama-jinja` | **MIT**（同上作者）+ 同目录 **UNICODE LICENSE V3**（© 1991-2026 Unicode, Inc.） | `third_party/llama-jinja/LICENSE:1-3`、`third_party/llama-jinja/UNICODE-LICENSE:1-5` | chat template 渲染（Jinja） |
| `third_party/nlohmann`（json） | **MIT**（© 2013-2025 Niels Lohmann） | `third_party/nlohmann/LICENSE.MIT:1-3` | 全树 JSON |
| `third_party/spdlog` | **MIT**（© 2016-present Gabi Melman and spdlog contributors）；内含 bundled **{fmt}**（同为 MIT） | `third_party/spdlog/LICENSE:1-3`、`:23-25`、`third_party/spdlog/include/spdlog/fmt/bundled/fmt.license.rst:1` | 日志（`src/serve/operational_log.*`） |
| `third_party/utf8proc` | **MIT "expat"**（另含原始 utf8proc 许可段：Public Software Group e. V., Berlin） | `third_party/utf8proc/LICENSE.md:1-10`、`:30-32` | 文本 Unicode/NFC 规范化 |
| `third_party/xgrammar` | **Apache-2.0**（© 2024 XGrammar Contributors） | `third_party/xgrammar/LICENSE:1-3`、`third_party/xgrammar/NOTICE:1-3` | 结构化输出（opt-in `--structured-output`） |
| `third_party/xgrammar/3rdparty/dlpack` | **Apache-2.0** | `third_party/xgrammar/3rdparty/dlpack/LICENSE:1-3` | xgrammar 的 C-API tensor 结构 |
| `third_party/xgrammar/3rdparty/picojson` | **BSD-2-Clause**（许可**嵌在头文件里**，目录内**无 LICENSE 文件**） | `third_party/xgrammar/3rdparty/picojson/picojson.h:1-27`（© 2009-2010 Cybozu Labs, Inc. / © 2011-2014 Kazuho Oku） | xgrammar 的 JSON/schema 解析 |
| `tools/chat_templates/LICENSE` | **Apache-2.0**，附录版权行已填 = **`Copyright 2026 Alibaba Cloud`** | `tools/chat_templates/LICENSE:1-3`、`:190` | Qwen chat 模板（`qwen3_6.jinja` / `qwen3_8.jinja`） |
| `model-cards/*/LICENSE`（5 份，逐字节相同） | **Apache-2.0** | `model-cards/Qwen3.8-27B-NInfer/LICENSE:1-3`（附录版权行仍为占位符，**未填** —— 见 §6） | 引擎派生制品（`.ninfer`）的模型卡，**不是权重本体** |

**有版本记录的**：spdlog `v1.17.0`（commit `79524ddd…`）· xgrammar `v0.2.7`（commit `82505d0d…`）· dlpack commit `bbd2f4d3…` · ggml-quants / llama-jinja 各记来源 commit。
**树内无版本记录的**：cpp-httplib / nlohmann / utf8proc / picojson ⇒ **未核**（§6 第 7 条）。

### 5.3 模型权重归属（本仓**不含**权重，只写归属与核实状态）

| 模型 | 版权归属（读到的事实） | 许可 | 核实状态 |
|---|---|---|---|
| **Qwen3.8-27B**（底座） | **Copyright 2026 Alibaba Cloud** | **Apache-2.0** | 制品 `artifact-NOTICE` 逐字转述 + 模型卡 README 许可段；**上游 LICENSE 正文未在本机打开 ⇒ 未逐字核** |
| **Ternary Bonsai 2 27B**（PTQ1_0 / PQ2_0） | **Prism ML, Inc.**（`Copyright 2026-present Prism ML, Inc.`）；MTP 头与 DFlash2 适配器 = **ProCreations**；底座 Qwen3.8-27B © Alibaba Cloud | 三个来源均声明 **Apache-2.0** | **制品 NOTICE 已逐字核**；**上游仓 LICENSE 原文未打开 ⇒ 未核** |
| **GSQ-RCO IQ3_S** | 基座 `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`；本机制品打包者 `WaveCut/Qwen3.8-27B-GSQ-RCO-IQ3_S-NInfer-v3` | 记为 apache-2.0 | **未核**（两个上游仓的许可原文均未到手） |
| **Swift-RCO IQ3_S** | 上游仓 `ticeclock/Swift-Qwen3.8-27B-RCO-GGUF` | — | **未核**（该仓 LICENSE/NOTICE 原文不在本机；**不得**用同族 Swift-Bonsai-2 的 Apache-2.0 替证） |

---

## 6. 未核与待办（**合并清单**；每条写明缺什么证据）

1. **上游仓公开 URL 未核** —— 按目录名与 `VERSION = 0.11.0-rtx3090` 认门（`iamwavecut` / NInfer-all）；缺确切地址与分支。
2. **`CraneBW` / `laamaafung` 线：仅语义参照，未并入其代码** —— `kvmem_resident_pages` 的定池做法与其同向（出处注释随源码，见 §2.1），两边**没有代码交换** ⇒ 不构成再分发，**不触发其许可义务**（不需要取得其许可原文）。将来若并入其代码，才需先取得并随仓附许可。
3. **本仓未含构建产物** ⇒ "照本仓能编出可用二进制"**未在他人的工具链上验过**。
4. **curl 许可文本不在手**（我们自述"MIT 类"，未逐字核）。
5. **NVIDIA CUDA EULA 文本未随包**，且其"可再分发清单"是否覆盖我们随包的 3 支 CUDA 13 运行库**未逐条对上**。
6. **FFmpeg 许可版本口径**：随包二进制实证为 **GPL v2 or later**（configure 含 `--enable-gpl --enable-version3`），而旧文档写"LGPL/GPL" ⇒ **以 GPL 为准**；文本与本机 DLL 是否同一构建**未与上游资产逐字对齐**。
7. **`third_party` 四个组件的版本未核**（cpp-httplib / nlohmann / utf8proc / picojson）。
8. **`tools/chat_templates/froggeric_v22_5.jinja`** 标注 Apache-2.0（来源 HF 仓 `froggeric/Qwen-Fixed-Chat-Templates`，unmodified copy），**上游许可原文未核**。
9. **`model-cards/*/LICENSE` 的附录版权行仍是占位符**（`Copyright [yyyy] [name of copyright owner]`）—— 若该目录对外发布，需按 Apache-2.0 §4(a) 填明版权人。本仓**未分发**该目录。
10. **模型卡与制品的上游 LICENSE 正文**（Qwen / prism-ml / ISTA-DASLab / WaveCut / ticeclock）**均未逐字核**（见 §5.3）。
11. **`ticeclock` 与 `ukisai` 两个发布方的关系未核**（不影响许可结论，只影响署名表述）。
12. **分发包里的 `licenses\` 目录不存在** —— 旧文档曾声称"许可原文随包"，实际包内没有该目录；文本可在别处取得（见 §7），**对外分发二进制前应补回或改写声明口径**。

---

## 7. 运行时二进制（**不在本仓**；通过 GitHub Release 发布，模型权重仍不发布）

| 二进制 | 许可类型 | 许可原文是否随包 |
|---|---|---|
| `cublas64_13.dll` / `cublasLt64_13.dll` / `cudart64_13.dll` | NVIDIA CUDA Toolkit EULA 的**可再分发运行库**条款 | **未随包** |
| `avcodec-63` / `avformat-63` / `avutil-61` / `swresample-7` / `swscale-10` | **GPL v2 or later**（该构建启用 GPL 部件；二进制 configure 实证） | **未随包** |
| `libcurl-x64.dll` | curl 许可（MIT 类；二进制内含版本串 `libcurl/8.22.0`） | **未随包** |
| `ninfer-serve-86/89/120a.exe` | Apache-2.0（本引擎构建产物） | 随包（`licenses\LICENSE-ninfer-engine.txt` 形态存在过） |

> ⚠️ 这三条"未随包"**是已知待办**：对外分发这些二进制时，应把许可文本补进包内，或明确改写声明口径。
> 本仓**不含**上述任何二进制，故这些再分发义务**不在本仓触发**。
