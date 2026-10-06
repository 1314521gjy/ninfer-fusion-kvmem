---
license: apache-2.0
language:
  - zh
  - en
tags:
  - ninfer
  - kv-cache
  - kvmem
  - content-retrieval
  - paged-attention
  - prefix-cache
  - host-offload
  - long-context
  - inference-engine
  - cuda
  - sm-89
  - ada
  - windows
  - quantization
  - speculative-decoding
  - reproduction
---

# NInfer · 小显存长上下文（KVMem 环 + host-backed 复用）· 复现白皮书与引擎侧改动

> **本仓是上游 NInfer 的下游衍生（downstream derivative），不是上游官方仓**；上游出处、许可与逐文件改动量见 [`NOTICE.md`](NOTICE.md) §1 与本站的"我方工作"一节。

> # ⚠ 初期版本 · 含大量未解决 bug
>
> 本仓是**融合引擎的初期版本源码**：能启动、能出结果，但**不保证正确性与稳定性**；**不建议进行商用，仅可进行个人使用和探索**（源码许可仍是 Apache-2.0，见 `LICENSE`；这是使用建议，不是许可附加条款）。
>
> **Bug 账本**（B01–B20，合并本机实测 / 代码自述 / B 站社区反馈，去重后按重要性排序、**只列未关闭项**）见 `已知问题-初期版本.md`。

> ## ⚠️ 发布声明（请先读）
>
> | | |
> |---|---|
> | ✅ **发布** | **复现白皮书**（`复现白皮书-NInfer-KVMem环-20261002.md`）· 引擎侧源码改动（`patches/`）· 技术文档（`docs/`）· 自建工具（`tools/`）· 判据脚本（`verify/`）|
> | ❌ **不发布** | **任何模型权重**，以及由权重派生的 `.ninfer` 制品（**引擎二进制已改为通过 GitHub Release 提供**；CUDA/FFmpeg 运行库不在本仓）|
> | 🧱 **基线** | 上游 **NInfer**（Apache-2.0）源码树，`VERSION = 0.11.0-rtx3090` —— `patches/` 是针对它写的 |
> | 📄 **许可** | 本仓内容为 **Apache License 2.0**（**不覆盖**权重）；第三方署名与证据见 [`NOTICE.md`](NOTICE.md) |
> | 🐙 **GitHub 仓** | <https://github.com/1314521gjy/ninfer-fusion-kvmem>（源码 + Release 二进制）|

---

> **引擎源码树（整棵）在 src-tree/fusion-engine-src/**；我方改动逐文件在 patches/changed-files/。
>
> 这是**融合引擎**源码（NInfer v0.11.0 上游 + 我方改动整合），**不是上游 `master` 原树**；上游出处见 `NOTICE.md`，我方改动逐文件见 `patches/changed-files/`。
>
> 启动崩溃 x18c729 的根因与修法见 根因与修法-0x18c729-20261003.md。

## 0. 我方工作与贡献（先看这条）

本仓是**融合引擎**（NInfer-all 基座 + 我方引擎线工作）的源码与复现记录。逐文件 SHA256 比对上游基线：
**2,327 件逐字节相同、92 件为我方修改、44 件为我方新增、0 件缺失**（复算脚本 `verify/reconcile-vs-upstream.ps1`，清单 `patches/changed-files.txt`）。

我方做的工作，按类别列全：

1. **KVMem 环（新增 `src/ops/kvmem/` 44 件 + 相关改动）**：让"设备池 < 逻辑上下文"成为合法配置 ——
   内容打分选页 → 降到主机 KV → 映射回设备 → 隐藏页装掩码 → 按需搬回。
   落点：`src/ops/kvmem/`（`kvmem_score.h`、`kvmem_select.*`、`kvmem_retrieve*`、`kvmem_window_assembly.*`、
   `mean_k_index*`、`raw_k_shadow*`、`kvmem_port_bridge.*`）与 `program/storage/context.cpp`、`core/paged_kv_cache.*`。
2. **host-backed 复用 + 惰性领用**：设备池装不下的检查点整段登记、按需物化（`transactions/capture.cpp`、
   `planning/pressure.cpp`、`transactions/materialization.cpp`）。
3. **定容与不变量**：`host_pages + pool_pages >= logical_pages`；页 = 64 token；池页 = 窗口/64 + prefill 块/64 + 8 松弛；
   装不下就在启动期拒绝并报出所需页数（`planning/startup.cpp`）。
4. **低显存池实验**：显存池 4,000 token（63 页）下，多针题面第 1 轮 6/6、第 2 轮 6/6；
   红控 `TAIL=256` 第 2 轮掉到 2/6 —— 判据能变红。
5. **长题验证**：94,698 token（窗口的 23.5 倍）3/3 + 3/3；命中不稳的反例同样留档（见 `已知问题-初期版本.md`）。
6. **门禁与消融**：17,920 组门禁臂全绿（含 `noretr` 0/6 负控）；SHARE=25 预算下 A/B：词法选页 4/6 vs 内容打分 6/6。
7. **启动崩溃 0x18c729 的根因定位与修法**（30 / 50 系）：混合时期 obj 导致 `GenerationService` 跨编译单元布局不一致；
   改为每架构空目录清编，三支件哈希见 `根因与修法-0x18c729-20261003.md`。
8. **启动期可观测与降级**：构造期逐步日志；core 构造失败自动回退普通 core 继续服务，不再裸崩溃。
9. **量化与内核线**：PTQ1 / GSQ 档位的 T2 内核与派发改动（`src/ops/linear/t2/`、`softmax_attention/dense/causal_cache/` 等）。
10. **速度口径与实测**：PTQ1（mtp）194.6–196.8 tok/s；PQ2（dflash2, draft 12）571.9 tok/s / 2.2 s。
11. **交付工具**：按卡选引擎（`pick-engine.bat`）、套件自检、模型件自检、部署模拟、哈希门禁。

**要自己编？** → `编译指南-怎么编.md`：工具链版本（CUDA 13.3 / MSVC BuildTools / vcpkg 清单依赖 / 驱动 ≥580）、逐架构的 `cmake` 命令行、三条硬规矩（一架构一空目录 · 编译不需要模型 · 首次 1–2 小时）与常见编不过的原因。

**实测回执与反馈**（群内真机 + 本机读数，全部带日期与出处）→ `实测回执与反馈.md`：显存池 4,000 token 跑 96K 上下文命中 6/6 + 6/6（红控掉到 2/6）、94,698 token 长题 3/3 + 3/3、解码峰值 571.9 tok/s、群内 RTX 5080 上 104,991 token 题面命中中段针。

**与上游的逐项对账**（上游完全没有、本树有的能力；已解决的问题与判据）→ `我方-新增能力与已解决问题.md`：
命令行开关我方多 17 个、环境变量多 39 个、上游 0 个开关被我方删除；`kvmem`/`t2_ptq1` 关键词在上游命中 **0** 个文件。

**上游与许可（如实说，逐条带判据）**：引擎基座是上游 NInfer-all（Apache-2.0，2,327 件逐字节相同）；
树里**确实含他人代码**，逐文件清单在 `NOTICE.md` §2.1：`kvmem-qw3` 8 件（Di Chai，Apache-2.0，我方改造版，头部带 `PORTED` 声明，许可与 notices 逐字节随树）；
`tancau/ninfer-kvmem-ring` 4 件逐字节相同（Apache-2.0）+ 65 件共用改动面；`CraneBW` / `laamaafung` 线是 `kvmem_resident_pages` 定池做法的**语义参照**（出处注释随源码）—— **未并入其代码，不构成再分发、不触发其许可义务**。
上面 §0 的 1–11 条是在这些基础之上**我方做的事情**（移植、接入、改造、定容、实验、修崩溃、打包）—— 两件事分开看，互不抵消。

---

## 1. 这个仓解决什么问题

一句话：**让"显存很小"和"上下文很长"不再互相打架，而且每一轮不必把上下文重填一遍。**

引擎原本的不变量是「KV 设备池 == 逻辑上下文 == 提示上限」。本仓的改动把它拆开：

| 改成了什么 | 判据（实测读数，口径见白皮书 §9）|
|---|---|
| **设备池 ≪ 逻辑上下文**（池 280 页 = 17,920 token，逻辑上限 262,144）| 启动行 `capacity \| KV 17,920 tokens, k8v4, explicit \| pages 280/4,096` |
| **小池也不答错**（池 4,000 token = 63 页，题面 42k）| 多针 turn1 **6/6** + turn2 **6/6**；同支二进制把检索窗口改回 256 行 ⇒ turn2 **2/6**（红控） |
| **长题面照样答对**（超池 23.5×）| 94.7k 题面 + 63 页池 ⇒ 两轮 **3/3 + 3/3**，零静默错 —— ⚠️ **只在"显式 `QUERY_TAIL=64` + 打分确认在跑"的那条臂上成立**；同日另有一条"不设 env"的同形臂读到 `hit=0 / SILENT_WRONG=2`，**未定论**，反例与边界见白皮书 §9.1b |
| **每一轮不全量重填**（同会话追问）| 第二轮 **复用 100%（cache 80,075）**，TTFT **70,829 ms → 126.5 ms** |
| **按卡自适应**（同一支二进制按卡选调度）| 启动行 `device profile <class>: N routed keys`；`--device-profile auto/off/calibrate` |

**这不是"显存优化技巧"**：池变小是因为**工作集**变小（常驻窗口 + 一个预填块），装不下的页走宿主层、
按需搬回 —— 机制、公式与判据见白皮书 §2–§3 与 `docs/01-部署与编译总白皮书.md` §6。

---

## 2. 目录

```
复现白皮书-NInfer-KVMem环-20261002.md   ★★ 第一入口：机制 / 复现步骤 / 判据与负控 / 实测读数 / 协议署名
README.md                               本页
NOTICE.md                               署名、许可与我方改动声明（协议层权威）
LICENSE                                 Apache License 2.0（仅覆盖本仓内容）
patches/                                我们对上游引擎的改动（38 个源文件 + 逐文件说明）
  README-改动说明.md                    ← 基线、判据、按主题分类、诚实清单（做改动的人先读这个）
  changed-files/                        改动后的完整文件，路径与引擎源码树一一对应
  changed-source.txt38 行，机器可读
  changed-files.txt136 条（M 92 + A 44），逐文件对账原始清单
docs/                                   对外口径的技术文档
  00-三档口径与读数.md                   实测读数：速度 / 显存 / 长上下文复用
  01-部署与编译总白皮书.md                从零到出包：配置项逐条、构建、ring 五开关、验收判据
  02-排错手册 · 预案与处方.md             症状 → 判据 → 处方
  03-基础部署后的调优方案.md              三个旋钮（窗口 / 检索 / 宿主层）与"改完怎么验"
  04-卡死与循环的防治.md                  ★ 空交付 / 固定点 / 半截 / 平台守卫：为什么必须在平台层拦、怎么拦
  05-已知问题与禁忌.md                   ★ 照用会炸的那几条（每条带读数）
  06-Bug手册.md                          ★ 未解决 / 已修但老包还带 / 绕法 / 红线
  08-从零复现兜底.md                     只想起服务、不想编
  09-声明与链接.md                       第三方清单 / 许可 / 证据（协议层索引）
  11-按卡差异速查.md                     档位表（逐字取自引擎 device_profiles.json）与架构边界
  12-给接收方-agent-的操作手册.md         ★★ 接手部署的 agent 看的顺序清单
  判据.txt                               就绪判据的形状（日志逐字）
  docs/方案/                             我方自研路线的评估与立项（内部视角，含本机路径）
tools/                                  自建工具（都不依赖引擎，读文件即可用）
  peek_container.py                      读 .ninfer v3 容器布局（头 / 清单 / 组件 / 载荷）
  component_bytes.py                     按组件算字节账
  check_strip.py                         装载安全性闸：uses 有没有被饿死（引擎免跑）
  check_refs.py                          目录 JSON 的引用完整性（悬空对象 id 只在引擎里才炸）
  strip_components.py                    重打包容器：去掉某些组件（用上游 writer，布局重算）
  自检-模型件.ps1                        ★ 模型体检：完整性 + 带什么投机头 + 你这张卡该用哪个 --spec
  kit-sha256sums.ps1                     包校验清单增量生成（只重算变了的文件）
  smoke-mtponly.ps1                      起-问-复用的冒烟（含"答案退化"与"第二轮 cache≥90%"两道判据）
verify/                                 判据脚本（每个都有"怎么变红"的负控，见白皮书 §10）
  gate-内测包-清单与形状.ps1              包清单双向校验 + 引擎/DLL/启动器形状（PQ2 档）
  gate-ptq1档-清单与形状.ps1             同上，PTQ1 档
  自检-引擎与卡匹配.ps1                   本机卡能不能跑这个包（架构边界）
  verify-arch-engine.ps1                 本卡验收：起服务 + 过池告警臂 + 题面中段针
  verify-kit-manifest.ps1                清单逐文件校验
```

---

## 3. 四条技术线（这个仓的全部内容）

| 线 | 做了什么 | 落在哪 |
|---|---|---|
| **A. 存储解耦** | 「设备池 == 逻辑上下文」这条不变量拆开：池按工作集定、溢出的页进宿主层、按需搬回 | `patches/changed-files/src/core/paged_kv_cache.*`、`…/program/storage/*`、`…/state/decoder_state.*` |
| **B. host-backed 复用 + 惰性领用** | 池装不下的检查点**整段登记、按需物化** ⇒ 长题面的第二轮照样复用 | `…/program/transactions/{capture,materialization}.cpp`、`…/program/planning/pressure.cpp` |
| **C. 按卡自适应档位** | 同一支二进制按「硬件类 + SM 数」自动选实测最优调度；`off` 一条命令逐位等价回退 | `…/runtime/engine/device_profiles.json`、`…/serve/serve_options.cpp` |
| **D. 内容检索与查询窗口** | 页按**内容相关性**排序搬回（而非词法/先进先出）；检索查询 = **当前问题的 token span**（默认 64 行）；打分**默认开**且可显式关闭做负控 | `patches/changed-files/src/ops/kvmem/*`、`…/models/qwen3_5/execution/*`、`…/program/prefill.cpp`、`…/frontend/*`、`…/serve/*` |

---

## 4. 怎么用

### 4.1 想复现（推荐从这里开始）
读 **`复现白皮书-NInfer-KVMem环-20261002.md`**：§8 是逐字复现步骤（取基线 → 覆盖 → 构建 → 起服务 → 验收 → 短测），
§10 是每条结论的判据与负控，§9 是实测读数。

### 4.2 想做引擎侧改动
1. 取上游 **NInfer**（Apache-2.0）源码树，基线 `VERSION = 0.11.0-rtx3090`；
2. 把 `patches/changed-files/` 里的文件**按同名路径覆盖**上去；
3. 编译（工具链口径见白皮书 §8.2 与 `docs/01`）；**这一步我们只在自己机器上验过**（见 §6）。

### 4.3 只想把服务跑起来
看 **`docs/12-给接收方-agent-的操作手册.md`**（第一入口）与 `docs/08`。
⚠️ 本仓**不含引擎二进制**，你需要另有一份构建产物。

### 4.4 只想读结论
白皮书 §0 / §9 → `docs/05`（禁忌）→ `docs/04`（思考循环）三处就够。

---

## 5. 最短验收清单（照用）

```
1) 起服务必须看到三行：
   [ring] content scoring ON by default (the ring is configured): …
   INFO  engine ready | <model> | total <N>s | weights <X> GiB
   INFO  capacity | KV 17,920 tokens, k8v4, explicit | pages 280/4,096 | runtime <X> GiB | free <Y> GiB
2) 发过请求后必须有 ≥1 行：kvmem_score: SELECT label=text_prefill_chunk … query_tokens=64
   （题面小于窗口时本来就不产生 SELECT —— 那不是故障；超窗却没有 SELECT 才是故障）
3) 一条"数数字"短测（1000 进 / 1000 出），去引擎控制台读：
   req#1 done … TTFT <N> ms | total <N>s | prefill <N> tok/s | decode <N> tok/s
4) 小池臂的负控：把检索窗口改回 256 行 ⇒ 小池多针 turn2 必须掉下来（否则说明判据测不出东西）
```

---

## 6. 边界与未验（**不许当已验读**）

1. **我们只在一张卡上端到端验过**：RTX 4080 SUPER / sm_89 / 32 GB / Windows。其它卡**未验**。
2. `patches/` 是"38 个改后的完整文件"，**不是逐行 diff**；行级差异请自己用 `git diff --no-index` 生成。
3. **构建未随仓发布**：不提供 `.exe`/`.dll`，也不承诺在别人的工具链上一定能编过。
4. `docs/方案/` 里的文档带**本机路径**（`（本机构建根）\...`），是内部视角的原始记录，读时忽略路径即可。
5. 文档里的"三档"读数对应**我们自己构建的量化制品**（2.125 bpw 三元 / GSQ-RCO IQ3_S / Swift-RCO IQ3_S）；
   **这些制品与权重都不在本仓**，读数只用于说明引擎行为。
6. 已知问题与禁忌逐条写在 `docs/05-已知问题与禁忌.md`；**未解决 / 还没修完 / 老包还带着的写在 `docs/06-Bug手册.md`**（含绕法与红线），都不隐藏。
7. **设备池下限**：已测最小 4,032 token（63 页）；更小**未测**。**低池验收只在 `k8v4` 上做过**，其它 dtype 未做。
8. **跨页缝截断**（关键串跨两页交界、只搬回一侧）是**已知未修**的固有性质。
9. **2 分钟以上的冷预填仍可能在连接层被掐**：根因**未定位**，keep-alive/超时是**缓解不是修复**（白皮书 §7.2 / §11）。

---

## 7. 引用与许可

- 引擎基座：**NInfer**（Apache-2.0，基线 `0.11.0-rtx3090`）；本仓的改动声明见 [`NOTICE.md`](NOTICE.md)。
- KVMem 环的移植来源：**`tancau/ninfer-kvmem-ring`**（Apache-2.0，作者声明，本机逐字核过）。
- 策略层算法语义参照：**`kvmem-qw3`**（作者 Di Chai，Apache-2.0）—— **本仓含其源码改造版（`src/ops/kvmem/qw3/` 8 件）**，语义沿用其常驻窗口 / 检索预算 / 差量计划
  （许可原文在源码树内 `src/ops/kvmem/qw3/LICENSE-kvmem-qw3.txt`）。
- 引擎内第三方库（cpp-httplib / ggml-quants / llama-jinja / nlohmann / spdlog / utf8proc / xgrammar 等）
  与运行时二进制（CUDA / FFmpeg / libcurl，**不在本仓**）的清单与证据：`NOTICE.md` 与 `docs/09-声明与链接.md`。
- **模型权重**（Qwen 体系底座与各量化制品）版权归其各自作者与上游，按其各自许可发布；**本仓不包含权重**。

---

## 8. 引用本仓

```
NInfer · 小显存长上下文（KVMem 环 + host-backed 复用）· 复现白皮书与引擎侧改动, 2026-10-02.
https://modelscope.cn/models/shensanshu/ninfer-master-shensanshu-kvmem
```
