# NInfer · 小显存长上下文复现白皮书
## KVMem 环（内容检索）+ host-backed 复用 + 按卡自适应
**版本 2026-10-02 · 配套仓：`shensanshu/ninfer-master-shensanshu-kvmem` · 许可：Apache-2.0（仅覆盖本仓内容，不覆盖任何模型权重）**

---

## 0. 一句话与三条线

**一句话**：把引擎里「**设备 KV 池 == 逻辑上下文 == 提示上限**」这条硬不变量**拆开** —— 让显存只装**工作集**，
装不下的 KV 页落宿主层、按需搬回，并且**每一轮不必把上下文重填一遍**。

本仓的全部技术内容就是三条线：

| 线 | 做了什么 | 一句话判据 |
|---|---|---|
| **A. 存储解耦（KVMem 环）** | 设备池 ≪ 逻辑上下文成为**合法配置**；溢出的页降级到宿主层、按内容相关性检索搬回、对未搬回的页打掩码 | 启动行 `capacity \| KV 17,920 tokens, k8v4, explicit \| pages 280/4,096` |
| **B. host-backed 复用 + 惰性领用** | 池装不下的检查点**整段登记、按需物化** ⇒ 长题面的第二轮照样命中前缀复用 | 第二轮 `cache` 命中率与 TTFT 塌缩（见 §9） |
| **C. 按卡自适应档位** | 同一支二进制按「硬件类 + SM 数」自动选实测最优调度；`--device-profile off` 可逐位等价回退 | 启动行 `device profile <class>: N routed keys` |

**这不是"显存优化技巧"，而是配置空间的重定义**：池变小是因为**工作集**变小（常驻窗口 + 一个预填块），
不是一个 trade-off 被调松。

---

## 1. 基线与边界（**先看这一节，决定你能不能照用**）

### 1.1 基线

| 项 | 值 |
|---|---|
| 引擎基座 | **NInfer**（Apache-2.0），本机树的 `VERSION = 0.11.0-rtx3090` |
| 我方改动范围 | **38 个源文件**（KVMem 环与启动期修复那一批；整棵树对上游基线的重算差异为 **92 件修改 + 44 件新增**，见 `NOTICE.md` §2。逐文件完整文件放在 `patches/changed-files/`，路径与引擎源码树一一对应） |
| 改动清点判据 | 2,424 件里 **2,385 件与基线逐字节相同**，34 件为真源码改动（见 `patches/README-改动说明.md`） |
| 硬件口径（我们唯一端到端验过的形状） | RTX 4080 SUPER / sm_89 / 32 GB / Windows x64 |
| 本仓**不含** | 模型权重、`.ninfer` 制品、引擎二进制（`.exe`/`.dll`）、CUDA 运行库、FFmpeg |

> ⚠️ **本仓不发布构建产物**。`patches/` 是"改后的完整文件"，不是逐行 diff；行级 diff 请自己生成：
> `git diff --no-index --stat "<上游树>" "<你的树>"`。

### 1.2 谁该读哪一节

| 你想干什么 | 直接跳 |
|---|---|
| 只想知道结论 | §0 → §9（读数）→ §11（边界与未验） |
| 想复现 | §8（逐字步骤）→ §10（判据与负控） |
| 想做引擎侧改动 | §2–§7（机制）→ `patches/README-改动说明.md` |
| 关心协议与署名 | §12 与 `NOTICE.md` |

---

## 2. 总体思路

### 2.1 原始不变量为什么要拆

原引擎的 KV 规划把三件事绑成一个数：**设备池 = 逻辑上下文 = 提示上限**。推论是：
"上下文能开多长"直接由"显存装得下多少 KV"决定 —— 于是 27B 级模型在 12–16 GB 卡上只能跑很短的上下文，
或者必须牺牲精度（KV 量化到极低比特）来换长度。

**拆开之后的三条推论**（这是本仓的全部机制来源）：

1. **池只需要覆盖工作集**：一轮解码真正"反复读"的，是**常驻窗口**（最近若干 token）+ 一个**预填块**；
   再留几页余量给检索搬回。其余页**不必同时驻留**。
2. **溢出的页要有地方去、有办法回来**：去宿主层（可换页的 pinned 内存），回来要**按相关性**而不是按先进先出。
3. **回来之后必须让注意力"看不见"没回来的页**：否则等于假装有上下文 —— 这就是掩码（mask）与
   `kvmem_score` / `mask hidden` 这类**可读日志**存在的原因。

### 2.2 池页数的算术（可直接采用的公式）

```
池页数 = 常驻窗口 / 64  +  预填块 / 64  +  余量(8 页)
例：16384/64 + 1024/64 + 8 = 280 页 = 17,920 token
逻辑上下文 262,144 token ⇒ 需要 4,096 页 ⇒ 其余 3,816 页落宿主层
```

宿主层是**真实内存且被钉住**（pinned）：本机实测换算 **每 1 token 逻辑上下文 ≈ 25 KiB 主机池**
⇒ 256K≈6.6 GiB · 512K≈13.2 GiB · 1M≈26 GiB。**装不下就是启动期拒绝，不会静默降级**：

```
主机池页数 + 设备池页数 ≥ 逻辑上下文页数        （启动守卫，见 patches/.../planning/startup.cpp）
```

---

## 3. KVMem 环：机制

### 3.1 单位与数据结构

| 概念 | 值 / 说明 |
|---|---|
| 页 | **64 token**（`kPagedKVPageSize`） |
| 设备池 | `--kv-capacity <tokens>`，物理上就是页表 + 显存 KV 区 |
| 宿主池 | `--host-kv-mib <MiB>`，pinned 主机内存里的 KV 副本 |
| 检索窗口 | **当前问题**（默认 64 行查询），不是"最近 256 行" |
| 索引 | 每页的**均值 K**（mean-K）索引：`kvmem_index: installed layers=16 heads=4 head_dim=256 blocks=<池页数>` |
| 打分 | query-conditioned 点积，**对查询行求和**，契约 `sum_b score[b] == n_query_tokens` |

### 3.2 一轮里发生的四件事

1. **打分（content scoring）**：拿当前问题的查询向量，与索引里每页的 mean-K 做点积 → 页相关性排序。
   日志：`kvmem_score: ARMED capacity_blocks=… layers_total=… heads=…` 与
   `kvmem_score: SELECT label=text_prefill_chunk n_blocks=… query_tokens=64 span_mode=…`。
2. **降级（demote）**：本轮不进常驻集合的页写进宿主层（保留 K 的原始副本：`raw_k_shadow: harvest installed …`）。
3. **映射 + 搬回（restore）**：按排序结果把最相关的页搬回设备池，上限是「池 × 50%」这类每轮预算。
4. **掩码（mask）**：没搬回来的页在注意力里被屏蔽。日志：`mask hidden: which=text site=mapped span=… resident=… hidden=… mid_hidden=…`。

### 3.3 关键设计点：**检索窗口 = 当前问题**

在"池远小于题面"的形状下，用**题面末尾 N 行**当查询会把检索带偏（末尾往往是格式尾巴）。
本仓的做法：**用最后一个用户轮的 token span 当查询**；若该 span 不可用（太长 / 为空 / 越界）则**回落到
"块内末尾 64 行"**规则。这是一条**有负控的结论**：

| 臂 | 池 | 题面 | 结果 |
|---|---|---|---|
| **交付口径（检索窗口 = 问题）** | 4,000 token（63 页） | 多针题面 | turn1 **6/6** + turn2 **6/6** |
| 红控（`NINFER_TERNARY_KVMEM_SCORE_QUERY_TAIL=256`） | 同上 | 同上（同支二进制） | turn2 **2/6** |
| 更大池交付门禁 | 17,920 | 多针 / 不相交问句 / fits | 全绿，掩码 0 行 |

> **"判据会变红"这件事是被验过的** —— 同上二进制只改检索窗口就掉到 2/6，所以 6/6 不是"碰巧"。

### 3.4 内容打分**默认开**（这条曾是把结论搞错的根因）

有一轮交付里，启动行宣称打分已开、但**答的是填充数字**：根因是**主开关**（`NINFER_TERNARY_KVMEM`）
默认关 ⇒ 索引与打分都没跑 ⇒ 静默退回**词法排序**。修法与语义：

- **显式值优先**（`=0` 关）；**不设**时：只要配了环（`NINFER_KV_WINDOW > 0`）就**默认开**；
- 负控：`NINFER_TERNARY_KVMEM=0` ⇒ 启动行改为 `the content scorer was EXPLICITLY DISABLED`；
- 可读证据：发过请求后必须有 **`kvmem_score: SELECT`** 行。`SELECT = 0` 有两种含义，**必须区分**：
  - **题面很短（小于窗口）**：没有超窗的页需要检索 ⇒ 本来就不产生 SELECT（例如 71 token 题面、窗口 16,384）；
  - **题面超窗却没有 SELECT**：打分没跑 ⇒ 退回词法排序 ⇒ **小池下会静默答错**（这是要抓的故障）。

### 3.5 环的五个环境变量（缺一不可）

| 变量 | 作用 |
|---|---|
| `NINFER_KV_WINDOW` | 常驻窗口（也是"环是否启用"的判据） |
| `NINFER_KV_RETRIEVE` | 每轮检索/搬回预算 |
| `NINFER_KV_RING=1` | 启用环 |
| `NINFER_HOST_PAGEABLE=1` | 宿主层用可换页内存 |
| `NINFER_KV_REUSE_HOSTBACKED=1` | 允许复用宿主层支撑的前缀 |

> ⚠️ **静默降级警告**：只给 `NINFER_KV_WINDOW` + `NINFER_KV_REUSE_HOSTBACKED` 也能启动、缓存显示 ~99.9%，
> 但**长题面会答错**（中段内容不可见，零错误行）。这五个是"一个配置"，不是"五个可选优化"。

---

## 4. host-backed 复用 + 惰性领用

长题面的成本主要在**预填**。本仓让"池装不下的检查点"**整段登记**到宿主层，并在下一次同前缀请求时
**按需物化**（不是一次性全搬回来）。效果（80,063 token 题面、同一会话追问）：

| 轮次 | 前缀复用 | TTFT |
|---|---|---|
| 第一轮（冷） | — | 70,829 ms |
| **第二轮（同会话追问）** | **100%（cache 80,075）** | **126.5 ms** |

相关文件：`patches/changed-files/src/models/qwen3_5/program/transactions/{capture,materialization}.cpp`、
`…/program/planning/pressure.cpp`。

---

## 5. 按卡自适应档位

同一支二进制在不同卡上自动选路：首次在陌生卡上启动会跑一次**设备校准**，把该卡的路由标定写进用户缓存；
之后按该卡最优路由走。开关：

```
--device-profile auto        # 默认：有标定用标定，没有就标定一次
--device-profile calibrate   # 强制重标
--device-profile off         # 关闭（逐位等价回退，用来做 A/B 的基座）
```

启动行判据：`device profile <class>: N routed keys (calibration)`。
标定数据在内置表 `patches/changed-files/src/runtime/engine/device_profiles.json`（本机标定项随仓发布）。

---

## 6. 量化档与投机解码（决定"装得下"和"跑得快"的两件事）

### 6.1 三档制品的组件形状

| 档 | 制品（本仓**不含**权重） | 组件 | 用途 |
|---|---|---|---|
| PTQ1_0 | `bonsai2_27b_ternary_ptq1_native_mtp.ninfer` | text + **mtp**（无 proposal 头、无视觉塔） | 最小显存；RTX 30/40/50 系 |
| PQ2 瘦身 | `Ternary-Bonsai-2-27B-ninfer-v3-mtponly.ninfer` | text + mtp | **≤10 GB 显存**的唯一选择 |
| PQ2 完整 | `Ternary-Bonsai-2-27B-ninfer-v3.ninfer` | text + mtp + **dflash2** + proposal + vision | 最快 |
| GSQ / Swift | `gsq_rco_iq3_s_dflash2_prop.ninfer` / `swift-qwen38-rco-iq3s.ninfer` | text + mtp + dflash2 + proposal + vision | 大卡（光权重 13.4 GiB） |

**投机头与 `--spec` 是配对的，拿错就是"起不来"而不是"慢"**：
`--spec dflash2 --lm-head-draft` 需要 proposal 头；只有 mtp 的制品必须 `--spec mtp`，
加了 `--lm-head-draft` 会 `FATAL … selected proposal head is absent from artifact`。
**不确定就用模型体检工具读文件**（`tools/自检-模型件.ps1`，它只读文件、不启动引擎）。

### 6.2 投机窗口（`--draft-tokens`）的**收益是有条件的**

| 语料 | 窗口 4 | 窗口 12 | 说明 |
|---|---|---|---|
| 数数字（可预测） | 278 tok/s | **600 tok/s** | 草稿几乎全中，越深收益越大 |
| 随机四位数（不可预测） | 265 tok/s | **184 tok/s** | 每颗草稿都要验证 ⇒ **更深更慢** |

同一条规律在推理型长思考上更极端：本轮实测 GSQ + `reasoning xhigh` + `--draft-tokens 10`，
**混合投机接受率只有 28.8%**（34,292/119,215），解码均值 82.9 tok/s —— 思考文本不可预测，
投机窗口的收益被吃掉。**结论：`--draft-tokens` 要按任务定，别把某一任务的数当通用值。**

### 6.3 `--kv-dtype` 与掩码接线

交付口径 `k8v4`（K=FP8 行 256 + V=NVFP4 组 16）。已知禁忌：**行粒度的缩放**（如 `fp8`/`rk8v4`/`int8`）
在部分制品上会崩（只输出 2 个 token 就 EOS，四次复现）。**小池下的验收只在 `k8v4` 上做过**，其它精度
"掩码接线"验过、**低池验收未做**（见 §11）。

---

## 7. 服务层与工程化

### 7.1 OpenAI 兼容面

| 端点 | 用途 | 判据 |
|---|---|---|
| `GET /v1/models` | 探活 | `200` + `context_window` / `modalities.vision` |
| `POST /v1/chat/completions` | 生成（流式 / 非流式） | 流式可看 `delta.content` / `delta.reasoning_content` |
| 请求键 | `reasoning_effort`: `none\|minimal\|low\|medium\|high\|xhigh\|max`；`post_thinking` 对象 | 与源码 `serve/openai_chat_request.cpp` 一致 |

### 7.2 连接层（一条本地化的工程经验）

非流式响应在**预填期间一个字节都不发**（`serve/openai_chat_http.cpp`），而 HTTP 服务端此前的
keep-alive / read / write 超时是库默认的 **5 s** —— 于是"2 分钟以上的冷预填"会表现为**客户端连接被关闭**，
而引擎本身仍在正常预填、**零错误行**。缓解（已进交付件）：

- 服务端 keep-alive 窗口 5 s → **120 s**，读/写超时 → **300 s**；
- Windows 侧补上 Linux 构建一直有的 TCP keep-alive（`SO_KEEPALIVE` + `SIO_KEEPALIVE_VALS`，10 s / 3 s）。

> **缓解 ≠ 已修**：根因**未定位**（见 §11）。出现该现象时，**不要逐字节重发同一条超池题面**
> —— 有一条已知路径会把实例打成"此后所有请求 503"。

### 7.3 前缀复用与 `--max-shared-prefixes 0`

开着共享前缀发布时，「题面页数 > 设备池页数 **且逐字节重发同一条题面**」会抛
`active KV snapshot full page is not stable`，此后该实例所有请求 503。**关掉零代价**
（每轮走私有通道，实测 80k 题面仍复用 100%）⇒ 交付启动器一律带 `--max-shared-prefixes 0`。

---

## 8. 复现步骤（逐字）

### 8.1 取基线 + 覆盖改动

```bat
:: 1) 取上游 NInfer（Apache-2.0）源码树，本仓的改动针对 VERSION = 0.11.0-rtx3090
:: 2) 把本仓 patches\changed-files\ 里的文件按同名路径覆盖上去
xcopy /E /Y patches\changed-files\* <上游树>\src\
:: 3) 行级 diff（可选）
git diff --no-index --stat "<上游树>" "<你的树>"
```

> 覆盖范围只涉及 `src\core\`、`src\models\qwen3_5\`、`src\ops\`、`src\runtime\engine\`、`src\serve\` 五处，
> 见 `patches\changed-source.txt`（34 行）。

### 8.2 构建（工具链口径）

- Windows x64 + MSVC + CUDA 13.x + CMake + Ninja + vcpkg（`vcpkg.json` 在树根）；
- **一个架构一份二进制**：`sm_89`（RTX 40）/ `sm_86`（RTX 30）/ `sm_120a`（RTX 50）。
  架构不匹配**不会回退**（cubin 里没有别的 PTX），是"起不来"而不是"慢"；
- 增量构建注意：本树的**头依赖库不可靠**（`ninja -t deps` 多为 `#deps 0`），
  改了结构体头（如 `prepared_prompt.h`）必须**整片重编该模块**，否则会**静默漏编**。

### 8.3 起服务（模板 argv）

```bat
set NINFER_KV_WINDOW=16384
set NINFER_KV_RETRIEVE=8192
set NINFER_KV_RING=1
set NINFER_HOST_PAGEABLE=1
set NINFER_KV_REUSE_HOSTBACKED=1

ninfer-serve-89.exe "models\<你的制品>.ninfer" ^
  --host 127.0.0.1 --port 8091 --model-id qwen3.8-27b ^
  --max-context 262144 --kv-capacity 17920 --kv-dtype k8v4 --host-kv-mib 16384 ^
  --prefill-chunk 1024 --spec dflash2 --draft-tokens 12 --lm-head-draft ^
  --default-max-tokens 32768 --default-reasoning-effort none --max-concurrency 1 ^
  --max-shared-prefixes 0 ^
  --presence-penalty 0 --temperature 0.7 --top-p 0.9 --top-k 20
```

**启动必须看到的三行**（少一行就别继续）：

```
[ring] content scoring ON by default (the ring is configured): …
INFO  engine ready | <model> | total <N>s | weights <X> GiB | CUDA sync auto
INFO  capacity | KV 17,920 tokens, k8v4, explicit | pages 280/4,096 | runtime <X> GiB | free <Y> GiB
```

### 8.4 验收（判据 + 怎么变红）

| # | 判据 | 怎么变红（负控） |
|---|---|---|
| 1 | `capacity \|` 行数字与你的 argv 一致 | 改 `--kv-capacity` ⇒ 该行必须跟着变 |
| 2 | 启动期 `content scoring ON by default` | `NINFER_TERNARY_KVMEM=0` ⇒ 变成 `EXPLICITLY DISABLED` |
| 3 | 发过请求后 **`kvmem_score: SELECT` ≥ 1 行**（仅当题面超窗） | `NINFER_TERNARY_KVMEM_SCORE=0` ⇒ 退回词法排序 |
| 4 | 小池 + 超窗题面：多针答案**不丢中段** | 检索窗口改成 256 行 ⇒ turn2 从 6/6 掉到 2/6 |
| 5 | `mask hidden … mid_hidden` 在超池时非零、在装得下时为 0 | 池 ≥ 题面 ⇒ `hidden` 必须为 0 |

### 8.5 短测（移植后 15 分钟内该做的唯一一件事）

**数数字语料，1000 token 进 / 1000 token 出**：题面 = `1 2 3 … 300` 空格分隔 + 一句"接着往下数、只输出数字"，
`max_tokens=1000`、`temperature=0`；读数**去引擎控制台读**这一行：

```
req#1 done | openai-chat | output limit | prompt <N> | output 1000 | cache 0 (0.0%) |
   TTFT <N> ms | total <N>s | prefill <N> tok/s | decode <N> tok/s | mtp accepted <a>/<b>
```

期望量级（本机 4080 SUPER / 池 17,920 / 数数字 1000 出）：

| 档 | 题面 | TTFT | 预填 | 解码 | 总耗时 |
|---|---|---|---|---|---|
| PTQ1_0 (`--spec mtp`) | 1,133 | 565–578 ms | ~2.0k tok/s | 194.6–196.8 tok/s（mtp 接受 88.1%） | 5.7 s |
| PQ2 瘦身 (`--spec mtp`) | 1,133 | 344–350 ms | 3.25–3.30k tok/s | 237.9 tok/s | 4.6 s |
| **PQ2 完整 (`--spec dflash2 --draft-tokens 12`)** | 1,133 | 396 ms | 2.87k tok/s | **571.9 tok/s**（dflash2 接受 91.5%） | **2.2 s** |

卡型/档位不同，**2 倍以内**都算正常；**差 3 倍以上**再按 §11 的顺序排查。

---

## 9. 实测读数（口径 + 出处）

> 出处一律写成相对路径，前缀 `p0-20261002/` = `（本机构建根）\p0-20261002\`（本机内部记录目录，不在本仓）。
> 那些 `.log` 是 **UTF-16LE**，用 `grep`/`Select-String` 读，不要用按字节读的 `read`。
> 每条都标了**性质**：正控 / 负控 / 红控 / 判别实验 / 未定位。

### 9.1 小池多针（核心结论，n=6）

口径：`--kv-over 4000` ⇒ 引擎自报 **KV 4,032 token（63 页）**；`k8v4`；题面 42,355 token（635 页）；
针页 299，针在 19,168/42,355 = **45.3%** 处，页内偏移 32/64（跨页缝擦边）；turn1/turn2 各 n=6。

| 臂 | 环的配置 | 结果 | 出处（`p0-20261002/`） | 性质 |
|---|---|---|---|---|
| 修前基线 | `TAIL=256` | turn1 6/6、**turn2 2/6 + `SILENT_WRONG=4`** | `pool4000-multineedle-n6-20261002.log` | 红控 |
| **最终件上的红控**（二进制 `753095EC…`） | `TAIL=256` | turn1 6/6、**turn2 2/6（t3–t6 丢，答 `ARCH-420x`）** | `d12fix2-redcontrol-tail256-20261002.log` | **红控（判据能变红）** |
| D-12 检索 span 打开 | `TAIL=64` + span ON | turn1 6/6、**turn2 6/6（`ZX-7001…7006`）** | `d12fix2-pool4000-20261002.log` | 正控 |
| **出厂形态**（**不设任何 `NINFER_TERNARY_KVMEM*`**） | 吃引擎默认 | turn1 6/6 + turn2 6/6；引擎侧 **`SELECT` 252 行、全部 `query_tokens=64`** | `evidence/vhidden-multineedle-4000-delivered.err.log`（`grep -c` = 252） | 正控 |
| 直读"引擎默认就是 64" | `QUERY_TAIL=UNSET` | 6/6 + 6/6 | `v2-pool4000-taildefault-20261002.log` | 正控 |
| 剂量-响应（窗口更小） | `TAIL=32` | 6/6 + 6/6 | `pool4000-multineedle-qt32-20261002.log` | 正控 |
| **预算轴单变量判别**（SHARE = 每轮可搬回页数，只重启不重编） | 25→15 页 / 50→31 页 / 70→44 页 | 25：turn1 **5/6**、turn2 2/6；50：6/6、2/6；70：6/6、2/6（**丢的始终是同样三条**） | `F1-GATE-RED-20261002.md`；`pool4000-multineedle-share{25,-n6,70}-20261002.log` | 判别实验 |

> **结论**：修前那条 turn2 2/6 **不是"预算不够"**（把每轮可搬回页数从 15 页加到 44 页，丢的还是同三条），
> 而是**检索窗口用错**：把查询从"题面末尾 256 行"换成"**当前问题的 token span**"之后，同一支二进制、
> 同一夹具，turn2 从 2/6 变 6/6；**把窗口改回 256，立刻回到 2/6** —— 判据可以双向变红。

### 9.1b 超池倍率（**结论有边界，别外推**）

| 臂 | 配置 | 结果 | 出处 |
|---|---|---|---|
| 94,698 token 题面 + 63 页池 = **23.5×** | `TAIL=64`（显式） | 针页 670、针在 45.3%；turn1 **3/3** + turn2 **3/3**、零静默错 | `long100k-main-20261002.log` |
| 同形**红控** | `TAIL=256` | turn1 hit=2 且 **1 次传输层中断（`http=0`）**；turn2 hit=1 + `SILENT_WRONG=1` | `long100k-redcontrol-20261002.log` |
| ⚠️ **反例留档** | 同 94,698 形状，但**夹具缩放 ×2.2** 且**不设 env** | **hit=0 / `SILENT_WRONG=2` / `other_bad=1`**（其中一次也是 `http=0`） | `shipshape-95k-3x-20261002.log` |

> ⚠️ 上表第三行**未定位**：它与主臂同时变了两处（夹具缩放 ×2.2、是否显式设 env）。
> ⇒ **23.5× 这个结论只写在主臂口径下**（94,698 token / 63 页 / 显式 `TAIL=64`）；
> **"出厂形态在 23.5× 下稳过"这句话没有证据，不要写、不要外推。**

### 9.2 交付形态门禁（池 17,920、`k8v4`、n=6；单一出处 `v2-delivered-gate17920-20261002.log`）

| 臂 | 读数 | 性质 |
|---|---|---|
| `over`（词法问句，SHARE=50 ⇒ 预算 128 页） | turn1 6/6 + turn2 6/6；掩码 812 行、`NEEDLE-PAGE-HIDDEN` 448/812 | 正控 |
| `multineedle` | 6/6 + 6/6（针页 299，hidden 508/884） | 正控 |
| `askdisjoint`（词面不相交问句） | 6/6 + 6/6 | 正控 |
| **`noretr`（关掉检索）** | turn1 **0/6 + `SILENT_WRONG=6/6`**、turn2 **0/6 + 6/6**（答 `ARCHIVE CODE: 42 / 499`） | **负控（"藏"是真的）** |
| `fits`（池 = `--max-context` = 16,384，256/256 页） | **掩码 0 行、Σhidden=0**、2/2 | 仪器控制 |
| 仪器闸门 | `INSTRUMENT LIVE + GATE RED` | —— |

同形状另两条交付臂：池 8,192（比交付档更严：128 页 / 预算 64 页）`multineedle` 6/6+6/6
（`v2-delivered-multineedle-20261002.log`）、`askdisjoint` 6/6+6/6（`v2-delivered-disjoint-20261002.log`）；
D-12 最终件重跑 `askdisjoint` 6/6+6/6（`d12fix2-delivered-disjoint-20261002.log`）。

**内容打分 vs 词法排序的直接 A/B**（同预算 **SHARE=25** = 旧池的 1/4 = 70 页；`askdisjoint`，池 17,920，n=6）：

| 臂 | 读数 | 出处 |
|---|---|---|
| 关内容打分（词法 IDF） | hit **4/6**、`SILENT_WRONG=2/6`（t2/t4 答 `ARCHIVE-0`）；掩码 814 行、Σhidden=193,654；GATE RED | `control-share25-20261002.log`、`v2-ab-lexical-share25-20261002.log` |
| **开内容打分** | hit **6/6**、`SILENT_WRONG=0`；GATE NOT RED | `control-content-share25-20261002.log`（`TAIL=256`）、`v2-ab-content-share25-20261002.log`（`TAIL=64`） |

> ⚠️ **这条 A/B 只在 SHARE=25 这个更紧的预算上成立**：把预算放回 SHARE=50，**词法也 6/6**
> （`v2-delivered-gate17920-20261002.log`）；池 4,000 上词法与内容同为 3/3
> （`pool4000-lexical` / `pool4000-content`）—— 那些"不区分"的对照**不能当证据用**。

### 9.3 本轮"模拟安装"读数（全新 ASCII 路径，从零走接收方步骤）

| 臂 | 启动器 | 就绪 | TTFT | 预填 | 解码 | 1000 token |
|---|---|---|---|---|---|---|
| PQ2 瘦身 | `start-pq2.bat`（`--spec mtp --draft-tokens 4`） | 10 s | 344 ms | 3.30k tok/s | 237.9 tok/s（mtp 88.1%） | 4.6 s |
| PTQ1_0 | `start-ptq1-mtp.bat` | 5 s | 565 ms | 2.01k tok/s | 196.8 tok/s | 5.7 s |
| 引擎包 sm_89 同形 | `start-pq2.bat` | 10 s | 340 ms | 3.35k tok/s | 228.0 tok/s | 4.7 s |
| **PQ2 完整** | `start-pq2-dflash.bat`（**draft 12**） | 10 s | 396 ms | 2.87k tok/s | **571.9 tok/s**（dflash2 91.5%） | **2.2 s** |

四臂每一条都过：制品字节数/哈希（`certutil` 与 `Get-FileHash` 两条独立路径）· 启动三行 ·
`kvmem_score: SELECT` ≥1 · `req#1 done` 落盘。
（另有一套口径在包内教程 §0.2：题面 1,133 / 2,533 token、draft 4 ⇒ PTQ1_0 decode 194.6、PQ2 全量 decode 278.0 ——
**题面不同，别与上表直接比**。）

### 9.4 权重载入与显存账（本机实测）

| 制品 | 权重 | 载入 | 池运行时 | 空闲显存 |
|---|---|---|---|---|
| GSQ-RCO IQ3_S | 13.4 GiB | **7.6 s（1.77 GiB/s）** | 2.79 GiB | 15.2 GiB |
| PTQ1_0 | 5.94 GiB | **2.5 s（2.35 GiB/s）** | 1.74 GiB | 23.6 GiB |
| PQ2 瘦身（mtponly） | 7.13 GiB | 未单列记录 | 1.74 GiB | 22.5 GiB |

（出处：本轮 GSQ 运行的引擎日志 `weights ready` 行与 `capacity |` 行；「未单列记录」= 日志里没留这条读数，**不猜**。）

### 9.5 长思考实测（GSQ + `reasoning xhigh`，一次完整任务）

```
req#1 started | openai-chat stream | 1 message | max output 131,072 | thinking xhigh
req#1 done | openai-chat | stop token | prompt 71 | output 46,046 | cache 0 (0.0%) | TTFT 211 ms
           | total 9m 18.9s | queue 17.7 ms | prefill 339.0 tok/s | decode 82.9 tok/s
           | mixed speculation accepted 34,292/119,215 (28.8%) | ngram 2,445/5,055 accepted, 337 rounds
```

思考 90,074 字符后自然收束、正文 27,229 字符完整交付（`finish_reason=stop`，交付物以 `</html>` 收尾）。

> ⚠️ **口径更正（2026-10-02 复核）**：这一次运行的 `--draft-tokens` **取 10**，由操作者在启动命令行里指定；
> **引擎日志不回显 argv**，包内 `start-gsq.bat` 写的是 **12** ⇒ 这一行的"draft 10"**没有可指回的制品证据**
> （当时运行目录没留启动脚本）。可信的是引擎自报的两项：`max output 131,072 | thinking xhigh` 与
> `mixed speculation accepted …`。**引用投机窗口的具体数字时，以包内启动器（12）或自留 argv 记录为准。**

**注**：该题面只有 71 token ⇒ 不产生 `SELECT` 行，**属预期**（没有超窗的页需要检索），不是打分没跑。
同一次运行的两条结构读数：`raw_k_shadow` 装 16 层、`kv_size=1024`、32.0 MiB（设备侧）；
`kvmem_index` `blocks=4096`、256.2 MiB。

### 9.6 仪器纪律（判据分类，照用时要带上）

- `NEEDLE-PAGE-HIDDEN lines`（如 812/1232、448/812）**只是区间启发式**：按 `∈[first_hidden, last_hidden]`
  计数，而隐藏页允许有洞 ⇒ **既不充分也不必要，不许单用它下结论**。
- `ring retrieve` 的去重键曾不含页码 ⇒ 同尺寸不同内容的轮次被静默合并（已改）。
- **`/v1/models` 返回 200 ≠ 能服务**（worker 死后它照样 200）⇒ **判活必须真发一条请求**。
- HTTP 状态码与 `cache%` **只记录、不进判据**。
- **速度数不能跨口径比**：KV 精度、语料形状（数数字 vs 散文差 3.2 倍）、prompt/输出长度都要一起给。

---

### 9.7 制品 × KV 档位（**只列跑过的格子**，空 = 未跑）

| 制品 | `fp8`（行粒度 256） | `int8`（G64） | `rk8v4` / `rk4v4`（G32） | `nvfp4`（G16 值平面） | **`k8v4`** | `bf16` |
|---|---|---|---|---|---|---|
| 三元 `Ternary-Bonsai-2-27B-ninfer-v3` | ✅ | ❌ 早停 | ❌ 早停 | — | ✅ | ✅ |
| GSQ-RCO IQ3_S | ✅ | — | — | — | ✅ | — |
| Swift Q4_K_M | ✅ | — | — | — | ✅ | — |
| **Swift-RCO IQ3_S** | **❌ 只出 2 token 就 EOS（复现 5/5）** | — | — | ✅ | ✅ | ✅ |

- 机理（源码级）：这几档的**值平面**共用一个粗粒度 absmax scale —— `fp8` 是**整行 256 个值 1 个**（`Fp8E4M3Row256`）、
  `int8` 是 G64、`rk` 系是 G32；值激活里只要有**离群通道**，同组其余值就被压成 0 或 1 个码点。
  **`k8v4` / `nvfp4` 的值平面是 G16 ⇒ 免疫**；**键平面无罪**（`k8v4` 的键就是 fp8 那条 row-256 编码）。
- 阈值：夹在 **232 ✅ / 432 ❌**（fp8 与 int8 相同）；换 temperature、去掉 `--spec`、不用环，都照样崩。
- **不要用"位宽更高 ⇒ 更安全"来推断**：实测 8 bit 的 `fp8` 崩、4 bit 的 `nvfp4` 正常。
- 同夹具显存对照（比显存**必须写 dtype**）：`k8v4` ⇒ `pages 256/256 | runtime 875.1 MiB`；`bf16` ⇒ 同页数 `runtime 1.47 GiB`。

### 9.8 同一件事的多口径：**并列，不许求平均**

| 事项 | 并列读数 | 为什么不能合并 |
|---|---|---|
| GSQ 解码速度 | **178.1**（draft 4，题面≈1,100，输出 96）· **240.8**（K12 = draft 12，题面 1,037，输出 96，n=2，抖动 ±4%）· **163.6**（bf16 KV 口径） | 投机窗口、KV 精度、题面长度三处都不同 |
| 80k 题面第二轮 TTFT | **126.5 ms**（80,063 token 冷启动后）· **195 ms**（另一处口径） | 池与记录口径不同 |
| 94.7k 形状 | **3/3 + 3/3**（主臂，显式 `TAIL=64`）· **2/3**（红控 `TAIL=256`）· **0/3**（`fixture ×2.2` 且不设 env，未定位） | 三条臂的配置不同（见 §9.1b） |
| MTP vs dflash2 | dflash2 **626** tok/s vs MTP **311** tok/s | MTP 省的是**权重**（7.12 vs 8.16 GiB），不是时间 |

### 9.9 容量守卫的"起因"（两条红控现场，用来证明这条守卫不是想出来的）

| 现场 | 读数 | 出处 |
|---|---|---|
| 主机层太小 + 池 8,192 | 盘子 = 池 128 页 + 主机 256 页 = **384 页** < 题面所需 **626 页** ⇒ `worker crash: … logical_capacity=384` | `infer-docs/05-链路与坑/坑与链路总表.md` §5.127 |
| `--host-kv-mib 64` + 环 + 池 1,024 + 40k 题面 | `req#1` 即 `worker crash: Paged KV reservation invariant was violated` + HTTP 500，日志里**没有任何** `adopt`/`publish` 行 | 同上 §5.138 |
| 环 + 仅 16 页池 + 40k 题面 | 起跑 **0.26 s** 就 `Paged KV reservation invariant was violated`；同配置补 `--prefill-chunk 256` 立刻两轮 200 且针命中 | `docs/01-部署与编译总白皮书.md` |

> 这三条就是"**启动期就该拒绝**"的来源：与其在跑起来之后崩，不如在**算得出页数的那一刻**拒绝并报出
> `host_pages=… pool_pages=… logical_pages=…`（见 §2.2 的守卫判据）。

### 9.10 出厂自证与"打分默认开"的成对负控

| 项 | 读数 | 出处 |
|---|---|---|
| 出厂形态自证（n=1，池 4,000 多针） | **`SELECT` 42 行 / `KEPT` 42 行**（打分真跑）、turn1 `4201` ✓、turn2 `ZX-7001` ✓、`SILENT_WRONG=0` | `p0-20261002/缺陷清单与处方-20261002.md`（D-1b 条） |
| 负控（关主开关） | `NINFER_TERNARY_KVMEM=0` ⇒ 启动行变成 `the content scorer was EXPLICITLY DISABLED` | 同上 |
| 静默降级（**最"静"的一条**） | 只设 `NINFER_KV_WINDOW` + `NINFER_KV_REUSE_HOSTBACKED`：引擎一切正常（`pages 152/152`、`cache 40,068 (99.9%)`），但 40k 题面**中段**的针看不见 —— **全绿日志 + 错答案** | `infer-docs/05-链路与坑/坑与链路总表.md` §5.143 |
| 假绿反例 | 漏掉 `NINFER_KV_REUSE_HOSTBACKED` 时：针测两轮全 PASS，但 `reuse host-backed: off`、第二轮 `cache 0 (0.0%)`、耗时 23.9/30.9/31.1 s ≈ 第一轮（**全量重填**） | 同上 §5.167 |

---

## 10. 判据与负控（"怎么知道它真的在跑"）

| 结论 | 正向判据 | 负控（必须能变红） |
|---|---|---|
| 环已启用 | `NINFER_KV_WINDOW` 出现在启动环境 | 去掉 ⇒ 无 `[ring] content scoring` 行 |
| 打分在跑 | `kvmem_score: SELECT … query_tokens=64` | `NINFER_TERNARY_KVMEM=0` ⇒ `EXPLICITLY DISABLED` |
| 检索窗口 = 问题 | 小池多针 turn2 6/6 | `QUERY_TAIL=256` ⇒ 2/6 |
| 掩码在工作 | `mask hidden … hidden=N` | 池 ≥ 题面 ⇒ `hidden=0` |
| 前缀复用 | 第二轮 `cache` ≈ 100% | 换一条前缀 ⇒ cache 归零 |
| 服务健康 | **真发一条请求**（`/v1/models` 200 不等于能服务：worker 死后它照样 200） | 杀掉 worker ⇒ 请求失败但 `/v1/models` 仍 200 |

---

## 11. 边界、未验与已知问题（**不许当已验读**）

1. **只在一张卡上端到端验过**：RTX 4080 SUPER / sm_89 / 32 GB / Windows。其它卡**未验**。
2. **构建未随仓发布**：不提供 `.exe`/`.dll`，也不承诺在别人的工具链上一定能编过。
3. **设备池下限**：已测最小 **4,032 token（63 页）**；更小会让"每轮可搬回的页数（池 × 50%）"变薄，**未测**。
4. **dtype 覆盖不均**：低池验收只在 `k8v4` 上做过；其它精度"掩码接线"验过、**低池验收未做**。
5. **跨页缝截断**：关键串正好跨在两页交界处时，若只搬回一侧，答案可能被截半（如 `ARCH-4201` → `ARCH-420`）。
   **已知未修的固有性质**，表现为"答案像被砍了尾巴"。
6. **2 分钟以上的冷预填仍可能在连接层被掐**：根因**未定位**；§7.2 的 keep-alive/超时是**缓解不是修复**。
7. **投机窗口收益依任务而定**：见 §6.2；`--adaptive-mtp`（仅 `--spec mtp`）是"自动选宽度"的路子，**未测**。
8. **它不是"无限上下文"**：逻辑上下文仍有上限（`--max-context`），超了请求会被拒（这是对的，不是 bug）。
9. **文档里的"三档"读数对应我们自己构建的量化制品**；**制品与权重都不在本仓**，读数只用于说明引擎行为。
10. **`reasoning xhigh` 的收益本身未量**：我们没做"同任务、思考开/关"的配对评测，**标未验**。

---

## 12. 协议、署名与来源

> 完整的第三方清单、许可名与**可复核证据路径**见 [`NOTICE.md`](NOTICE.md) 与
> [`docs/09-声明与链接.md`](docs/09-声明与链接.md)。本节给出摘要，**以那两份为准**。

### 12.1 本仓自身的许可

| 项 | 值 |
|---|---|
| 本仓内容（白皮书 / 我方改动文件 / 工具 / 文档） | **Apache License 2.0**（[LICENSE](LICENSE)） |
| 是否覆盖模型权重 | **不覆盖**（本仓不含权重；权重版权归其各自作者与上游） |
| 是否覆盖引擎二进制 / CUDA / FFmpeg | **不覆盖**（本仓不含这些二进制） |

### 12.2 基座与算法来源（**署名按各自许可要求保留**）

| 组件 | 在本仓的形态 | 许可 | 证据 |
|---|---|---|---|
| **NInfer**（引擎基座） | `patches/changed-files/` 是针对它写的改动 | Apache-2.0 | 上游树根 `LICENSE`；基线标识 `VERSION = 0.11.0-rtx3090` |
| **`tancau/ninfer-kvmem-ring`** | KVMem 环（打分 → 降级 → 映射 → 按需搬回 → 掩码）已并入引擎 | Apache-2.0 | 该仓 `LICENSE`（本机逐字核过） |
| **`kvmem-qw3`**（作者 Di Chai） | **含其源码改造版（8 件，`src/ops/kvmem/qw3/`，头部带 `PORTED` 声明）**，语义沿用其常驻窗口 / 检索预算 / 差量计划 | Apache-2.0 | 树内 `src/ops/kvmem/qw3/LICENSE-kvmem-qw3.txt` |

### 12.3 引擎树内 vendored 第三方（**本仓不分发**这些目录；列出是因为它们会编进二进制）

| 组件 | 许可 | 证据（树内路径） |
|---|---|---|
| `third_party/cpp-httplib` | **MIT**（© 2017 yhirose） | `third_party/cpp-httplib/LICENSE` |
| `third_party/ggml-quants` | **MIT**（© 2023-2026 The ggml authors） | `third_party/ggml-quants/LICENSE` |
| `third_party/llama-jinja` | **MIT** + 同目录 **UNICODE LICENSE V3**（© Unicode, Inc.） | `third_party/llama-jinja/LICENSE`、`third_party/llama-jinja/UNICODE-LICENSE` |
| `third_party/nlohmann`（json） | **MIT**（© 2013-2025 Niels Lohmann） | `third_party/nlohmann/LICENSE.MIT` |
| `third_party/spdlog`（含 bundled **{fmt}**） | **MIT** ×2（© Gabi Melman and contributors；© Victor Zverovich and {fmt} contributors） | `third_party/spdlog/LICENSE`、`…/fmt/bundled/fmt.license.rst` |
| `third_party/utf8proc` | **MIT "expat"**（另含原始 utf8proc 许可段：Public Software Group e. V., Berlin） | `third_party/utf8proc/LICENSE.md` |
| `third_party/xgrammar` | **Apache-2.0**（© 2024 XGrammar Contributors） | `third_party/xgrammar/LICENSE` + `NOTICE` |
| `third_party/xgrammar/3rdparty/dlpack` | **Apache-2.0** | `…/3rdparty/dlpack/LICENSE` |
| `third_party/xgrammar/3rdparty/picojson` | **BSD-2-Clause**（许可嵌在头文件里，无独立 LICENSE 文件） | `…/3rdparty/picojson/picojson.h:1-27` |
| `tools/chat_templates`（Qwen chat 模板） | **Apache-2.0**，附录版权行 = **`Copyright 2026 Alibaba Cloud`** | `tools/chat_templates/LICENSE` |

有版本记录的：spdlog `v1.17.0` · xgrammar `v0.2.7` · dlpack / ggml-quants / llama-jinja 各记来源 commit；
**cpp-httplib / nlohmann / utf8proc / picojson 树内无版本记录 ⇒ 未核**。

### 12.4 运行时二进制（**不在本仓**，随"分发包"分发时才出现）

| 二进制 | 许可类型 | 许可原文是否随包 |
|---|---|---|
| `cublas64_13` / `cublasLt64_13` / `cudart64_13` | NVIDIA CUDA Toolkit EULA 的**可再分发运行库**条款 | **未随包** |
| `avcodec-63` / `avformat-63` / `avutil-61` / `swresample-7` / `swscale-10` | **GPL v2 or later**（该构建启用 GPL 部件；**随包二进制里实证** configure 含 `--enable-gpl --enable-version3`） | **未随包** |
| `libcurl-x64` | curl 许可（MIT 类；二进制内含版本串 `libcurl/8.22.0`） | **未随包** |
| `ninfer-serve-{86,89,120a}.exe` | Apache-2.0（本引擎构建产物） | 随包 |

⚠️ **口径更正（2026-10-02 复核）**：旧文档把 FFmpeg 写成"LGPL/GPL（随构建）"，**以 GPL 为准**；
旧文档又称"许可原文随包"，实际分发包**没有 `licenses\` 目录** —— 上表三条"未随包"是**已知待办**，
对外分发二进制前应补回许可文本或改写声明。逐条见 `NOTICE.md` §6–§7。

### 12.5 模型权重归属（本仓**不含**权重）

| 模型 | 版权归属 | 许可 | 核实状态 |
|---|---|---|---|
| Qwen3.8-27B（底座） | **Copyright 2026 Alibaba Cloud** | Apache-2.0 | 制品 `artifact-NOTICE` 逐字转述 + 模型卡；**上游 LICENSE 正文未逐字核** |
| Ternary Bonsai 2 27B（PTQ1_0 / PQ2_0） | **Prism ML, Inc.**；MTP 头与 DFlash2 适配器 = **ProCreations** | 三来源均声明 Apache-2.0 | **制品 NOTICE 已逐字核**；上游仓 LICENSE **未核** |
| GSQ-RCO IQ3_S | 基座 `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-GGUF`；打包者 `WaveCut/…-NInfer-v3` | 记为 Apache-2.0 | **未核** |
| Swift-RCO IQ3_S | 上游仓 `ticeclock/Swift-Qwen3.8-27B-RCO-GGUF` | — | **未核**（不得用同族别的制品替证） |

### 12.6 我方改动声明（Apache-2.0 §4(b)）

改动逐文件清单与判据见 `patches/README-改动说明.md`；类别级声明见 `NOTICE.md` §2。

---

## 13. 附录

### 13.1 本仓文件地图

```
复现白皮书-NInfer-KVMem环-20261002.md   本文件（入口）
README.md                                仓说明（解决什么问题 / 目录 / 边界）
NOTICE.md                                署名、许可与我方改动声明（协议层权威）
LICENSE                                  Apache License 2.0
patches/                                 我方 38 个改动文件 + 逐文件说明 + 机器可读清单
  changed-files/                         改后的完整文件（路径与引擎源码树一一对应）
  changed-source.txt / changed-files.txt 改动清单
  README-改动说明.md                     基线、判据、分类、诚实清单（先读这个）
docs/                                    对外口径的技术文档（读数/白皮书/排错/调优/禁忌/Bug 手册/按卡差异/操作手册）
docs/方案/                               我方自研路线的评估与立项（内部视角）
tools/                                   自建工具：容器读法、字节账、装载安全闸、引用完整性、重打包、模型体检、清单、冒烟
verify/                                  判据脚本：包清单门、形状门、按卡匹配、本卡验收、清单校验
```

### 13.2 术语

| 词 | 含义 |
|---|---|
| 设备池 | 显存里那块 KV（`--kv-capacity`，单位 token） |
| 宿主池 | pinned 主机内存里的 KV 副本（`--host-kv-mib`） |
| 常驻窗口 | 每轮必驻留的最近 token 段（`NINFER_KV_WINDOW`） |
| 检索（retrieve） | 按相关性把页从宿主层搬回设备池 |
| 掩码（mask） | 让注意力看不见"没搬回来"的页 |
| 降级/搬回 | demote / restore 的动作对 |
| 前缀复用 | 同一会话下一轮直接复用已算好的 KV（cache 命中） |

### 13.3 变更记录

| 日期 | 内容 |
|---|---|
| 2026-10-02 | 本白皮书首版：三条线 + KVMem 机制 + 复现步骤 + 判据与负控 + 本轮实测读数（含"检索窗口=问题"与"内容打分默认开"两条关键修正） |
