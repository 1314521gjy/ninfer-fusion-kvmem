# NInfer KVMem：输出租约（`--kv-lease-growth`）缺陷报告

> 反馈对象：`ninfer-serve-89.exe`（sm_89 版，sha256 `19222a2a68df6f7d87889ee3b33bedf2f3486ea309b6394a8f8a1a412262771e`）
> 报告日期：2026-10-08 ｜ 由用户实测提供，含最小复现与对照实验
> 平台：Windows / RTX 4070 Ti SUPER 16GB (sm_89) / 驱动 596.36

---

## 0. 一句话

**`--kv-lease-growth` 在「池被占满」时会把输出租约静默冻结在初始 4096 token，既不扩也不说明原因；
默认给准入搜索的预算只有 5 ms，搜索还在 expansion 阶段就被掐断（`stop_reason=insufficient_expected_gain`）。
后果是单次回答被静默截断在约 4056 token；对思考模型而言，思维链先花掉这 4096，于是响应里 `content` 为空串
（`finish_reason=output_limit`），上游 agent 框架完全无法判断真实原因。**

---

## 1. 环境

| 项 | 值 |
|---|---|
| 引擎 | `ninfer-serve-89.exe`，sha256 `19222a2a…`（`engine.old-4bc7-20261003/` 内旧版为 `4bc7bdbf…`） |
| 模型制品 | `Ternary-Bonsai-2-27B-ninfer-v3.ninfer`（8.87 GiB，`Qwen3_5ForCausalLM` 结构，n_ctx_train 262144） |
| 运行参数（复现用） | 见 §3 |
| 请求日志 | `--request-log-jsonl`（`schema_version: 27`），含 `request_start` / `request_done` / `materialization` 全精度字段 |

---

## 2. 现象

上游（Hermes agent）用 8091 跑长会话 + 思考常开（`reasoning_effort=medium`），约 10 分钟后报
「No visible answer was produced — the model hit its output-token limit on every continuation attempt」。
抓服务端日志：**单次回答一律停在 `completion_tokens ≈ 4036–4085`，`finish_reason=output_limit`，`content` 为空。**

但客户端请求的是 `max_tokens=16384`（`requested_output_tokens_source=server_default`），
说明**绑定约束不是客户端预算**。

---

## 3. 最小复现

### 3.1 运行参数（池 < 上下文，即所谓"小池档"）

```
ninfer-serve-89.exe Ternary-Bonsai-2-27B-ninfer-v3.ninfer
  --host 127.0.0.1 --port 8091 --model-id "Bonsai 27B KVMem"
  --max-context 131072 --kv-capacity 65536        <-- 关键：池 = 上下文/2
  --kv-dtype rk4v4 --host-kv-mib 16384
  --max-concurrency 1 --max-shared-prefixes 0 --prefill-chunk 1024
  --spec dflash2 --draft-tokens 10 --lm-head-draft --ngram-draft-tokens 0
  --kv-lease-growth --recover-invariant-failures
  --default-max-tokens 16384
  --unconstrained-response-format --vision --vision-residency overlay
```

环境变量（KVMem 相关）：`NINFER_KV_REUSE_HOSTBACKED=1`、`NINFER_KV_WINDOW=16384`、
`NINFER_KV_RETRIEVE=8192`、`NINFER_KV_RING=1`、`NINFER_HOST_PAGEABLE=1`

### 3.2 步骤

1. **灌池**：发一条约 **32 000 token** 的题面（任意长文），`max_tokens=64`。
   → `GET /metrics`：`kv_cache_tokens 65024` / `kv_cache_usage_ratio 0.9922`（池 65536，空闲仅 **512**）
2. **施压**：发一条题面极小、但**强制要求超长输出**的请求：
   ```json
   {"model":"Bonsai 27B KVMem","max_tokens":16384,"reasoning_effort":"none",
    "messages":[{"role":"user","content":
      "Output exactly 1200 lines. Line n must be exactly: \"<n>. The quick brown fox jumps over the lazy dog.\" Start at 1, go to 1200, do not skip, do not summarize, do not stop early."}]}
   ```
3. **观察**

### 3.3 实测结果

| 情形 | 题面 | 池 | 输出 | `finish_reason` | `search_granted_ns` | `search_stop_phase` | `stop_reason` |
|---|---:|---:|---:|---|---:|---|---|
| 池 65536（默认搜索） | 31 | 65024/65536 | **4 056** | `output_limit` | 5 000 000（5 ms） | `expansion` | `insufficient_expected_gain` |
| 池 65536（默认搜索，另一条） | 32 394 | 65024/65536 | **4 077** | `output_limit` | 5 000 000 | `expansion` | `insufficient_expected_gain` |
| 池 65536 + `--thorough-admission-search` | 66 | 65024/65536 | **16 384** ✅ | `output_limit` | 249 795 400（249.8 ms） | `expansion` | `queue_exhausted` |

**⇒ 同一实例、同一 99.22% 池占用，唯一变量是搜索预算：5 ms → 249.8 ms，输出 4 056 → 16 384。**
即：**容量本来是有的，引擎只是没搜到。**

> ⚠️ 注意：本例之所以能扩，是因为占住池的是**另一个会话**的常驻 KV，可以被腾走。
> 若占住池的是**本请求自己的长题面**，即使给足 250 ms 也扩不动 —— 那是**缺陷 2**（§5），
> 且它是「池 < 上下文」配置下调参绕不过去的，必须靠放开池容量或压题面。

---

## 4. 缺陷 1：默认准入搜索预算过小，且失败无信号

- `--kv-lease-growth` 的 `--help` 原文：
  > `reserve a 4096-token output window and extend it instead of the whole max_tokens budget; an answer the pool cannot extend ends with length`

  但实测：**"the pool cannot extend" 在默认配置下并不是真的不能，而是没给它时间去找。**
- `materialization` 里三条证据齐了：
  - `search_granted_ns = 5 000 000`（5 ms；启动后头两条请求甚至是 0）
  - `search_elapsed_ns = 4 906 300`（把 5 ms 全用光）
  - `search_stop_phase = "expansion"` ← **搜索还在"扩展"阶段就被时间掐断**
  - `stop_reason = "insufficient_expected_gain"` ← 随后判定收益不足而放弃
- 对照 `--help`：`--thorough-admission-search  search up to 250 ms ... (otherwise 10 ms)`。
  **默认 10 ms 已经偏短，实测只拿到 5 ms，而且是在 expansion 阶段被截断。**

**建议**：
1. 输出租约扩容这一路，默认预算不应沿用"准入复用规划"的 10 ms —— 它是**决定一条回答能不能写出来**的路径，值得更高的预算（或按剩余空闲页/预估输出长度自适应）。
2. `stop_reason` 不应只在内部日志里；客户端侧需要能看到"输出租约因池不足而未扩展"。

---

## 5. 缺陷 2（更根本）：输出租约的扩容不参与"本请求自己常驻 KV"的降级

这一条是**「池 < 上下文」配置下无法通过调参绕开**的原因。

### 5.1 实测（真实 agent 负载，池 65536，已开 `--thorough-admission-search`）

实例 `serve-17440-…`，Hermes 会话、`enable_thinking=true`、`reasoning_effort=medium`：

| 请求 | 题面 | 输出 | `finish_reason` | `selected_degradation_units` | `search_granted_ns` |
|---:|---:|---:|---|---:|---:|
| 1 | 88 216 | **4 063** | `output_limit` | **0** | 0 |
| 11 | 97 218 | **4 085** | `output_limit` | **0** | 234.8 ms |
| 22 | 100 851 | **4 036** | `output_limit` | **0** | 67.5 ms |
| 25 | 101 462 | **4 065** | `output_limit` | **0** | 35.4 ms |
| 32 | 105 380 | **4 051** | `output_limit` | **0** | 64.5 ms |

- 搜索预算已经给到 35–235 ms（旗标确实生效），**但 `selected_degradation_units` 五次全是 0**。
- 同时期能自然收尾的请求（`stop_token`）输出都在 4 100 以下 ⇒ **本实例输出天花板就是那口 4096 初窗**。

### 5.2 通过日志能建立的统一模型

设 **输出额度 = `--kv-capacity` − 常驻 KV**（常驻随题面增长，直到顶满池）：

| 场景 | 题面 | 池 | 输出 | 解释 |
|---|---:|---:|---:|---|
| 灌池后施压（常驻是**别的**会话） | 31 / 66 | 65536 | 4 056 → **16 384**（开旗标后） | 引擎**能**腾出别人的 KV |
| 真实会话（常驻是**本请求自己**） | 88k–105k | 65536 | **≈4 056** | **腾不出自己的**，额度恒为 0 |
| 历史记录（未开旗标） | 46 610 | 65536 | **16 384** | 题面只占池一半，额度天然存在 |
| 历史记录（未开旗标） | 63 174 | 65536 | **4 081** | 题面接近池，额度≈0 |
| 池放开后（本轮修复） | 113 299 | 131072 | **8 383** | 额度 = 131072−常驻 ≈ 26k，够用 |

该模型在**全部 495 条真实会话请求 / 23 个实例**上无一反例（见 §8）。

**观察：KVMem 的"池 < 上下文"能力目前只对 prefill/检索方向成立——把题面送进池、检索回来都没问题；
但在 decode/输出方向，"输出"这个需求在优先级上排在"保住（本会话的）常驻 KV"之后，且不可让渡。**
`--recency-eviction` / `--value-aware-demote` 的 `--help` 语义是
`give up the least recently used **owners** ... demote kept ones to Host`，动的是**别的会话**，
管不到本请求自己的常驻 KV —— 与上面 `selected_degradation_units = 0` 的观察一致。

**建议**：让输出租约的扩容**也**能触发本会话的 KVMem 降级/检索路径
（即：把本会话较老的段按既有检索语义降级到 Host，换出输出额度；允许以一点延迟换"回答能写出来"）。
否则在「池 < 上下文」这种档位上，"长题面 + 需要长输出的回答"必然被截断，且用户完全无从判断。

---

## 6. 缺陷 3：失败不可诊断（建议顺手修掉）

客户端只能看到 `finish_reason = "length"`（OpenAI 线）与空 `content`，**无法区分**：

- 客户端的 `max_tokens` 到了；
- 还是**池装不下 ⇒ 输出租约没扩成**。

`materialization.stop_reason`（`insufficient_expected_gain` / `queue_exhausted` / `no_pressure`）
与 `search_stop_phase` 只落在服务端日志里。上游把这种响应理解为
"模型自己把预算用完了"（Hermes 就把它显示成 "its reasoning consumed the entire budget each time"），
**诊断方向完全错了**，也就无法自动做出正确降级（例如减小题面、或换池档）。

**建议**：给这类终止一个可区分的 `finish_reason`（如 `kv_lease_exhausted`）或在响应里附一个字段（如
`output_lease_granted_tokens` / `pool_free_tokens`），让客户端能自动应对。

---

## 7. 修复验证（已实测）

把池放开到等于上下文（**仅此一处改动**）：

```diff
-  --max-context 131072 --kv-capacity 65536
+  --max-context 131072 --kv-capacity 131072
```

（`--thorough-admission-search` 保留）

| | 池 | 请求数 | `output_limit` 次数 | 最大输出 |
|---|---:|---:|---:|---:|
| 改前 `serve-17440` | 65536 | 32 | **5** | 4 085 |
| 改后 `serve-10740` | 131072 | 36 | **0** ✅ | **8 383** |

- 改后 36 条请求 `finish_reason` **全是 `stop_token`**，题面 105k–118k、`enable_thinking=true`、`effort=medium`。
- **同一个 Hermes 会话（此前失败过两次）在 03:11:08 跑完**：`status=complete, error_retained=False, duration=639.1s`。

**⇒ 两个机制都需要：`--thorough-admission-search` 解决"能扩却不扩"，池=上下文解决"额度本身为 0"。**

---

## 8. 影响面

- 「池 < 上下文」正是 KVMem 这一档的**卖点配置**（用主机 KV + 检索把 128k 上下文塞进更小的设备池）。
  但该配置下，**任何"题面 + 期望输出 > 池"的请求都会被静默截断到约 4056 token**。
- 对思考模型是**致命组合**：思维链先花掉那 4096，于是 `content` 为空 —— 用户看到的是"模型不回答"，
  而不是"输出被截断"，排查方向会被带偏。
### 全量回看（`--request-log-jsonl` 至今全部记录）

| 统计 | 值 |
|---|---|
| 真实会话请求（`message_count > 3`） | **495** 条 |
| 实例数（`server_start`） | **23** 个 |
| 满足 `题面 + 输出 > 池` 的样本 | **233** 条 |
| 其中**输出 > 4200** 的反例 | **0** ✅ |
| 输出 > 4200 的全部样本 | 11 条，**全部**满足 `题面 + 输出 ≤ 池` |

输出 > 4200 的 11 条（两个池档都在内）：

| 实例 | 题面 | 输出 | 池 | 题面+输出 |
|---|---:|---:|---:|---:|
| serve-30092 | 46 610 | 16 384 | 65536 | 62 994 |
| serve-5812 | 49 361 | 12 277 | 65536 | 61 638 |
| serve-31980 | 41 531 | 9 856 | 131072 | 51 387 |
| **serve-10740（修复后）** | 113 299 | **8 383** | 131072 | 121 682 |
| serve-4832 | 47 432 | 7 709 | 65536 | 55 141 |
| serve-32784 | 57 914 | 7 549 | 65536 | 65 463 |
| serve-10740（修复后） | 105 312 | 4 678 | 131072 | 109 990 |
| serve-10740（修复后） | 114 153 | 4 445 | 131072 | 118 598 |
| serve-4832 | 48 367 | 4 444 | 65536 | 52 811 |
| serve-31980 | 84 602 | 4 436 | 131072 | 89 038 |
| serve-31980 | 88 254 | 4 436 | 131072 | 92 690 |

**⇒ 「题面 + 输出 ≤ `--kv-capacity`」是这批数据里 495 条无例外的上界。**

---

## 9. 附：证据字段索引

服务端 `--request-log-jsonl` 中，`event = "request_done"` 记录里可直接取到：

| 字段 | 用途 |
|---|---|
| `result.completion_tokens` / `result.finish_reason` | 实际输出长度与终止原因（`output_limit` = 撞输出上限） |
| `request.requested_output_tokens` / `requested_output_tokens_source` | 确认绑定约束**不是**客户端预算 |
| `request.enable_thinking` / `requested_reasoning_effort` | 确认思考是否开启 |
| `result.prompt_tokens` / `prefix_cache_hit_tokens` / `prefix_reuse_path` | 确认常驻是"本会话"还是"别的会话" |
| `materialization.search_granted_ns` / `search_elapsed_ns` / `search_stop_phase` / `stop_reason` | **缺陷 1 的直接证据** |
| `materialization.selected_degradation_units` | **缺陷 2 的直接证据**（真实会话恒为 0） |
| `GET /metrics` → `llamacpp:kv_cache_tokens` / `kv_cache_usage_ratio` | 池占用 |

---

## 10. 附：其他疑点（低置信，供作者确认）

1. **`result.model_thinking_tokens` 恒为 0**：在 `request.enable_thinking = true` 的多条记录里，
   该字段都是 0；而对同一引擎直接请求时，OpenAI 侧 `usage.completion_tokens_details.reasoning_tokens`
   能正常返回非零值（实测 66 / 89 / 109）。疑似该日志字段只统计"预算受控"的思考，统计口径与预期不符 ——
   会让排查者误以为模型没有思考。**请确认是否为统计口径问题。**
2. **`search_granted_ns` 在同一实例内差异很大**（0 / 24 / 42 / 67 / 93 / 235 ms……），
   而请求形态相近。若这是按"请求成本"自适应，建议在文档或日志里说明判据，便于用户判断何时该开 `--thorough-admission-search`。

---

## 11. 复现包清单

| 文件 | 说明 |
|---|---|
| `verify-output-window.py` | 单点探针：打印 `n_predict` / 池占用 / 空闲池 / 实测输出上限 |
| `ab-test-thorough-search.py` | 完整 A/B：确认旗标进 argv → 灌池至 ≥95% → 强制配额题面施压 → 回读 `materialization` |
| `Bonsai-KVMem-输出墙-机制定稿.md` | 完整推导与全部原始读数（含 19 个历史实例的回看） |

（两个脚本均只用标准库，`python verify-output-window.py 8091` 即可跑。）
