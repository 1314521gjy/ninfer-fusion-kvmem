# NInfer + KVMem 在 RTX 5080（16 GB）上的实测记录

> 给三叔的一份数据包 —— 全部数字都是**本机在役部署上跑出来的**，每条都带可复现的命令与原始日志片段。
> 整理：2026-10-09 · 来源：`1314521gjy/ninfer-fusion-kvmem` release `engine-v0.11.0-kvmem-20261008`（融合件）
> 我们这边不是研究者，是用户：一台 16 GB 的 5080 跑 27B 档做日常 agent 活。下面是我们**撞到的边界**和**量出来的账**，哪条有用你挑。

---

## 0. 环境（复现需要的最小信息）

| 项 | 值 |
|---|---|
| GPU | **RTX 5080 16 GB**（sm_120；本机 `nvidia-smi` 报 NVML 错误，显存用 `Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage'` 读） |
| 引擎件 | `ninfer-serve-120a.exe` **686,013,440 B**，sha256 前缀 `98DDB8C4…`（与官方 `SHA256SUMS.txt` MATCH） |
| 权重 | `bonsai2_27b_ternary_ptq1_native_mtp.ninfer`（6,394,697,216 B）· 另备 Ternary-v3（8.87 GiB）/ Swift / Heretic |
| 端口 | 引擎直连 **8087**；我们另有一层代理 1249（字节不变转发） |
| 关键 env | `NINFER_KV_WINDOW=16384` `NINFER_KV_RETRIEVE=8192` `NINFER_KV_RING=1` `NINFER_HOST_PAGEABLE=1` `NINFER_KV_REUSE_HOSTBACKED=1` `NINFER_TERNARY_PTQ1_FAST=1` |
| 启动后自证 | `pinning host KV 16.0 GiB` → `capacity \| KV 17,920 tokens, k8v4, explicit \| pages 280/4,096 \| runtime 1.74 GiB \| free 7.69 GiB` → `listening on http://127.0.0.1:8087` |

**注意一条 CLI 契约**（我们踩了两轮才发现）：模型路径必须**紧跟 exe、在所有开关之前** ——
`ninfer-serve-120a.exe <model.ninfer> --host … --port …`。
把 `--host/--port` 放在模型前面会得到 `ninfer-serve: unknown argument: <模型路径>`（而 `--help` 的 usage 行正是这么写的，容易看漏）。

---

## 1. 我们的部署形状（agent 活：工具多、题面中等、要长输出）

`ptq1` 档实际 argv（逐字）：

```
<exe> <model> --host 127.0.0.1 --port 8087 --default-max-tokens 32768 --default-reasoning-effort xhigh \
  --max-concurrency 1 --max-shared-prefixes 0 --presence-penalty 0 --model-id ninfer-ptq1 \
  --max-context 262144 --kv-capacity 17920 --kv-dtype k8v4 --host-kv-mib 16384 \
  --prefill-chunk 1024 --spec mtp --draft-tokens 4 \
  --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0.05
```

典型客户端请求：**7 条消息 / 39 个工具 / `thinking xhigh`**，所以**单次 prompt 常态就是 2 万 token 上下**（工具 schema 与系统提示占大头）。
⇒ 在我们这儿 **池 17,920 天生小于一次请求的 prompt**，"超池"是常态而不是异常。

---

## 2. 四条实测（每一条都附原始行）

### 2.1 `429` 的那条真路：**池里没有空闲页可回填**

题面 20,741 token（325 页）> 池 280 页，引擎必须靠 host-backed 放；
紧接着的两次请求都倒在同一步：

```
14:00:24 [warning] prompt exceeds the resident Device KV pool: prompt 20741 tokens (325 pages) > pool 17920 tokens (280 pages)
14:00:24 [ninfer] adopt host-backed: frontier=20346 need=318 usable=280 host_backed=187
14:00:24 [diag] materialize-one FAIL tag=host-restore usable=281 allocated=281 reserved=38 reservation=38
14:00:24 WARN  req#3 failed during generation | HTTP 429 | server overloaded
14:00:24 （req#4 同形：adopt → materialize-one FAIL → HTTP 429）
```

可疑点（**给你定位用**，不是结论）：`usable 281` 比 `capacity`（280 页）**多 1 页**，且 `lent 0` 而 `allocated = 281`、`reserved 38`。
另一头的对照是 `2026-10-06` 那次（**同一台机、不同档**）：

```
[ninfer] adopt host-backed: frontier=30195 need=472 usable=280 host_backed=472
ERROR engine | worker crash: Paged KV reservation invariant was violated
→ 之后每发 503
```
⇒ **同一类"池不够 + host-restore"在不同档上分别表现为 `worker crash(500)` 和 `429`**，触发条件我们没能稳定复现（见 §3 的负结果）。

### 2.2 同一个 200K 级长题面，两臂对照：**超池 ≠ 降质**

夹具：约 265,000 字符（中文）⇒ **实测 prompt = 198,128 token（3,096 页）**，密语埋在 60% 处，要求三行输出（口令/动作/风险）。两臂**只差池子**：

| 臂 | 池 | 首发 | 复用（逐字重发） | 质量 |
|---|---|---|---|---|
| A | **17,920**（280 页，k8v4） | 200 / **284.4 s** / out 844 tok | 200 / 16.8 s | 口令 ✅、三行齐、**重复窗口 0/2**、复述原文 0 处 |
| B | **34,176**（534 页，k8v4，关投机） | 200 / **282.1 s** / out 530 tok | 200 / 48.6 s / **正文 0 字** | 首发同样 ✅（第 2 发是"思考吃光输出预算"，与池无关） |

⇒ **两臂都全文命中、都不重复、都不复述**；池子差 1.9 倍，**质量无可观测差别**。
（对我们最有用的启示：池子买的是"少告警、少 host-back"，不是正确率。）

### 2.3 真正的墙是 **prefill 算力**，不是内存

同一 198K 题面，两臂首发票**都是 ~283 秒**（≈ **1.75k tok/s prefill**），与池子大小、是否开投机都无关。
⇒ 在我们这张卡上，**"一次吃 20 万 token"的代价是分钟级**，这是物理量级。

### 2.4 单请求层面的失败：`thinking` 吃光输出预算

另一条独立现象（与池无关，但**用户会当成"模型坏了"**）：
`max_tokens=2000` + `thinking xhigh` 时，**思考先吃掉全部预算** → 正文 0 字、`finish_reason=length`。
⇒ 客户端要把 `max_tokens` 留足（我们这边至少 4k 起步），或者由我们在请求侧给出"思考预算"提示。

---

## 3. 我们**没能复现**的（负结果，同样给你）

在隔离口（另一台实例）上，试图复现 §2.1 的 `429`：

| 尝试 | 配置 | 结果 |
|---|---|---|
| 20.5K 题面连发两轮 | 池 280 / 窗口 16384 / 投机开 | 两轮 200、密语命中、**无 FAIL/429** |
| 同上，池加大 | 池 420 页 | 同上（无差别） |
| 先打一发 **8,000 token 长输出**把池填满，再连发 | 池 280 页 | 三发全 200、命中、**无 FAIL/429** |

⇒ `429` 那条路**需要某种我们还没卡到的状态**；从生产现场看，当时的时间特征是"**GUI 连续重发**"（`req#3` 与 `req#4` 相隔 **90 ms**，两次 `max_tokens` 分别 65,536 / 32,768）。
**这一条我们只能提供现场，不能提供最小复现** —— 谁有源码谁更容易钉死它。

---

## 4. 顺手读到的论文（可能比我们的数据更值得看）

**ActKV: Efficient LLM Agents through Action-Guided KV Cache Management**（USTC，arXiv **2609.31395**，2026-09-25）：
- 主张：agent 式推理里 **action token 才是决定成败的**，observation/reasoning 占 >99% 缓存却对用户无用；
  现有压缩法（H2O / SnapKV / StreamingLLM / R-KV）按"对下一句的注意力"淘汰，**会误删动作关键的条目**；
- 做法：① 按 **action 区域注意力** 打分 + 注意力感知 LRFU 保留；② 用**模型置信度**动态调预算（预算不够时模型会先冒 `wait / I guess / let me check` 这类低置信措辞）；③ 页感知三原语（算注意力 / 淘汰 / 就地压实）+ 定制 kernel；
- 数字：长轨迹任务 **98.53% FullKV 准确率 / 25.98% 峰值 KV**；token、任务吞吐 3.97× / 3.58×。

**为什么觉得跟你相关**：我们的 KVMem ring 已经在做"分页 + 按需检索"（`kvmem_score: SELECT …` 是我们日志里最常见的行），
但**打分维度是"查询/内容重合度"**；按这篇的说法，"对动作的贡献"可能是更该用的维度。
我们这边的观测是：**池 280 页做 3 万 token 的 agent 轮次很吃力（每轮都告警 + host-back），而池加大一倍也没有质量收益** —— 如果换成"按动作价值保留"，也许**小池也能稳住 agent**。是否值得，你比我清楚。

---

## 5. 我们这边可复用的东西（都在磁盘上）

**✅ 已经拷进本目录（可直接跑/直接看）**

| 文件 | 用途 |
|---|---|
| `复现步骤_20261009.md` | **照着敲就能复现**：隔离实例怎么起、探针怎么发、怎么收尾 |
| `原始日志摘录_20261009.md` | `capacity` / `exceeds the resident` / `adopt host-backed` / `materialize-one FAIL` / `HTTP 429` / `req#` 的**逐行原文**（未改写） |

**留在我们本机、没拷进来的**

| 文件 | 用途 |
|---|---|
| `…\Documents\ai\tmpwork\_kvpool_ab_start.py` | 从启动器逐字取 argv、**只覆盖 `--kv-capacity/--port`**，在**隔离端口**起一台（支持 `--window` / `--no-spec`） |
| `…\Documents\ai\tmpwork\_kvpool_ab_ask.py` | 造长题面（中段埋密语）→ **连发两轮** → 打印状态/耗时/命中 + 引擎日志新增关键行 |
| `…\Documents\ai\tmpwork\_hedge_fixture_120k.py` | **198K token 级夹具**（265K 字符）+ 重复率/复述率统计 |
| `…\Documents\ai\tmpwork\_fixture_120k.txt` | 上面那份夹具原文（565 KB，可直接复用；本目录也有一份） |
| `…\Documents\ai\tmpwork\_wmi_fire.ps1` | 本机起常驻引擎的正确姿势（WMI 发射；agent shell 直接起的会被连带杀掉） |
| 原始日志 | `…\Documents\ai\tools_dashboard\infer-ptq1.{out,err}.log`（生产的完整日志，含 §2.1 的现场） |
| 更早一份源码树（**v1.0.8 fork 线，不是 120a 线**） | `…\Documents\ai\tmpwork\cs_backup\deleted_20261003_ninfer_old\src-tree\ninfer-4090w-ternary\`（1,697 文件 / 243.5 MB，含 `src\core\paged_kv_cache.{h,cpp}`、`host_kv_arena.{h,cpp}`） |

> ⚠️ 最后一条特别说明：我们**在用的 120a 包与融合件里没有任何源码文件**（只有 exe/models/文档），
> 所以我们**没有改引擎**、也没法改；上面那份旧源码树只是我们本地备份，仅供你参考它的结构，**别当成当前版本**。

---

## 6. 一句话总结（我们最想告诉你的）

> **在我们这台 16 GB 卡 + agent 工具负载下：超池本身不降质，池子加大也不提速；真正卡人的是 ① `host-restore` 在某种状态下拿不到空闲页（→ 429 / 曾见过一次 worker crash），② 20 万 token 的 prefill 要 ~4.7 分钟。**
> 如果要在引擎侧投入，我会先投 (1)（因为它会**直接失败**），然后是 ActKV 那种"按动作价值保留"的路子；池子大小我会放在最后。

---

*本文由本机 agent 汇总落盘，数据来源均为本机日志与实测脚本输出；如有哪条想让我补原始日志或补跑，说一声即可。*
