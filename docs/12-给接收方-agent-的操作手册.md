# 12 · 给接入方 agent 的操作手册（**先读这一页**）

> **这份是写给你（跑这个包的 AI / 工程师）看的**，不是我们自己的备忘录。
> 它按"你会遇到的顺序"写：落地 → 起服务 → **哪几秒是引擎在测你的卡** → 就绪判据 → 怎么把长上下文用起来 →
> 按卡微调 → 会炸的禁忌 → 已知 bug → 思考循环 → **怎么把读数回给我们**。
>
> **口径出处（一个事实一个出处，别处不复制）**：
> 本包三档的实测读数 = `00-三档口径与读数.md`｜坑与禁忌 = `05-已知问题与禁忌.md`｜
> 思考循环 = `04-卡死与循环的防治.md`｜引擎全量开关 = `01-部署与编译总白皮书.md`（⚠ 见下方封条）。
>
> ⚠️ **封条**：`01 / 02 / 03 / 08 / 11` 五篇是从**另一个包（单档 8090）**搬过来的，里面
> `权重 7.28 GiB / 池 152 页 / --host-kv-mib 4096 / 端口 8090 / 磁盘 10.9 GiB` 是**那个包的形状**。
> **本包只看 `00` 与本页**——配参数时以本页 §2/§5 的逐字 argv 为准。

---

## §0 包里有三件，各干什么

| 件 | 目录 | 你只需要知道 |
|---|---|---|
| **模型包** | `models\` | 三个 `.ninfer`。**别改名、别放中文路径** |
| **引擎包** | `engine\` | `ninfer-serve.exe` + 9 个 DLL + 3 个自检脚本。**DLL 必须与 exe 同目录**，别拆开 |
| **使用说明类** | `docs\` + `README-内测包.md` | 本页是入口；`README-内测包.md` 是给人看的短版 |

---

## §1 落地（5 步，每步都有判据）

**1. 整包解到纯 ASCII 路径下**（例 `D:\betakit\`）。
判据：路径里一个非 ASCII 字符都没有。反例：`D:\模型\…` ⇒ 引擎报
`[json.exception.type_error.316] invalid UTF-8 byte at index 4`（原生 exe 读不了中文路径）。

**2. 校验完整性。**

```powershell
powershell -ExecutionPolicy Bypass -File .\engine\自检-内测包.ps1
```

判据：最后一行 `KITCHECK_VERDICT=PASS`（34 项哈希全对）。

**3. 卡匹配（架构）自检。**

```powershell
powershell -ExecutionPolicy Bypass -File .\engine\自检-引擎与卡匹配.ps1
```

判据：报 OK。**本包只含 sm_89（RTX 40 系）引擎**，架构不符是**直接失败、不会回退**——
不是"慢"，是**起不来**。4090 / 4080 / 4080 SUPER / 4070 系都是 sm_89 ✅。

**4. 先看一眼自己落在哪条档位**——见 §3 的内置档表。这决定你**会不会看到那段自适应**。

**5. ★ 模型先做体检（**不管模型是随包来的还是自己下的，第一步都是这个**）**

```powershell
powershell -ExecutionPolicy Bypass -File .\engine\自检-模型件.ps1 -Model .\models\<文件>.ninfer -VramGb <你的显存GB数>
```

它**只读文件、不启动引擎**，输出是 `KEY=VALUE`（可 grep），最后一行是 `MODELCHECK_VERDICT=`。
它一次回答三个问题：**① 这个文件完整吗 ② 它带什么投机头 ③ 你这块卡该用哪个 `--spec`。**

| 拿到的文件 | 它会打出 | 你要做的 |
|---|---|---|
| 本包 `models\` 的完整件 | `COMPONENTS=dflash2,mtp,text,vision`、`SPEC_CHOICES=dflash2,mtp` | 大卡：`--spec dflash2 --draft-tokens 4 --lm-head-draft`；**≤10 GB 卡：只准 `--spec mtp`**（见下） |
| **裸件**（公共源下的、或别人转的） | `COMPONENTS=text`、`ACCELERATION=NONE`、`MODELCHECK_VERDICT=BARE`、**退出码 3** | **绝对不要加 `--spec`**（加了会 `FATAL … missing component`），只能裸跑 ⇒ 实测 **71.9 tok/s**，同题面完整件 **276.8** ⇒ **慢 3.9×**。要加速只能换带头的完整件 |
| 下坏的 / 下成网页 / 老容器 | `MODELCHECK_VERDICT=FAIL` + `MODELCHECK_REASON=…` | 按 REASON 办：`truncated download` → 重下；`not a NInfer container` → 你存下来的是网页；`container version 2` → 得先转 v3 |

它同时会把**截断**当场抓出来（比对文件长度 vs 头部声明的 `files[0].payload_bytes`，并打出差值），
所以"下下来少了几个 G"这种事不用靠猜。

### ★ 8 GB 卡（以及一切 ≤10 GB 的卡）的硬口径：**只能用 MTP，绝不开 dflash2**

原因就是**显存**：dflash2 要**多常驻 1.3–2.07 GiB 的草稿权重**再加快照工作区；**MTP 头只要 332–430 MiB**。
差了整整一个多 G —— 这一个多 G 就是"起得来"和"起不来"的分界。

工具在 `-VramGb` ≤10 时**直接给唯一答案**，并打这一行：

```
SMALL_CARD_RULE=USE --spec mtp ONLY. Do NOT enable dflash2.
```

**这不是选项，是结论** —— 别自己换成 dflash2。

8 GB 卡还要**四条一起上**（⚠️ **未验**：我们没在 8 GB 卡上跑过本包）：

1. **只有 PQ2（三元）这一档有机会**；GSQ / Swift 光权重就 13.4 GiB，直接排除。
2. **`--vision` 一律不要。**
3. 池与宿主层压到最小（`--kv-capacity` 调小、`--host-kv-mib` 调小）。
4. 必要时再上 `--gdn-state-fp16`（≈ −130 MiB）、`--embedding-q4 --lm-head-q4`（把 embedding 与输出头降到 4bit）。

> **诚实提醒**：PQ2 权重 **8.16 GiB**，8 GB 卡**本来就装不下**。上面这几条是"能不能挤进去"的全部指望，
> **必须实测**，**不许当已验读**。挤不进去就是挤不进去，别硬烧时间。

### 模型如果不是随包来的（我们只发引擎 + 文档，模型给你下载链接）

**文件名必须逐字对上 `start-*.bat` 里引用 `models\<名字>`**，下完放进 `models\`，**别改名**。
下载方式用能**断点续传**的（`curl.exe -C - -o <文件名> <链接>`），别用浏览器右键另存。
下完**先跑第 5 步那个体检**，过了再起服务。

**要加速（裸件 + 想要投机头）目前没有随包工具**：注入头需要上游的 artifact 库 + Python，
本包按设计**不含 Python**。所以现在只有两条路：**(a) 找我们要带头的完整件**；
**(b) 等我们把注入工具做成零依赖再发**（我们 09-27 做过这件事，方法在我们这边，工具还没出包）。
**在拿到之前，裸件就按裸跑用，`--spec` 一个都别加。**

---

## §2 起服务（三个样例，逐字 argv）

| 样例 | 端口 | 模型（`models\` 下） | 大小 | 用哪个投机头 |
|---|---|---|---|---|
| **`start-pq2.bat`** | 8091 | `Ternary-Bonsai-2-27B-ninfer-v3-mtponly.ninfer` | **7.13 GiB** | `--spec mtp`（**小卡用这个**） |
| **`start-pq2-dflash.bat`** | 8094 | `Ternary-Bonsai-2-27B-ninfer-v3.ninfer` | **8.87 GiB** | `--spec dflash2 --lm-head-draft`（**最快**） |
| `start-gsq.bat` | 8092 | `gsq_rco_iq3_s_dflash2_prop.ninfer` | 13.99 GiB | `--spec dflash2 --lm-head-draft` |
| `start-swift-iq3s.bat` | 8093 | `swift-qwen38-rco-iq3s.ninfer` | 13.99 GiB | `--spec dflash2 --lm-head-draft` |

> ⚠️ **两个 PQ2 的旗标是反的，拿错就是起不来**：
> `…-v3.ninfer`（带 dflash2+proposal）要 `--spec dflash2 --lm-head-draft`；
> `…-v3-mtponly.ninfer`（只有 mtp）要 `--spec mtp`，**并且绝不能加 `--lm-head-draft` 或 `--vision`**
> —— 加了会 `selected proposal head is absent from artifact` 直接拒启动。
> 不确定就先用 `engine\自检-模型件.ps1` 探一下，它会告诉你这个文件该用哪个 `--spec`。

除投机头旗标外，四个样例的其余参数完全一样：

```bat
ninfer-serve.exe "models\<该档模型>" ^
  --host 127.0.0.1 --port 8091 --model-id qwen3.8-27b ^
  --max-context 262144 --kv-capacity 17920 --kv-dtype k8v4 --host-kv-mib 16384 ^
  --prefill-chunk 1024 --spec dflash2 --draft-tokens 4 --lm-head-draft ^
  --default-max-tokens 32768 --default-reasoning-effort none --max-concurrency 1 ^
  --max-shared-prefixes 0 --vision ^
  --presence-penalty 0 --temperature 0.7 --top-p 0.9 --top-k 20
```

**两个不能动的地方**：

- `--max-shared-prefixes 0` —— **必须带**。不带会在"超池题面 + 逐字节重发"时把实例打砖（§7 第 2 条）。
- **五个 `NINFER_*` 环境变量**（§5）——**只有环境变量形式，命令行没有对应参数**，必须由启动器/父进程注入。

---

## §3 ★ 自适应：**哪几秒是引擎在测你的卡**

这是你最该先知道的一段。

**它会自己测自己。** 引擎第一次在"没见过的卡"上启动时，会**在你的卡上实跑一遍候选调度**，
把这块卡的最优路由（哪条 kernel 在哪个 T 区间用哪个形状）测出来。**这段发生在加载权重之前。**

### 3.1 判定只看两行日志（逐字）

```
INFO  calibrating routes for <hardware_class> (<N> SMs)          ← 出现这行 = 正在自适应
INFO  device profile <hardware_class>: <K> routed keys (<origin>) ← 自适应结束（或命中了已有档）
```

- **只有第二行** ⇒ 命中了内置/已存档位，**这次没有自适应**，启动直接进。
- **两行都有** ⇒ 这次跑了自适应，**此时千万别杀进程**（看起来像"卡住不动"，其实在测）。

### 3.2 多久

**一次性，约 10–50 秒**【实测；出自单档 8090 包那批，**本包形状未复测**】。
你自己的准确值 = **这两行日志的时间戳差**，别用我们的数字当你的 SLA。

### 3.3 结果落在哪、下次还测不测

- 落盘：`%LOCALAPPDATA%\ninfer\device-profiles.json`
  （取不到就退到 `%XDG_CACHE_HOME%\ninfer\` → `~/.cache/ninfer\`）。
- **同一个进程内只测一次；测完写盘，下次启动直接命中，不再测。**
  写盘失败会在日志里报 `device profile not saved to <path>: <原因>`——**这不影响本次运行**，只影响"下次还要再测一遍"。
- 想**强制重测**：`--device-profile calibrate`。想**完全不装档**（逐位不带自适应）：`--device-profile off`。

### 3.4 内置档表（逐字，取自引擎的 `device_profiles.json`，共 7 条）

| hardware_class | 架构 | SM 数 | 典型卡 |
|---|---|---|---|
| `nvidia-geforce-rtx-3090-sm86` | sm_86 | 82 | RTX 3090 |
| `nvidia-geforce-rtx-4090-sm89` | sm_89 | 128 | RTX 4090 |
| `nvidia-geforce-rtx-5090-sm120` | sm_120 | 170 | RTX 5090 |
| `nvidia-rtx-pro-6000-blackwell-workstation-edition-sm120` | sm_120 | 188 | RTX PRO 6000 工作站版 |
| `nvidia-rtx-pro-6000-blackwell-max-q-workstation-edition-sm120` | sm_120 | 188 | RTX PRO 6000 Max-Q |
| `nvidia-rtx-pro-6000-blackwell-server-edition-sm120` | sm_120 | 188 | RTX PRO 6000 服务器版 |
| **`nvidia-geforce-rtx-4080-super-sm89`** | sm_89 | 80 | RTX 4080 SUPER |

**匹配规则 = `hardware_class` 字符串 + SM 数，两者都要对上。**

⇒ **直接对号入座（这三类你会看到什么）**：

1. **卡在表里且架构 = sm_89**（4090 / 4080 SUPER）⇒ 启动**不会**出现 `calibrating routes`，
   只有 `device profile …: K routed keys (built in: …)`。**这段自适应秒数对你是 0。**
2. **卡是 sm_89 但不在表里**（例：4080 非 SUPER、4070 Ti、4070、4060…）⇒ **会出现自适应**，
   一次性（§3.2），测完写盘，**第二次启动就没有了**。
3. **卡是 sm_86 / sm_120**（3090、5090、PRO 6000 虽然**在档位表里**）⇒
   **本包跑不了**：本包只含 sm_89 引擎。档位表里有它 ≠ 本包含它的引擎；要用得换对应架构的引擎。

---

## §4 起来之后的判据（三行日志 + 两问）

启动日志必须出现（逐字，本包口径）：

```
INFO  loading weights | 8.16 GiB                                     ← 权重进去了
INFO  capacity | KV 17,920 tokens, k8v4, explicit | pages 280/4,096 | runtime … GiB | free … GiB
[ninfer] reuse host-backed: on
```

再加 `pinning host KV | 16.0 GiB`。

**两问**：

1. `GET /v1/models` 返回 **200**，且 `context_window = 262144`。
   （⚠️ **端口在监听 ≠ 可服务**：判就绪只看这个 200，别用 `netstat` 有结果当通。）
2. 发一条真请求，响应里 `timings.predicted_per_second` 与 `00-三档口径与读数.md` 同量级。

### ★ 本包的全部意义：**第二轮不重填**

同一会话**追问一次**，日志里应出现：

```
INFO  req#N done | openai-chat | … | prompt <大数> | … | cache <大数> (≥90%, …) | TTFT <明显变小> | …
```

判据两条同时成立：**`cache` 百分比 ≥ 90%**，且 **TTFT 掉到百毫秒量级**。
（本机 PQ2 实测：80,063 token 冷启动 TTFT 70,829 ms → 第二轮 **cache 100% / TTFT 126.5 ms**。）

**这一条不过，其它数都别信** —— 它才是"显存压小 + 长上下文 + 每轮不全量重填"三条同时成立的判据。

---

## §5 KVMem（ring + host-backed）怎么调用

### 5.1 五个环境变量，缺一不可

```
NINFER_KV_WINDOW=16384          # 常驻窗口（设备上真的留多少 token）—— 也是"环是否启用"的门（B30 起）
NINFER_KV_RETRIEVE=8192         # 检索预算 —— 缺它 = 静默答错（见下）
NINFER_KV_RING=1                # 历史开关：新件里已不再被读取（环的门看上面那行）
NINFER_HOST_PAGEABLE=1          # 宿主页可换入
NINFER_KV_REUSE_HOSTBACKED=1    # 允许复用宿主驻留页
```

**没有 CLI 参数**，只能环境变量。**为什么必须五个**：只给 WINDOW + REUSE 也能启动、缓存也显示 99.9%，
但**长题面会答错**——ring 只留最近窗口，没有 `RETRIEVE` 就把中段搬不回来，
而且**零错误行**（这是最危险的失效形态）。判据：40k 题面**正中**那根针答不出来。

### 5.2 池算术（你的显存账）

```
池页数 = 窗口/64 + 预填块/64 + 8(余量) = 16384/64 + 1024/64 + 8 = 280 页 = 17,920 token
```

- `--kv-capacity 17920` 给的是**设备上留多少**（不是上下文上限）；
- `--max-context 262144` 是**逻辑上限**（原生窗口），放不下的页落宿主层 `--host-kv-mib 16384`；
- **`--kv-capacity` 必须 < `--max-context`** 才进 ring 语义，且此时 **`--max-concurrency` 必须 = 1**（单路）。

### 5.3 长题面怎么发

**一次给全，不要自己切。** 引擎会自己 prefill、把放不下的页换到宿主层、下一轮按前缀复用。
你切碎了反而拿不到复用。

---

## §6 自适应之后：针对**你这块卡**的微调（两步）

### 第一步：把档"烘"进包里（便宜、一次性，可选）

自适应结果默认只落在**你这台机器的用户缓存**里（§3.3）。若你要**换机器也不重测**，
就把该档并进引擎源码的 `src\runtime\engine\device_profiles.json`——⚠️ **改这个 JSON 必须重新 configure 重新编**，
因为它是 configure 期烘进生成源码的。

**判据（能变红）**：启动日志出现该卡的 `device profile …: N routed keys (built in: …)`，
且**不再出现** `calibrating routes`。把外部档移走后必须回落到内置档（否则判据是假的）。

### 第二步：三个旋钮（按判据调，别凭感觉）

| 旋钮 | 作用 | 什么时候动 | 判据 |
|---|---|---|---|
| `--kv-capacity <页数×64>` | **显存 ↔ 延迟**的主要交换 | 显存宽裕、只求最低延迟：把池放大到 ≥ 前缀页数 | 启动日志 `pages X/Y` 与 `runtime … GiB`；TTFT 变小 |
| `NINFER_KV_WINDOW` | 常驻窗口大小 | 显存紧就压小；压太小会掉命中率 | 第二轮 `cache %` 不能掉；掉了就是压过头 |
| `--host-kv-mib` | 宿主层容量 | 题面很长、宿主层被填满时 | 日志 `host … GiB KV`；不够会有换出/重算迹象 |

**动的顺序 = 一次只动一个，每次都用 §4 那两条判据复验**（`cache %` 与 TTFT）。
⚠️ 不要同时改两个旋钮，否则你不知道是谁的功劳。

> **`--kv-capacity` 与 `--prefill-chunk` 的关系**：两者并非争用同一份显存 —— 分块在池公式里只占 `page_count(prefill-chunk)` 页
> （每 256 token ≈ 4.4 MiB），占用较大的是分块自身随 chunk 线性增长的临时工作区。
> 因此"提高池容量必须降低分块"不成立；处置顺序与**按可用显存（8 GB / 10 / 12 / 16 / 24 / ≥32 GB）配档的方法与建议**
> 见 `03-基础部署后的调优方案.md` **§1.5**；8 GB 档的完整配方见仓根 `配方-8GB档-实测支撑-20261008.md`。

**边界（诚实的）**：本包**不做**内核级/调度级的针对性调优（三元 GEMM 档位阈值、tile 形状那些），
那需要重编引擎且收益未评估。`--device-profile` 已经把"调度层按卡自适应"这件事做掉了。

---

## §7 注意事项：会炸 / 会静默错的（**照用会出事**）

| # | 现象 | 真因 | 你要做的 |
|---|---|---|---|
| 1 | 长题面**中段**的内容被答成"不存在"，短问答正常，**日志全绿** | 漏了 `NINFER_KV_RETRIEVE` | 五个变量补齐（§5.1） |
| 2 | 某请求 500 后，该实例**之后所有请求 503** | 「题面页数 > 池 280 页」且**逐字节重发同一题面** ⇒ `active KV snapshot full page is not stable` | **argv 里必须有 `--max-shared-prefixes 0`**（引擎默认值是 `max(max-concurrency,7)`，**必须显式传 0**） |
| 3 | 只输出 **2 个 token** 就 EOS（点名 `swift-qwen38-rco-iq3s.ninfer`） | 行粒度缩放的 KV 精度打垮离群通道 | **`--kv-dtype k8v4`**（或 `bf16`）。别用 `fp8`/`rk8v4`/`int8` |
| 4 | 加了 `--spec X` 起不来：`missing component X` | 该制品没有这个组件 | **先探组件再配 `--spec`**；本包三档都有 `dflash2`，能直接用 |
| 5 | 路径带中文 ⇒ 启动即报 `invalid UTF-8 byte` | 原生 exe | 包与模型放纯 ASCII 路径 |
| 6 | `0xC0000135` 秒退 | 缺 DLL | 确认 `PATH` 指到 `engine\`（以及 `engine\x64`）；`engine\` 里已自带 cublas/cudart |
| 7 | 磁盘层（`--disk-kv-*`）"开了没变快" | 本构建 `NINFER_DIRECTORSTORAGE=OFF`，回灌 ≈ 全量重填 | **灰度阶段不要开磁盘层**；判"回灌有没有生效"看 `cache_n`，**别看 TTFT** |

完整版与读数：`05-已知问题与禁忌.md`（七条，每条带症状→真因→处方）。

---

## §8 已知 bug 清单（我们会修，你先按处方绕）

| 症状 | 状态 | 处方 |
|---|---|---|
| 超池题面逐字节重发 ⇒ 实例打砖（§7 第 2 条） | **已知，配置层已绕开**；引擎侧真修待做 | 带 `--max-shared-prefixes 0` |
| 行粒度 KV ⇒ Swift 制品只出 2 token（§7 第 3 条） | **已知，已换默认档** | 用 `k8v4` / `bf16` |
| 磁盘层零收益 | **已知**，等带 DirectStorage 的版本 | 别开 |
| 某 Q4_K_M 制品偶发"只答 1 个字符"（三种 KV 精度都会，**未定位**） | **已知未定位**；该制品**不在本包内** | 遇到就 **×3 重复判定**，别用 n=1 下结论 |
| `.idx`/`diskvk_*` 体积看着很大 | **不是 bug**，是 open 期预分配 | 别当写入量 |

---

## §9 思考循环（空交付 / 固定点 / 半截）

**一句话**：这类退化**在模型侧没有任何信号**——HTTP 200、`finish_reason=stop`、日志零告警。
它的触发器是**"把上一轮的推理回灌进下一轮"**。所以**必须在你的平台层拦**。

**最小处方（三层，至少上第 1 与第 3 层）**：

- **L1 参数层**：`--default-reasoning-effort none`（本包默认已设）+ **不回灌上一轮推理** + `max_tokens` 给足
  （给少了会正好截在思考中间 ⇒ 正文 0 字）。
- **L2 采样层**：按"这一轮是否思考"配两套预设，别一套打天下。
- **L3 平台层护栏**：判四态 —— `EMPTY`（正文 < 400 字符）/ `TRUNCATED`（围栏未闭合 /
  `</svg>` 缺失 / `finish_reason=length`）/ `FIXED_POINT`（与上一轮逐字相同）/ `SAME_PLAN`（计划行重复）；
  判不过就**带纠正指令在同一次调用内重发**。

判据阈值、纠正文案原文、两种模式对照、"接收方要实现什么"的表，
以及**我们的参考实现**都在 `04-卡死与循环的防治.md`。

---

## §10 ★ 回执模板（**把这段填了发回给我们**）

> 请**原样复制**下面这段，填完发回。带得越全，我们一轮就能定位。

```
【NInfer 灰度回执】
1. 档位：PQ2 / GSQ / Swift-IQ3S     端口：______
2. 卡：nvidia-smi --query-gpu=name,compute_compute,driver_version,memory.total --format=csv 的输出
   （逐字贴）
3. 引擎 argv（逐字贴，含 --host-kv-mib / --max-context / --kv-capacity / --spec）
4. 你改过 argv 或环境变量的哪几处：______（改过必说）
5. 自适应：启动日志里有没有出现 `calibrating routes for …`？
   有 → 请贴这两行 + 它们的时间戳差（= 你这块卡的真实校准秒数）：
        calibrating routes for ______
        device profile ______
   没有 → 贴 `device profile … : N routed keys (…)` 那一行
6. 这三行的逐字原文：
   loading weights | ______
   capacity | ______
   [ninfer] reuse host-backed: ______
7. 第二轮复用的那一行（逐字）：
   req#N done | … | cache ______ | TTFT ______ | decode ______
8. 题面长度（token）+ 你问的是什么形态（单问 / 多轮 / agent 循环 / 要整份文件）
9. 现象：______（贴原始响应 JSON 的 finish_reason 与 usage；**别只贴文字**）
10. 完整引擎日志：______（文件或粘贴）
```

**我们最想要的三个数**：① 你的**校准秒数**（第 5 条）② 你的**第二轮 `cache %` 与 TTFT**（第 7 条）
③ 你那块卡上的 **decode tok/s**。这三个数一到手，就能判断"是配置问题还是卡的问题"。

---

## §11 文档地图

| 想知道 | 看 |
|---|---|
| 三档实测读数（速度/显存/长上下文） | `00-三档口径与读数.md` |
| 坑与禁忌（七条，带读数） | `05-已知问题与禁忌.md` |
| 思考循环怎么防（含护栏参考实现） | `04-卡死与循环的防治.md` |
| 引擎全量开关与编译 | `01-部署与编译总白皮书.md`（⚠ 形状数字见页首封条） |
| 出事怎么查（症状→处方） | `02-排错手册 · 预案与处方.md` |
| 从零复现 / 兜底 | `08-从零复现兜底.md` |
| 按卡差异与档位表 | `11-按卡差异速查.md`（⚠ 形状数字见页首封条） |
| 许可、署名、我方改动 | `09-声明与链接.md` |

**分发包，按现状提供。** 请把 §10 的回执发回给我们。
