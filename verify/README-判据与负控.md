# `verify/` · 判据脚本（**每个都写了"怎么变红"**）

> 这一组脚本是**交付件里实际跑过的那几个**：它们在"分发包"上跑出过 `KITCHECK_VERDICT=PASS (0 failures)`。
> 之所以连脚本一起发，是因为**白皮书里的结论都要能被别人重跑**——只会变绿的判据等于没有判据。

## 0. 运行前提

- Windows + **Windows PowerShell 5.1**（脚本是纯 ASCII，故意如此：5.1 会把无 BOM 的 `.ps1` 按 ANSI 读，
  脚本里写中文就会静默出错）；中文文件名一律**按内容特征查找**，不在脚本里写字面量。
- 需要有一个待校验的目录（"包"）。`-Kit <目录>` 指到它；不给就取脚本所在目录的上一层。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\verify\verify-kit-manifest.ps1 -Kit <包目录>
```

## 1. 五个脚本各管什么

| 脚本 | 它证明什么 | 判据行（成功时） | **负控（怎么让它变红）** |
|---|---|---|---|
| `verify-kit-manifest.ps1` | 清单里每个文件的 sha256 与磁盘一致 | `ok=<N> mismatch=0 missing=0` | 改一个字节 ⇒ `mismatch`；删一个文件 ⇒ `missing` |
| `gate-内测包-清单与形状.ps1` | **双向**：清单里的都在 + 磁盘上没有"清单外"的文件；并检查引擎/DLL/启动器形状与启动器 argv 必需项 | `KITCHECK_VERDICT=PASS` | ① 新增一个未入册文件 ⇒ `unlisted`；② 抽掉一个 DLL ⇒ `runtime dll` 变红；③ 把启动器里的 `--max-shared-prefixes 0` 删掉 ⇒ 该行变红 |
| `gate-ptq1档-清单与形状.ps1` | 同上，针对 PTQ1 档（含 `NINFER_TERNARY_PTQ1_FAST=1` 这条负载开关） | `KITCHECK_VERDICT=PASS (0 failure(s))` | 删掉启动器里那行 `NINFER_TERNARY_PTQ1_FAST=1` ⇒ 变红 |
| `自检-引擎与卡匹配.ps1` | 本机这张卡能不能跑这个包（读 `nvidia-smi` 的 compute capability 与引擎支持的架构比对） | `HWCARD_VERDICT=PASS` | 拿 `sm_86` 的包在 `sm_89` 的卡上跑 ⇒ 架构不符，明确报错而不是"慢" |
| `verify-arch-engine.ps1` | **本卡端到端验收**：起服务 + 过池告警臂（题面**中段**针）+ 装得下臂 | 起服务成功、过池臂答对中段针、装得下臂无告警 | 题面**中段针丢失 = FAIL**；`finish_reason=length` 或空答案 ⇒ **INCONCLUSIVE（不算漏针）** |
| `verify-tests-ctest-negcontrol.ps1` | 受影响测试真跑通，且失败分类表被测试钉住（**源码批 2026-10-07 实测 `VERDICT subset = PASS`**） | `VERDICT subset = PASS` + `negative control verdict: PASS(...)` | 翻 `failure_class.h` 一行 ⇒ ctest 必须 **failed** 并打出 `std::bad_alloc must classify as Capacity`；字节级还原后必须回到 **passed** |
| `verify-b01-overpool-429.ps1` | 装不下的请求被**可见拒绝**且引擎不被打死（429 + 编号消息 + 后续 200 + 存活 + 指标） | `refused with 429 = True` · `message carries a number = True` · `engine still serving (200) = True` · `server survived = True` | `-ShortProbeOnly`（不打断言）⇒ `refused` 必须 **False**、指标必须 **0**；`-ExpectStartRefusal` 是守卫正控，⚠️ **在本仓二进制上不成立**（见 B21），别拿它当红控 |
| `verify-256k-nonring-regression.ps1` | 关环（KVMem off）下的 256k 长题 + 4 路并发 + checkpoint 复用，用的是 PR #2 自带的校验脚本 | `script exit = 0` 且报告每条 `passed=True`；`near_256k.prompt_tokens ∈ [250000, 262144]`；`cached_tokens > 0` | `-RedControl`（`--long-rows 200`）⇒ `script exit` 必须非 0（实测 **1**） |
| `resolve-build-env.ps1` | **不是判据脚本**：上面两个需要工具链的脚本用它从**被测构建目录的 `CMakeCache.txt`** 反推 cl/ninja/CUDA 根/vcpkg 根，vcvars 用 `vswhere` 找 —— 所以这些脚本不带作者的机器路径 | `Show-BuildEnv` 打印出的每项都非空 | 任一字段取不到 ⇒ 脚本**打印** `could not derive <字段>`（不静默兜底）；把 `-BuildDir` 指到一个没配过的目录 ⇒ 直接以 "no CMakeCache.txt" 报错退出 |

## 2. 两条使用纪律（踩过才写的）

1. **`/v1/models` 返回 200 不等于能服务** —— worker 死后它照样 200。判活必须**真发一条请求**。
2. **同一个探针别用两种读法**：本篇所有"红控"都要在**同一支二进制**上做（例如检索窗口 64 → 256），
   否则你分不清是改动生效了还是换了个东西。
3. **改头不触发重编（本树实测 2026-10-07）**：本构建用 CMake 的 scanned C++20 规则，`DEP_FILE` 指 `<obj>.ddi.d`
   （**模块**依赖）⇒ **普通 `#include` 的头不是该 obj 的输入**，改它之后 `ninja` 会回答 `no work to do`。
   照原样读，你会拿**旧二进制**的绿灯当成新代码的结果（我们第一次跑分类表负控就是这样得了假绿）。
   修法：负控里**显式删掉 `.obj` 与 exe** 强制重编（`verify-tests-ctest-negcontrol.ps1` 已这么做并打印 `forced: …`）。
   系统含义：**这棵树的增量构建对头文件改动不可靠**，改头后请整目标重编。
   ⭕ **2026-10-07 深夜补：这条同样咬"测试侧"，而且症状更凶（实测）** —— 改了 `src/serve/serve_options.h` 之后，serve 侧 36 个消费者 obj 是重编了的，但**链接进 `ninfer_tests.exe` 的那几个测试 TU 没重编**，于是同一个结构体在不同 TU 里布局不一致。症状**不是编译错误**，而是：
   ① `ninfer_request_log_test` / `ninfer_serve_options_test` 直接 **fail-fast**（`0xC0000409`）或 **SEGV**；② `ninfer_openai_schema_test` **判错**三条 post_thinking / reasoning-effort 合并语义的断言；③ `ninfer_anthropic_schema_test` 在整跑里红、隔离跑绿（假红）。
   **真正的测试侧消费者（改 `serve_options.h` / `generation_service.h` / `http_server.h` / `request_log.h` / `translate.h` 之一就要重编它们）**：`test_anthropic_schema.cpp` · `test_http_error_handler.cpp` · `test_openai_responses.cpp` · `test_openai_schema.cpp` · `test_request_log.cpp` · `test_serve_options.cpp`（对应 obj 目录 `build-ninja\tests\CMakeFiles\ninfer_<用例名>.dir\`）。
   **修法（实测有效）**：删掉这些目录里的 `*.obj`（连带 `.dd`/`.ddi`）再 `ninja all` —— 本例删 7 个目录 21 个文件、重编 24 步后，**7/7 用例全绿、四项验收闸门 4/4**。
   **纪律**：**改头文件之后，验收不要用"整跑过了"当唯一证据**；先按"头 → 直接/间接消费者 TU"清单强制重编，再跑；否则你会看到一组凭空出现、且**无法用逻辑解释**的红（本例就是，差点被当成 S2/S3 的功能回归）。
4. **未知开关会被静默忽略**：参数解析链以 `src/serve/serve_options.cpp:992`
   `} else if (parse_dispatch_options(arg)) { }` 结束，**没有 else-throw**，全文件只有 `--help` 与 `argv[1]` 两处校验。
   ⇒ 每个开关都要对源码核实，并抓服务器**自报字段**反证它生效了（`INFO capacity | KV <N> tokens, <dtype>, explicit | pages X/Y`）；
   本机就踩过：脚本里传的 `--no-kv-lease-growth` **在本树不存在**，被静默吞掉，而那一跑看起来是"成功"的。
   ⭕ **2026-10-07 复核：本条的前提不成立（实测证伪）** —— 引擎**会拒绝**未知开关：那个
   `parse_dispatch_options` lambda **自己**以 `else { throw std::invalid_argument("unknown argument: " + arg); }`
   结束（`serve_options.cpp:709-711`）。四臂实测（本仓件与出厂件）：`--definitely-not-a-flag`、
   `--no-kv-lease-growth`、`--vram-reserve-mib` **全部** `unknown argument: X` + **exit 1** + usage 全文；
   对照臂「不给开关」引擎继续往下走 ⇒ 两态可区分。**同一事实在 `docs/02-排错手册`（§1 表）
   与 `docs/08-从零复现兜底` 里记的就是正确行为**（`unknown argument: --disk-cache` ⇒ 删掉它）。
   **本条里仍然成立的**：① 要抓服务器自报字段反证配置生效（这永远成立）；② 别的批（F2 之后）的启动器
   参数**照抄到本树会启动即拒**（`--no-kv-lease-growth` 只有肯定式的 `--kv-lease-growth`，默认 `false`）。
5. **"真红被读成绿"：崩溃 / 缺 DLL 的用例会被漏数（2026-10-07 实测，已修）** —— `verify-tests-ctest-negcontrol.ps1`
   原先有三处叠加缺陷：① 用 `[xml](Get-Content -Raw)` 读 JUnit（前言声明 `encoding="UTF-8"` 时 .NET 抛异常）⇒ 返回 `$null`；
   ② 按 `<testsuites><testsuite>` 找用例，而本项目 JUnit 的**根元素就是 `<testsuite>`** ⇒ **永远解析到 0 条**；
   ③ 退回控制台解析的正则只认 `Passed|Failed|Skipped|Timeout`，而崩掉的用例打的是
   `… Exit code 0xc0000135***Exception: 550.57 sec` ⇒ **该用例被直接丢掉**。
   **后果（当真发生过）**：4 个用例里 3 个崩，脚本报 `passed=1 failed=0 of 1` ⇒ **`VERDICT = PASS`**，而 `ctest` 自己返回 8。
   **修法**：JUnit 改成按路径 `XmlDocument.Load()` 读、两种根元素都接受、控制台正则补崩溃形态，并加两条**硬不变量** ——
   **`ctest` 自己的返回码**与**子集应有条数（`ctest -N -R` 的 `Total Tests`）**必须与解析结果一致，否则一律 `NOT-CLEAN` 并打印原因。
   **规矩**：解析器的"少算/漏报"必须由**外部权威**兜住，不能只靠 `tests>0 且 failed==0` 这种自证式判据。
6. **跑 ctest 前把 CUDA bin 放进 PATH**（`E:\cuda-13.3\bin\x64`；`cudart64_13.dll` 只在那里，`bin\` 下没有）：
   否则多数用例以 **`0xC0000135`（STATUS_DLL_NOT_FOUND）** 失败，**而且可能表现为"先卡几分钟再失败"**（实测 `ninfer_hadamard_transform_test` 崩前挂了 550 s）。
   **这不是代码回归** —— 同一二进制带上 `bin\x64` 后 4/4 通过（2.31 s）。
7. **A 档跑批（2026-10-07）留下的四条可复用判据**（都已在发布线二进制 `3B4103B4…` 上跑过；驱动器只做编排，不新造仪器）：
   - **kMin 工作页判据 = `pool − sink ≥ 16` 页**：池 58/59（`free` 2/3）⇒ 工具答错；池 60/61（`free` 4/5）⇒ 答对（复用 `probe-kmin-sweep.ps1`，sink 固定 44 页）。**⚠️ 2026-10-07 深夜更正**：不要写成"`free ≥ 4`"——`free` 只是**分给检索预算后的余数**；把 `RETRIEVE` 提到 1536 后 `free` **恒为 0** 而池 60 仍答对 ⇒ 控制量是 `pool − sink`（三组 × 三点边界一致在 16 页；`RETRIEVE`/`SHARE` 都不移动它）。**怎么变红**：若 `pool − sink = 15` 就答对、或 `= 16` 仍答错，则阈值不是 16。
   - ⛔ **2026-10-07 19:0x 再更正：`pool − sink ≥ 16` 页这条也被证伪 —— 控制量其实是"池装得下整个题面"（`prompt ≤ pool`）**。那三个"16 页"是**固定题面 3,923 token + 固定 sink 44 页**下的巧合。判据（`probe-kmin-sweep.ps1`，新增 `-FillerLines` 把题面按比例放大）：

     | 臂 | 池 | 题面真值（引擎自报） | `pool − sink` | 工具题 |
     |---|---|---|---|---|
     | A 对照 | 60 页 | 3,923 token（62 页） | 16 | **答对** |
     | B 判别 | 60 页 | **7,563 token（119 页）** | **16** | **答错** |
     | D | 62 页 | 7,563 token（119 页） | **18** | **答错** |
     | C | 120 页 | 7,563 token（119 页） | 76 | **答对** |

     ⇒ slack 16 / 18 都会答错 ⇒ slack 不是控制量。**引擎自己把条件写在告警行里**（逐字）：`prompt exceeds the resident Device KV pool: prompt 7563 tokens (119 pages) > pool 3840 tokens (60 pages) … raise --kv-capacity to at least the prompt's token count (measured: prompt >= pool is the condition that reproduces it)`。
     **⇒ 该用什么当判据**：① 启动守卫只管网**结构**（`sink + 工作页 ≤ pool`，已修，见 B21）；② **每请求**的判据是**那行 over-pool 告警**（题面 > 池 ⇒ 中段可能无声丢失、答案按未核对待），**不是**固定页数的松弛量；③ 先前"守卫欠保护 12 页、应改成 16"的推论**作废**。⚠️ 注意 A 臂也是 over-pool（3,923 > 3,840）却答对 ⇒ **告警 ≠ 必错**。
   - **`NINFER_KV_SINK` 两份复刻同值**：`SINK=640` ⇒ 预算行 `skeleton=10`；`SINK=1920` ⇒ `skeleton=30`（若不等即为两份漂移）。
   - **池页数 == 逻辑页数 ⇒ 环被关**：池 1024 页 + `--max-context 65536` ⇒ 日志里**没有** `[ring] budgets:` 行（若出现，说明判据或引擎行为变了）。
   - **17920 交付档工具可见性**：池 280 页 + `SINK=2816` ⇒ 工具题必须答对（读同一探针的 `tool=` 列）。实测 **100%**。

   ⚠️ **2026-10-07 晚追加（B1 重跑）**：墙钟必须与**引擎自报**分开记 —— `curl` 端到端推出的 decode ≈ 49 tok/s，而引擎 `req#1 done` 行自报 **59.7 tok/s**（端到端含 HTTP 开销），**两者不是同一把尺子，不许混用**；`TTFT`/`total` 的单位是 **ms**，别当秒（我第一次就当成了秒，派生出的解码率变成了负数）。
8. **全量套件（258 项）在同一个 `ctest` 会话内存在"顺序效应假红"**（2026-10-07 实测，A 级）——
   把 T1 的红**单独成组重跑**：早段 16 红 ⇒ **15 绿**；后段 18 红抽样 ⇒ **13 绿**；其余 28 红 ⇒ **18 绿**。
   ⇒ **62 红里 46 个是会话内顺序效应**，另外 16 个隔离也红（**15 个是 `gdn/attn_input_proj` 家族的上游既有缺陷*、1 个是本机缺 Python `jinja2`），**0 个由本仓改动引入**。
   **已夹到一条最小触发链**：同一会话内 `ninfer_device_sync_empty_test`(#20) → `ninfer_decode_graph_test`(#21) **必红**（报 `NINFER_CUDA_SYNC must be spin, blocking, yield, or auto`）；换成 `device_sync_invalid`(#19) 则绿；单独跑也绿；**手工把该变量置"未设置 / 空串 / spin"后单独跑** ⇒ **三次全绿** ⇒ 不是"父进程设置环境变量即污染"（`tests/CTestTestfile.cmake` 里 **59/258** 个用例带 `ENVIRONMENT` 属性）。**泄漏路径未定位**（写清楚，别猜）。
   **复现仪（2~3 分钟）**：按注册顺序取累计 ≤240 s 的 130 项前缀块拼成正则，跑 `run-tests.ps1 -Subset <正则>` ⇒ **1:1 复现同段 16 红**。
   **规矩**：全量套件只用来"摸清有哪些红"，**每个红都要在隔离或换序后复测**才允许判红；四项子集仍是验收闸门。
   \* 该上游缺陷的判据过宽问题记在 `已知问题-初期版本.md` 的 **B26**：`t2_two_parent()` 把"两父都非三元、但格式不同"（= 同文件明文要求的 `Q4+Q5` split 构型）也当错误抛；**这不是"不支持三元"**。

## 3. 与白皮书的关系

白皮书 §8.4 的判据表逐条对应本目录的脚本；§9 的读数就是这些脚本与孪生脚本（模拟安装、门禁跑台）跑出来的。
读数口径、正控/负控/红控的分类见白皮书 §9–§10。

## 4. 不在这里的东西

- **模型体检**（读 `.ninfer` 容器、判断带什么投机头、给出该用哪个 `--spec`）在 `tools/自检-模型件.ps1`。
- **模拟安装**（把包复制到全新 ASCII 路径、走完接收方全部步骤）与**批量重算清单/门禁**的脚本是本机运维脚本，
  带硬编码路径，未随仓发布；白皮书 §9.3 的读数是它们跑出来的，方法在 §8 里逐字写清了。

## 5. 2026-10-07 晚新增（本目录下）

- **`run-a-tier-batch.ps1` · `run-b-tier-batch.ps1`** —— **编排驱动**（只做编排：起停引擎、发请求、读引擎自报；**不新造仪器**）。A 档 7 问、B1 三杠杆速度、B2 多路真跑、B3 kMin 随 `RETRIEVE`/`SHARE`；B 档可 `-SkipB1/-SkipB2/-SkipB3` 分跑，B2 可 `-Lanes N`。
  ⚠️ 带**本机默认路径**（`-Exe`/`-Model`/`-OutRoot`/`-KminSweep` 都可传参覆盖）；跑法见脚本头注释与回执 §8.7–§8.9。
  两条纪律已内建：① B2 臂跑完**再发一条完成**证明引擎还活着（`/v1/models` 假健康，见 §2 第 1 条）；② 引擎起不来时**打印日志路径 + 尾部 ERROR 行**（否则现场只剩一句"没起来"）。
  ⚠️ **每调用一次会整表重写自己的 `b-tier.tsv`** ⇒ 不同臂请给**不同 `-OutRoot`**，否则先跑的行会被后跑的清掉（本条踩过）。
- **`读数-20261007/`** —— 上面脚本当天的**原始读数表**（TSV），文件名对应回执节次：`A档-a-tier.tsv` → §8.1–8.6 · `B档-b1-b2.tsv` → §8.8 · `B档-b2-4lanes.tsv` → §8.9 · `B3-*.tsv` → §8.3 的 kMin 判据 · `T1-full-258.tsv` + `T1-隔离-*.tsv` → §8.7。
  **2026-10-07 19:0x 补入（L0 批）**：`L0-驱动读数.tsv`（B27 起图失败率 10 次启动 + #17/#15 汇总）· `L0-17-假绿-*.tsv`（**假绿那一版**：池 280/sink 265，三格全 yes —— 保留它是为了记住"题面装进 skeleton 时该格不是判别格"）· `L0-17b-{A对照,B判别,C,D}.tsv`（**证伪 `pool − sink ≥ 16` 页的那四点**：题面 7,563 token 时 slack 16/18 都答错、slack 76 答对）· `L0-15-ring-{on,off}-*.tsv`（`max-context == 池` ⇒ 环被关）。
  **2026-10-07 深夜补入**：`B29-15臂-复跑.tsv`（B29 修复后那一族 15 条臂 **15/15 PASS**）· `四项验收闸门-B29后.tsv`（**4/4 PASS**）· `S2-配对重复-每臂3次.tsv` + `S2-判据结论.tsv`（S2 运行时统计与 C8–C11）· `B30-环两门矩阵.tsv`（**只设 `NINFER_KV_RING` ⇒ CLI 放行 + 引擎 FATAL；只设 `NINFER_KV_WINDOW` ⇒ CLI 拒；两个都设 ⇒ 200**）· `B24-空响应三臂.tsv`（不带预算 ⇒ `content` 空；带 `thinking_budget:1024` ⇒ 非空）。
  **2026-10-07 深夜再补**：`B24-遥测-两臂.tsv`（引擎自报的 `repeat_dup8`：无惩罚 **0.23829** vs `--thinking-presence-penalty 2.0` **0.19222**，与 §9.2 的离线读数吻合 ⇒ 遥测本身的验证见回执 §11）。
  原始**大**日志（JUnit XML、ctest 控制台全文、引擎 `err.log`、E2/E5 harness 臂日志）留在本机未随仓发布；需要时按 §8.7 的**复现仪**与本节第 7 条重跑。
- **`读数-20261008/`（2026-10-08 全量重编后的回归批 + 同日复核与测速 + 8G 档六件，26 件）** —— `批-五步汇总.tsv`（五步 rc 与耗时）· `件身份-体积与sha256.tsv`（新件 `9FA7490BCBC4825A…` = **795,030,528 B，比旧件小 47.4%**，这是关掉 `-lineinfo` 的直接判据）· `冒烟-池64.tsv`（http=200）· `四项验收闸门-直跑四条.tsv`（**4/4 rc=0**）· `数值不变性子集14条.tsv`（**14/14 passed**，不含 415 s 的 wide 臂）· `B24-服务端默认预算-四臂判决.tsv` + `B24-真实响应-D{1,2,4}*.json`（D1 无旋钮 ⇒ `finish=length` / `reasoning=2048` / **content 0 字符**（issue #3 症状复现）；D2 `--default-thinking-budget 1024` ⇒ content 非空、reasoning 1049；D3 `=0` ⇒ 拒绝启动；D4 旋钮 1024 + 请求 1500 ⇒ reasoning 1525 = **请求优先**）· `B30-只设RING.tsv`（http=-1，CLI 拒绝并点名正确变量）· `B30-只设WINDOW.tsv`（http=200）。
  ⚠️ **B24 的读法**：仪器内联的那个 `content_len` 量的是 **JSON 尾巴长度**、不是正文（实测会把"content 为空"的正确答案显示成 8377 字符）⇒ **必须解析引擎返回的 JSON**；本目录的 `B24-真实响应-*.json` 就是原始证据。
  ⚠️ **闸门口径**：ctest 子集要求**每个注册用例的可执行文件都存在**。只编 `ninfer-serve`/`ninfer_tests` 时 `ninfer_failure_class_test.exe` 不存在，ctest 记成 `Not Run`（看着像失败，其实是构建范围问题）⇒ 本目录的闸门读数用**直跑四条 exe 的 rc**。另：`tests\` 可能没有 FFmpeg/curl 运行库（staging 是 ALL 目标）⇒ 跑测试时 `PATH` 要含 `apps\`。
- **同日复核（已修 bug 逐个回判）** —— `复核-已修项-臂清单.tsv`（本轮跑的臂与 rc）· `复核-已修项-B21守卫-池45到48.tsv`（**B21 ✅**：池 45/46/47 被拒、48 启动成功并回 429）· `复核-B32-兜底开-判决段.txt`（**B32 ✅ 一个方向**：不显式关兜底 ⇒ 3/3 命中）· `复核-B32-显式0-判决段.txt`（**B32 ✅ 另一个方向**：显式 `0` ⇒ 0/3 静默答错；池容得下 2/2 正常）⇒ 兜底默认 ON、可被显式关掉，两个方向都按预期走 · `复核-窗口与精度可调性三档.tsv`（8G 档驻留窗口实测可调：窗口 65,536 ⇒ 66,432 tok/1,038 页/运行时 1.32 GiB + 权重 5.52 = **6.84 GiB**；窗口 32,768 ⇒ 526 页/790.8 MiB；`rk2v4-e8` @65,536 ⇒ 1,038 页/1.07 GiB，三档 `[ring] budgets` 自洽）· `复核-B01-关兜底后-可见拒绝.tsv`（**B01 ⚠️ 未复现**：兜底显式关掉后过池请求仍回 200 ⇒ 该拒绝路径需要"大输出租约 + 两层同时满"，本机仪器没有这个旋钮，**判为未验**，不是"已修"）。
- **同日测速（单臂）** —— `速度-PQ2-MTP12.tsv` + `速度-PQ2-MTP12-响应.json`：新引擎 + PQ2 8.87 GiB 件 + `--spec mtp --draft-tokens 12` + KV bf16 窗口 32,768（512 页，运行时 3.91 GiB），6 s 就绪，**解码 206.4 tok/s**、prefill 557.9 tok/s（题面仅 38 tok）、TTFT 68.8 ms，输出 292 token 顺序数，引擎自报 `mtp accepted 243/576 (42.2%)`。
  ⚠️ **比速度必须连配置一起报**：本仓已知的四组是①常驻 bf16 KV 池 262,144 + 图开 + 无投机 + PTQ1 ⇒ 解码 59.6–60.0 tok/s；②同配置关三杠杆/图 ⇒ 55.9–56.1；③过池 8k 池 + 环 + 主机降级 + `--spec mtp --draft-tokens 4 --lm-head-draft` ⇒ prefill 1.23k→715 tok/s、TTFT 27.4 s；④本条 PQ2 + MTP 12 ⇒ 206.4 tok/s。**社区 638.7/110.8 那两个数不是同一配置（12 GB 卡 + PQ2 + MTP），不可直接比。**

## 6. 2026-10-07 深夜补记（跑测试的两条操作事实 + 判据指针）

- **测试是一个多程序 bundle：`build-ninja\tests\ninfer_tests.exe`**（文件名带 `n`）。直接跑它时用法是
  `ninfer_tests.exe <程序名>`，`ninfer_tests.exe --list` 会列出全部程序名（程序名 = ctest 用例名）。
  ⚠️ 用 `infer_tests.exe`（少一个 `n`）去 `Test-Path` 会得到假阴性 —— 本机实测过，并因此误判"二进制不存在"。
- **三条 `*_routes_*` 注册跑的是同一条命令**：`ninfer_gdn_input_proj_conv_record_test`、
  `…_conv_record_legacy_routes_test`、`…_conv_record_unified_routes_test` 的 `add_test` 命令行**完全相同**
  （都是 `ninfer_tests.exe ninfer_gdn_input_proj_conv_record_test`），差别只在 ctest 的 **`ENVIRONMENT` 属性**。
  ⇒ 想跑"legacy / unified 路线"的臂，**只能走 ctest**（本目录的 `verify-tests-ctest-negcontrol.ps1` / 内部 `run-tests.ps1` 都行），
  直接调 exe **只能得到默认路线**（`ctest -N -V -R conv_record` 可以把三行命令原样打出来核对）。
- **一条时序参考**：`…_conv_record_*` 这一族 15 条臂合计 ≈ **788 s**，其中 `ninfer_gdn_replay_fold_wide_test`
  单项 **415 s** ⇒ 用 ctest 跑这一族要**后台跑**（内联 10 分钟必超时；本机踩过）。
- **判据与逐格读数（含"怎么变红"）的唯一归属处是 `实测回执与反馈.md`**：S3 请求级思考预算 + S2 分通道 presence penalty
  在 **§9**（S2 的口径 = 同题面 × 每臂 3 次 × 同一引擎实例，统计量 = 思考文本 8-gram 复读率 `dup8`，**跨臂比较必须等长前缀**）；
  B24「轮级空响应」的复现与缓解在 **§10**（判据看 `content` 是否为空，**不能只看 `finish_reason`**）。
  本节只记"怎么跑"，不复制那里的数字。

## 7. 2026-10-20 新增：缓存复用类断言必须用 **append-only 夹具**（外部实测回执 §6.1）

> 来源：外部 RTX 5080 16 GB 实测回执（2026-10-20）§6.1。**本节只新增规则，不改 §1–§6 原文。**

**规则（硬）**：凡断言 `cached_tokens > 0`、或者断言"前缀复用生效"，**夹具必须是 append-only 增长会话** ——
每轮 `messages` = **上一轮全部 messages + 上一轮回答 + 新内容**；不许每轮重写/重排提问，也不许单条 user 拼全文后反复问。

同机、同件、只改夹具形状（**外部实测 2026-10-20 / RTX 5080 16 GB**，表内数字原样引自该回执）：

| 夹具 | 题面 token | 第二轮 cached |
|---|---:|---:|
| 单条 user 拼全文 | 38,589 | **0%** |
| 多轮 messages，但每轮末尾都重写提问（前缀分叉） | 49,640 | **0%** |
| append-only（上轮全部 + 上轮回答 + 新内容） | 49,640 | **66.7%**（后续 83.3%） |

**判据（正控）**：同一引擎/同一池/同一题面下，append-only 夹具第二轮 `cached > 0`，且 `cached / prompt` 与上一轮输出规模相称；
环 + host 构型下同时要求 `adopt declined = 0`（引擎日志行）。

**怎么变红**：
1. 用非 append-only 夹具（每轮重写提问/单条拼全文）却断言"引擎不复用" ⇒ **假阴性，判据无效**；先换 append-only 夹具重跑，再下结论。
2. append-only 夹具第二轮仍 `cached = 0` ⇒ 真红：查 `reuse host-backed: on/off`、`NINFER_KV_REUSE_HOSTBACKED`、`NINFER_KV_WINDOW`、`--max-shared-prefixes 0` 是否带，以及 `adopt declined` 行。
3. 自称 append-only 但上一轮回答没有进入下一轮 `messages` ⇒ 口径不是 append-only，按第 1 条处理（这类夹具会把"题库形状"误判成引擎缺陷）。

## 8. 2026-10-20 新增：ring + `--auto-long-anchors` 超池 429 回归夹具

**脚本**：`verify/repro-ring-longanchor-429.ps1`（绝对路径参数集中在 `param()` 块，默认读本机 build-89 exe 与 `E:\Ternary-Bonsai-2-27B-ninfer-v3.ninfer`）。
**解决的 bug**：ring 模式（池 534 页 = 34,176 tok < 逻辑 262,144 tok）下开 `--auto-long-anchors --long-anchor-spacing 1024`，
3/4 段可能 200，但 **5 段与随后的 16.5K 短请求 429**（Device KV 池被 cached continuation 占满，host-restore 无空页）；
未修版本上这是"引擎进入不可用态、重启才恢复"的复现件。
**夹具**：每段约 16.5K token（ASCII filler，每段一条 user message），序列 3 → 4 → 5 段 → 同题重发 → **16.5K 短请求**；
判据 **R1'** 3/4/5 全 200 · **R2'** 无 500 / 无 `retained prefix exceeds its active KV entitlement` · **R3'** 5 段后短请求 200（不重启） · **R4'**（可选）不传 flag 的负控全 200。

**跑法（负控防假绿，必须按顺序）**：
1. 未修版本先跑：`powershell -NoProfile -ExecutionPolicy Bypass -File verify\repro-ring-longanchor-429.ps1 -Expect unfixed`
   ⇒ 必须打印 **`RED-EXPECTED: unfixed baseline reproduced`** 并 exit 0；若全 200 则打印 `FALSE-GREEN` 并 exit 2（**脚本本身被证伪**）。
2. 修复后验收：`… -Expect fixed [-RunNoAnchorNegControl]` ⇒ 全绿 exit 0，否则打印 `RED: fixed expectation not met. R1=… R2=… R3=… R4=…` 并 exit 1。
3. 脚本只起一个引擎（起前断言无残留 ninfer 进程 + 端口空闲），`finally` 停引擎；**不编译、不改源码**。

**两种 429 怎么区分（排障时必须先看报文）**：

| 报文 | 抛点 | 含义 |
|---|---|---|
| `Paged KV pool has no free page for host-restore: usable … allocated … this reservation …` | `src/core/paged_kv_cache.cpp:526-570`（`DeviceKVPagePool::materialize_one`），由 `src/models/qwen3_5/program/storage/kv_store.h:561-568` 的 `reserve_device_replica(..., "host-restore", demote_oldest_to_host)` 调起；`demote_oldest_to_host` 见 `kv_store.h:527-553` | **materialization 阶段的物理页不足**：reservation 已存在（逻辑承诺），实际要落一页时池里没有 free run，且最多 4 次 evict（`demote_oldest_to_host`）都没腾出页。与准入无关。 |
| `context cache exhausted: an idle Engine found no placement` | `src/runtime/engine/engine_core.h:1943` | **准入/放置阶段**：所有 active lane 都释放不出候选，Engine 在排队/放置层直接判 429。通常伴随 `cumulative_stats_.context_cache_exhausted_requests`，不是 `[diag] materialize-one FAIL`。 |

**怎么变红**：`-Expect unfixed` 却全 200 ⇒ 夹具不再能区分修与未修（假绿脚本，exit 2）；`-Expect fixed` 但 R1'/R3' 任一非 200 ⇒ 修复未达成（exit 1）；running 期间出现残留引擎 / 端口被占 ⇒ 脚本首断言直接抛错，不产生读数。

## 9. 2026-10-21 修复落地：C4 + R-2（实测读数与残留）

**状态**：已修，仓内（8 文件 +353/−29）。下表取自**同一次引擎启动**的夹具运行，日志目录 `E:\视频\_flashnext\experiments\20261020-回执修复\repro-c4h-fixed\`。

| 判据 | 读数 | 结论 |
|---|---|---|
| R1' 3/4/5 段 | **200 / 200 / 200**（20.6 / 34.9 / 49.2 s） | 通过 |
| R1' 同题重发 | 200（51.9 s） | 通过 |
| R2' 无 500 / 无 `retained prefix exceeds its active KV entitlement` | 日志 0 命中 | 通过 |
| R3' 5 段后 16.5K 短请求 | **200（9.4 s）**，不重启 | 通过 |
| R4' 不传 flag 负控 | 3/4/5/同题/短请求全 200（20.4/34.6/48.6/53.2/9.5 s） | 通过 |
| 总墙钟 | 锚点臂 166.0 s vs 无锚点臂 166.3 s | 通过（≤1.2×） |

**改动**：
1. `src/core/paged_kv_cache.{h,cpp}`：`materialize()` 增 `tag`/`evict` 形参，容量判定前按需驱逐；`materialize_one()` 把"固定 4 次上限"改成"按需到够（progress 断言 + 硬上限 64）"，并把**许可 0**（逻辑预留用完，`allow_logical_reservation` 下可 resize）与**物理 0**（free run 用完，要驱逐）分开；自证串 `[kv-evict]`。
2. `src/models/qwen3_5/program/storage/kv_store.h`：落页 sink 可注入；`host-restore` 走 `demote_other_addresses_to_host`（能**造**宿主副本的强 demoter），而不是只搬"本来就有宿主副本"页的 `demote_oldest_to_host`。
3. `.../storage/context.cpp`：demoter 不再整段跳过 `active` 地址（逐页守卫仍在）；`flush()` 里把 active 引用释放到 0 再 drop，并给静默 false 补 `[kv-drop-refused]` / `flush_refused`。
4. `.../planning/pressure.cpp`：**超池请求不采纳 continuation**（`skip-adopt-overpool`）；**当下没有足够空页也不采纳**（`skip-adopt-nofree`，条件 `restore_need > usable - allocated`）；采纳成功打 `[anchor-adopt]`。
5. `.../transactions/capture.cpp`：capture 的 `prompt_pages` 原来恒 0（门形同虚设），改用已确定的映射页数；超池请求的锚点直接 `skip-capture`。
6. `.../transactions/materialization.cpp`：restore 循环把强 demoter 传下去。

**口径（别当"关功能"读）**：LongAnchor 从"承诺"改成"**尽力而为的缓存**"——只有池当下装得下才采纳/保留；装不下就按 fresh root 正常服务（答案正确、慢一点），**不 500、不 429、不把引擎打死**。

**已知残留（带证据，别当已修）**：
1. **"采纳 234 页锚点"与"短请求 200"在本夹具里互斥**：5 段跑完 `free=8`，采纳 234 页锚点要 restore 234 页 ⇒ `skip-adopt-nofree` 必然弃采纳。无 nofree 门那版能打 `[anchor-adopt] frontier=14916 pages=234 reuse_pages=234`，但短请求 429。⇒ `[anchor-adopt]` 正控**无法在 R3' 同时成立**；要单独证它，须用"池宽裕时连续两次同前缀"的场景（未做）。
2. **位精确未验**：本夹具无 greedy 臂，没与基线 `3bd403e748a8bb6dc1b715db72021e46c2febfc3e395f1f2df65460dcf78c7a7` 逐字节比过。
3. **同页冷却无样本**：本次运行 `[kv-evict]` 178–180 页量级的成功降级未出现（`freed=0 exhausted` ×3），"同页反复降级"没被真实触发过。
4. `--auto-long-anchors` 在超池请求上**不再带来加速**（它本来就不该被采纳）——设计口径，非缺陷。

**怎么变红**：R1'/R3' 任一非 200 ⇒ 修复未达成；日志出现 `Paged KV pool has no free page for host-restore` ⇒ 回到物理页不足的老路；出现 `skip-adopt-overpool` 而 `prompt_pages=0` ⇒ capture 的 prompt 页数又没填上（门形同虚设）。

