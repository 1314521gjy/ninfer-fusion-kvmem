# NInfer KVMem 输出租约缺陷 —— 反馈包

反馈对象：`ninfer-serve-89.exe`（sm_89，sha256 `19222a2a68df6f7d87889ee3b33bedf2f3486ea309b6394a8f8a1a412262771e`）
平台：Windows 11 / RTX 4070 Ti SUPER 16GB (sm_89) / 驱动 596.36
模型制品：`Ternary-Bonsai-2-27B-ninfer-v3.ninfer`
报告日期：2026-10-08

---

## 包内文件与阅读顺序

| # | 文件 | 作用 | 建议 |
|---|---|---|---|
| 1 | `NInfer-KVMem-输出租约缺陷报告.md` | **主报告**：三条缺陷 + 最小复现 + 对照实验 + 修复验证 + 证据字段索引 | **先读这个** |
| 2 | `verify-output-window.py` | 单点探针：10 秒量出「池占用 / 空闲池 / 当前单次回答的输出上限」 | 想快速复现缺陷 1 时跑 |
| 3 | `ab-test-thorough-search.py` | 完整 A/B：确认旗标进 argv → 灌池至 ≥95% → 强制配额题面施压 → 回读 `materialization` | 要跑完整对照时用 |
| 4 | `附录-Bonsai-KVMem-输出墙-机制定稿.md` | 完整推导过程与**全部原始读数**（含 23 个历史实例的回看、495 条请求的统计） | 需要深挖时读，非必读 |

两个脚本**只用 Python 标准库**，无需安装依赖：

```bash
python verify-output-window.py 8091          # 单点探针
python -u ab-test-thorough-search.py 8091    # 完整 A/B（约 1~2 分钟）
```

---

## 30 秒版结论

1. **`--kv-lease-growth` 的输出租约在池被占满时会静默冻结在初始 4096 token**，单次回答被截断在约 4056，
   `finish_reason = output_limit`，`content` 为空串。对思考模型即表现为"模型不回答"。
2. **默认准入搜索预算只有 5 ms**（`search_granted_ns = 5 000 000`），且**在 `expansion` 阶段就被时间掐断**，
   随后判 `stop_reason = insufficient_expected_gain` 放弃 —— 容量本来有，只是没搜到。
   给到 250 ms（`--thorough-admission-search`）后，**同一实例、同一池占用，输出 4 056 → 16 384**。
3. **输出租约的扩容不参与"本请求自己常驻 KV"的降级**：真实长会话（题面 88k–105k）即使给足 250 ms，
   `selected_degradation_units` 仍为 0，输出恒卡在 ~4056。
   ⇒ 「池 < 上下文」这一档目前只对 **prefill/检索**方向成立，**对 decode/输出方向不成立**。
4. **失败不可诊断**：客户端无法区分"你的 max_tokens 到了"与"池装不下所以租约没扩成"，
   `materialization.stop_reason` 只落在服务端日志里。

---

## 修复验证（已实测，可直接复现）

| | 池 | 请求数 | `output_limit` 次数 | 最大输出 |
|---|---:|---:|---:|---:|
| 改前 | 65536 | 32 | **5** | 4 085 |
| 改后 | 131072 | 36 | **0** | **8 383** |

改后 36 条真实请求（题面 105k–118k、`enable_thinking=true`、`effort=medium`）**`finish_reason` 全为 `stop_token`**；
同一个此前失败过两次的上游会话在 03:11:08 执行完成（`status=complete, error_retained=False, duration=639.1s`）。

**⇒ 两个机制互补：`--thorough-admission-search` 解决"能扩却不扩"；池 = 上下文解决"额度本身为 0"。**

---

## 说明

- 包内**不含**早期草稿：那两份里有一句被后续实验推翻的过强结论（"输出租约从未扩展成功"），
  已在主报告与附录中明确更正，故不附带，避免版本混淆。
- 主报告的结论建立在上游 `--request-log-jsonl`（`schema_version: 27`）的原生字段上，
  未做任何插桩或反汇编；所有数字均为服务端自报。
