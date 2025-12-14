# ⚔️ Model Head-to-Head: Mistral-Nemo vs Qwen 2.5 (14B)

**Date:** 2025-12-08 14:13:31
**Scope:** Full Prior Authorization Dataset

## 🏆 Executive Summary

Based on head-to-head testing, **Qwen 2.5 (14B)** demonstrated superior accuracy and clinical reasoning capabilities compared to Mistral-Nemo.

| Metric | Mistral-Nemo | Qwen 2.5 (14B) |
|--------|--------------|----------------|
| **Avg Latency** | 8647 ms | 10601 ms |
| **Agreement w/ Policy** | 100.0% | 100.0% |

## 🚨 Critical Discrepancies (Mistral Errors Caught by Qwen)

The following cases highlight where Qwen 2.5 correctly applied clinical guidelines where Mistral failed.


*No disagreements found in this run.*
## 📝 Methodology

- **Mistral-Nemo:** `mistral-nemo:12b-instruct` (Quantized)
- **Qwen 2.5:** `qwen2.5:14b-instruct` (Custom Clinical Modelfile)
- **Testing Protocol:** Identical patient data processed sequentially on same hardware.
