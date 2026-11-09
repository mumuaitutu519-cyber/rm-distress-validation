# sun21 不一致与未覆盖报告

> 公开版删节说明（342）：按无逐公司/逐调用预测值的公开规则，删去引言单个案例的三条概率核对行和对应说明项；其余汇总差异、数值、限制及原项目编号均原样保留。完整报告可申请授权审阅。
> 逐格 CSV 因原表 793 行超过 500 行上限，按表号分卷见 [VERIFICATION_sun21/](VERIFICATION_sun21/)。六条单案例分数/概率记录未公开。表体 561/568 一致；下方原始 589/608 口径另含表注。

表体和表注：589/608 一致；7 不一致，12 未覆盖。星号 4/4。正文：19/27 一致。

以下为有底层计算支持的条件性稿件替换值；本任务未修改稿件或统计结果。

| 表与位置 | 稿件 | 直接舍入应显示 | 原因与证据 |
|---|---|---|---|
| Table 2 / object 2, row 2, cell 7, token 1 | 0.082 | 0.081 | Formatting/double-rounding difference: raw grouped-Platt log loss is approximately 0.081498; the intermediate table uses 0.0815, which becomes 0.082 at three decimals. Direct rounding is 0.081. Original-vintage result agrees with the recomputation.；output/analysis/strict_timing_probability_metrics.csv score=qwen_canonical_naive |
| Table A6 / object 12, row 2, cell 2, token 1 | 2.62 | 2.63 | Exact halfway case: the manuscript uses Python/native round-to-even formatting, while the requested decimal half-up verification rounds away from zero. Same observations and underlying value. Replacement applies only if half-up is the chosen manuscript convention; this is not a substantive statistical error.；output/analysis/strict_timing_extended_paired_summary.csv strict_adverse_after |
| Table A6 / object 12, row 2, cell 3, token 1 | 3.12 | 3.13 | Exact halfway case: the manuscript uses Python/native round-to-even formatting, while the requested decimal half-up verification rounds away from zero. Same observations and underlying value. Replacement applies only if half-up is the chosen manuscript convention; this is not a substantive statistical error.；output/analysis/strict_timing_extended_paired_summary.csv strict_adverse_after |
| Table A6 / object 12, row 24, cell 3, token 1 | 8.62 | 8.63 | Exact halfway case: the manuscript uses Python/native round-to-even formatting, while the requested decimal half-up verification rounds away from zero. Same observations and underlying value. Replacement applies only if half-up is the chosen manuscript convention; this is not a substantive statistical error.；output/analysis/strict_timing_extended_paired_summary.csv strict_random_order_04 |
| Table A6 / object 12, row 28, cell 4, token 1 | 56.2 | 56.3 | Exact halfway case: the manuscript uses Python/native round-to-even formatting, while the requested decimal half-up verification rounds away from zero. Same observations and underlying value. Replacement applies only if half-up is the chosen manuscript convention; this is not a substantive statistical error.；output/analysis/strict_timing_extended_paired_summary.csv strict_sentences |
| Table A6 / object 12, row 29, cell 4, token 1 | 31.2 | 31.3 | Exact halfway case: the manuscript uses Python/native round-to-even formatting, while the requested decimal half-up verification rounds away from zero. Same observations and underlying value. Replacement applies only if half-up is the chosen manuscript convention; this is not a substantive statistical error.；output/analysis/strict_timing_extended_paired_summary.csv strict_signed_language |
| Table A8 / object 14, row 4, cell 5, token 1 | 56.2 | 56.3 | Exact halfway case: the manuscript uses Python/native round-to-even formatting, while the requested decimal half-up verification rounds away from zero. Same observations and underlying value. Replacement applies only if half-up is the chosen manuscript convention; this is not a substantive statistical error.；output/analysis/strict_timing_information_margins.csv adverse/system |
| Body / paragraph 329, prompt_field_pairs | 17,548 | 10536 | There are 1,756 cases × 6 percentage/decimal ratio fields = 10,536 pairs, of which 12 differ. Across all ten fields there are 17,560 pairs and 17,548 equal pairs. Thus 17,548 is an equality count, not the denominator. A valid wording is: 12 of 10,536 percentage/decimal ratio-field pairs differ by 0.01 percentage points; or 12 of 17,560 total field pairs.；raw_rebuild/strict_timing_{primary,probability}_prompts*.parquet |

## 未覆盖项

| 位置 | 稿件值 | 独立诊断值 | 原因 |
|---|---|---|---|
| Table 4 / paragraph 114, table note, token 8 | 1.6 | 1.5 | Independent diagnostic: original draw stream/seed absent; cannot assert exact agreement or a manuscript correction |
| Table 4 / paragraph 114, table note, token 9 | 14.6 | 14.9 | Independent diagnostic: original draw stream/seed absent; cannot assert exact agreement or a manuscript correction |
| Table 4 / paragraph 114, table note, token 10 | -0.6 | -0.3 | Independent diagnostic: original draw stream/seed absent; cannot assert exact agreement or a manuscript correction |
| Table 4 / paragraph 114, table note, token 11 | 9.6 | 9.4 | Independent diagnostic: original draw stream/seed absent; cannot assert exact agreement or a manuscript correction |
| Table 4 / paragraph 114, table note, token 12 | -13.5 | -13.9 | Independent diagnostic: original draw stream/seed absent; cannot assert exact agreement or a manuscript correction |
| Table 4 / paragraph 114, table note, token 13 | 3.4 | 2.1 | Independent diagnostic: original draw stream/seed absent; cannot assert exact agreement or a manuscript correction |
| Table 4 / paragraph 114, table note, token 14 | -8.5 | -8.7 | Independent diagnostic: original draw stream/seed absent; cannot assert exact agreement or a manuscript correction |
| Table 4 / paragraph 114, table note, token 15 | 5.0 | 5.5 | Independent diagnostic: original draw stream/seed absent; cannot assert exact agreement or a manuscript correction |
| Table 4 / paragraph 114, table note, token 16 | -3.8 | -4.4 | Independent diagnostic: original draw stream/seed absent; cannot assert exact agreement or a manuscript correction |
| Table 4 / paragraph 114, table note, token 17 | 9.0 | 8.9 | Independent diagnostic: original draw stream/seed absent; cannot assert exact agreement or a manuscript correction |
| Table 4 / paragraph 114, table note, token 18 | -1.3 | -1.7 | Independent diagnostic: original draw stream/seed absent; cannot assert exact agreement or a manuscript correction |
| Table 4 / paragraph 114, table note, token 19 | 9.0 | 9.3 | Independent diagnostic: original draw stream/seed absent; cannot assert exact agreement or a manuscript correction |
| Body / paragraph 402, random_tie_jaccard_5 | 0.283 | 0.282 | Original random tie-break seed/draws unavailable; independent seeded diagnostic only |
| Body / paragraph 402, random_tie_jaccard_10 | 0.457 | 0.455 | Original random tie-break seed/draws unavailable; independent seeded diagnostic only |
| Body / paragraph 137, independent rerun count | 63 |  | Neither the 63 independently rerun raw replies nor a run-ID comparison ledger is in the 11.06 archive or supplied project. Archived 192-call reload is a different experiment. No inference rerun permitted. |
| Body / paragraph 137, raw-string matching reruns | 58 |  | Neither the 63 independently rerun raw replies nor a run-ID comparison ledger is in the 11.06 archive or supplied project. Archived 192-call reload is a different experiment. No inference rerun permitted. |

## 最不利事实

1. 原始生成调用未记录最终 token 序列和每次调用时间；当前包不能验证 sun21 新增的 63 次独立重跑、58 次原文一致声明。已存档的 192 次 reload 不能替代该证据。
2. 表 4 新增容量差区间、正文随机 tie-break 均缺原始随机种子和抽样流。已用包内逐公司数据补做独立 1,000 次计算，但保守记为未覆盖，不冒充原计算。
4. quick 模式从原始冻结 1,000 次自助抽样重新计算已存档区间；不重新执行全部 calibration bootstrap，也不重新选择基准模型超参数。
5. 私有库保留源自授权数据的逐公司分析数据及提示词；改公开前须按 DATA_REVIEW.json 移除，并另审查聚合结果披露许可。
6. sun21 五幅嵌入图与已核查 sun15 原图逐字节相同；424 个底层数值在既定容差内一致。见 FIGURE_VERIFICATION_sun21.csv。
