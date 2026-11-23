# 混合信号分离

使用 WaveNet 恢复 QPSK 或 OFDM-QPSK 目标信号。当前样本库覆盖 CommSignal2、CommSignal5G1、EMISignal1 三类干扰，每组 50 条，共 300 条；不在页面中提供 CommSignal3。每条保存混合 IQ、对齐真值比特与目标 IQ、WaveNet 输入和哈希。

## 文件

- `src/prepare_sample_library.py`：从已有 Seed 42 实验混合数据与真值导出 300 条固定样本，拒绝覆盖已有样本库。
- `src/mit_sionna_official_smoke.py`：读取原始干扰帧并使用 Sionna 生成混合样本及真值。
- `src/mit_sionna_wavenet_infer.py`：读取指定实验数据与原始 WaveNet 权重，批量推理。
- `src/mit_sionna_official_ber.py`：使用 Sionna 解调并逐样本计算 BER 和波形误差。
- `src/summarize_sinr.py`：汇总 BER 及 MSE 的分档结果。
- `rfchallenge/`：MIT 原始信号处理及 WaveNet 模型代码；`dataset/` 是原始干扰数据，`pretrained-models/` 为原模型。
- `sample-library/`：演示样本、真值、WaveNet 输入和索引；`interactions/`：每次真实交互的输入和结果。
- `output/`：保留三个有效干扰的源实验数据和历史汇总；可重新计算的中间推理数据不作为当前运行依赖。

## 运行

日常演示在项目根目录按 [README.md](../README.md) 启动 `start.sh`，或在 8090 同服务器云端页面选择样本；无需手动运行以下离线脚本。

重建原始样本的示例（不会覆盖非空输出目录）：

```bash
cd /home/flyvideo/Sata2/cloud-edge/RF-SepEdge/signal-separation
conda activate rf-sionna
CUDA_VISIBLE_DEVICES=-1 python src/mit_sionna_official_smoke.py \
  --soi-types QPSK,OFDMQPSK --interference CommSignal2 \
  --samples-per-sinr 10 --sinrs=-24,-18,-15,-12,-9,-6,0 \
  --seed 42 --out output/example_seed42
conda activate rf-spectrum
CUDA_VISIBLE_DEVICES=0 python src/mit_sionna_wavenet_infer.py \
  --interference CommSignal2 --source output/example_seed42 \
  --out output/example_seed42_infer --batch-size 1
conda activate rf-sionna
CUDA_VISIBLE_DEVICES=-1 python src/mit_sionna_official_ber.py \
  --interference CommSignal2 --source output/example_seed42 \
  --predictions output/example_seed42_infer --out output/example_seed42_ber
python src/summarize_sinr.py --generated output/example_seed42 \
  --evaluated output/example_seed42_ber --out output/example_seed42_report
```

`src/prepare_sample_library.py` 使用 `output/mit_sionna_seed42`、`output/commsignal5g1_seed42`、`output/emisignal1_seed42` 的可信实验文件生成固定样本库。已有样本库时脚本会拒绝覆盖。重新构造对应模型输入需在项目根目录运行 `python backend/prepare_library_inputs.py`，也不会覆盖现有输入。

## 历史实验结果

Seed 42、每组 70 条、7 档 SINR 的总体 WaveNet BER：

| 目标 | CommSignal2 | CommSignal5G1 | EMISignal1 |
|---|---:|---:|---:|
| QPSK | 1.724% | 1.224% | 0.033% |
| OFDM-QPSK | 9.926% | 3.156% | 1.568% |

这些是多档 SINR 的历史实验统计，不是当前量化模型处理单一样本的结果。0 误码也不意味着理论 BER 恒为 0。
