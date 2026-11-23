# 信号分离与恢复

本任务使用 WaveNet 从混合复数 IQ 中恢复 QPSK 或 OFDMQPSK 目标信号。当前固定样本覆盖 CommSignal2、CommSignal5G1 和 EMISignal1 三种干扰，每种组合 50 个样本，共 300 个。

```text
spectrum-separation/
├── sample-library/        # 固定样本、真值和清单
├── pretrained-models/     # 六组本地原始 WaveNet 权重
├── interactions/          # 单次任务输入副本、结果和日志
└── src/
    ├── wavenet_model/     # WaveNet 模型、配置和可信旧权重加载
    ├── rfcutils/          # 信号生成、调制和解调工具
    └── *.py               # 样本准备、独立推理和实验评估脚本
```

每个固定样本包含混合信号 `input-iq.npy`、真值 `truth.npz` 和 WaveNet 输入。端侧使用云侧本次下发的模型处理 40,960 个复数 IQ 点，得到 `recovered-iq.npy`；对带参考目标 IQ 和真实比特的样本计算 MSE、输入 BER 和输出 BER。没有对应真值的新数据不能按当前方法计算真实 MSE 或 BER。

运行方式见项目根目录 README。单次任务结果保存在 `interactions/<任务ID>/`；`pretrained-models/` 的原始权重供独立实验使用，与当前云侧下发的模型包是不同文件。旧权重包含 Python 对象，独立脚本通过 `wavenet_model/checkpoint.py` 兼容加载；只应加载可信来源的权重。
