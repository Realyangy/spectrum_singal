# RF Challenge 必要源码

本目录保留项目使用的 [RF Challenge 0.2.0 源码](https://github.com/RFChallenge/icassp2024rfchallenge/tree/0.2.0)，并非完整上游仓库。

- `sampletest_testmixture_generator.py`：根据干扰帧和 Sionna 目标信号生成混合数据及真值。
- `rfcutils/`：目标信号生成、QPSK/OFDM 解调及其依赖的辅助函数。
- `src/torchwavenet.py`：WaveNet 网络结构。
- `src/config_torchwavenet.py` 和 `src/configs/wavenet.yml`：WaveNet 配置；原始权重中包含对 `src.config_torchwavenet` 的引用，名称不能随意更改。

上游 U-Net、训练示例和 notebook 未保留。目录名 `rfchallenge` 还被云侧量化代码引用，当前保留不改。
