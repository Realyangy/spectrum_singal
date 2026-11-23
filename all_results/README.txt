RF-SepEdge 样本库实验结果

模型：原始 WaveNet、原始 YOLO11n 与宽带适配 XCiT 权重；未使用量化模型。
信号分离：300 条样本；频谱检测与调制识别：100 条样本。

separation/summary.json：总体、六组及不同 SINR 的 BER 与 MSE 汇总。
separation/per_sample.jsonl：逐样本误码、MSE 及实验来源。
separation/predictions/：300 条恢复后的 IQ 结果（NumPy .npy）。
spectrum/summary.json：检测与识别汇总指标。
spectrum/per_sample.jsonl：逐样本真实框、预测框、匹配与分类结果。
logs/：实验运行日志；scripts/：本次实验脚本。

逐样本文件和汇总中的绝对路径是服务器实验时的来源路径；复制到笔记本后不能直接作为笔记本路径使用。分离结果文件可按样本编号在 separation/predictions/ 中查找。压缩包不包含原始样本库和模型权重。
