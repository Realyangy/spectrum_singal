固定样本整体实验资料

separation/ 保存 300 个信号分离样本的汇总指标、逐样本结果及恢复后的 IQ。
recognition/ 保存 100 个频谱检测与调制识别样本的汇总指标和逐样本检测、匹配及分类结果。
logs/ 保存实验日志；scripts/ 保存实验和汇总脚本。

前端整体结果展示读取 frontend/live/experiment-data.json；单次在线任务记录位于各任务的 interactions/ 中。

逐样本结果中可能包含原实验环境的绝对路径，迁移后需按实际路径调整。scripts/run.sh 中的 Python 路径和输出位置也需要按运行环境检查；脚本会执行批量实验，不是前端启动脚本。
