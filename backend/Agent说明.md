# 频谱项目内的 Agent

`edge_agent.py` 是频谱项目携带的云边通信 Agent。它向云侧登记并发送心跳、接收下发的模型、校验文件和反馈接收状态；频谱后台 `server.py` 负责发起任务、加载模型和回传结果。

本机目前还可能运行项目外的公共 Agent `/home/lenovo/cloud-edge/cloud-edge-agent/edge_agent.py`，该进程可能服务于行人等其他业务。项目内 Agent 与公共 Agent 是两个独立文件，修改项目内文件不会更新公共 Agent。

使用项目内 Agent 时，可参照项目根目录 README 的启动命令。Agent 的 `--node-id` 与后台 `RF_NODE_ID` 必须一致，Agent 的 `--model-dir` 与后台 `RF_MODEL_ROOT` 必须一致。不要同时运行两个具有相同终端编号的 Agent；如果公共 Agent 已使用该编号，先核实其业务用途，或为频谱配置独立编号和模型目录。

临时模型下载目录在模型目录内的 `.agent-staging`，便于在模型目录和 Agent 状态目录属于不同 Docker 数据卷时安全保存模型。移动此文件后，已构建的 Docker 镜像不会自动更新，重新打包时应使用新路径。
