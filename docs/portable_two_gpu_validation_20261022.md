# 两卡可迁移消融入口：本轮验证记录

本轮仅修改/打包代码并进行新增CPU测试，不启动真实模型、数据集、GPU实验或远程实验链。

## 已完成

- 从已归档的官方v4副本导出239份源码；原归档、暂存Git blob逐文件bytes/SHA256核对一致。
  baseline119文件保持原bytes，storage120文件保持已归档v4 bytes。
- 五个消融的替换文件均来自先前登记的源码副本，核对原始与替换SHA。默认使用其中三项。
- 新增固定配置、下载登记/续传、共享缓存、内容去重checkpoint、逐次结果保留、显存/PSS/NVML
  观察、完整state/Hadamard摘要、串行停发及Linux进程组清理入口。
- 两个Qwen固定Hub revision已通过只读metadata查询确认存在；未下载模型权重。
- Git暂存树单独导出到隔离目录后执行新增测试：**38 passed，83.00秒**。
  隔离目录没有依赖工作区未提交的旧tools/helper脚本。
- 新代码及配置语法检查、Git diff whitespace检查通过；新代码/文档U+FFFD为0。

CPU用例包括：中断续传状态与残块保留、已完成文件不再下载、不完整本地目录回退下载、
新显式路径生效、错误官方SHA拒绝、safetensors/parquet格式检查、锁冲突与异常释放、
不可覆盖的JSON收据、源码篡改拒绝、全部Config字段一致、固定batch/精度、两卡缓存隔离、
八臂矩阵、BF16内容摘要、535-state/28-Hadamard核对、CPU autograd不变、外层峰值及失败端点、
非零/超时/缺层/质量/NVML缺失停发、旧计划禁止重试、未封存cache拒绝覆盖、GPU UUID校验、
依赖导入失败拒绝，以及import/help无隐式torch加载或下载。

## 尚未证明的范围

- 本机测试环境是Windows、torch2.6.0+cpu，**不是**目标Linux/CUDA12.8/torch2.9.1环境。
- 额外尝试本机完整原生模块导入时，发现既有torch/torchvision不兼容
  （`operator torchvision::nms does not exist`），该检查未通过；未修改全局依赖、
  未屏蔽错误、未因此宣称原生执行环境已接受。
- 已加入目标机实际CPU导入准备门：隐藏CUDA，在下载模型前导入完整选定原生模块，
  依赖错误先失败，不能直接进入GPU执行。
- 无真实两卡CUDA、FSDP、Triton、NCCL、完整模型/完整QA或OOM容量实测。
  新入口不继承历史四卡33门/独立65请求/私有mapped QA审查接受状态。
- 两卡局部统计与collective分区可能不同；新消融仅与同机同两卡full-v4对照比较，
  不宣称与历史四卡bit-exact或混用时间数据。

使用与边界见根目录 `README_PORTABLE.md`。完整测试通过表示本轮新增CPU范围通过，
不表示可以省略目标服务器的依赖、资源、数值与质量检查。
