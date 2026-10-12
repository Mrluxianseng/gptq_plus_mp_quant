# FFN Fisher + MSE：进一步优化的具体假设

当前定位修正：谱补偿是通用误差度量修正，不利用 SwiGLU 的具体非线性，不能将它作为 SwiGLU 非线性建模贡献。用户要求暂不复现、优先优化非线性方法设计；新的响应匹配草案见 `ffn_swiglu_response_matching_design.md`，目前未运行实验。

状态：设计阶段，尚未验证 KL/PPL 收益。不是已有实验的正向结论。

评价判据修正：以最终 teacher→quantized KL 的可重复下降为主要成功条件；PPL 与任务准确率作为辅助报告，不要求与 KL 同时改善。原 fixed hybrid 在双方均使用 k=4 的一个匹配对照中已经有 KL −1.3947%、PPL +1.0484% 的结果，应优先复验该正向信号。谱补偿仍是待测候选，不能因为该对照的 PPL 上升就跳过复现而转向新方法。

## 实现核对与解释修正

`gptq_utils/gptq_plus_utils.py` 的 `compute_refresh_loss` 接收同一份真实 Transformer 层输出误差 `delta = out_hidden - fp_hidden`。Fisher 项和 FFN hybrid 的 MSE 项都已经经过实际 SwiGLU 前向。`fisher_diag_mse` 虽然名字含 diag，这条路径实际计算的是共享完整 H×H Fisher 二次型；只有 `legacy_fisher_diag_mse` 是另一种逐 token 对角形式。

因此原 hybrid 增加的是各向同性输出误差约束，不是补上原损失没有计算的 SwiGLU 非线性。用低秩 Fisher 替换现有完整 Fisher 也不能直接声称增加了原方法遗漏的通道相关信息。

当前配对日志确认 a_loss_ratio=1，loss_slide_window=False。另有潜在兼容性问题：hybrid 未被 is_fisher_backed_loss 判定包含，因而不会走其误差截断和部分滑窗分支。该问题不影响上述两项设置下的配对，但后续实现必须审查这些分支，不能把路由差异当成损失收益。

## 原 hybrid 的谱解释

省略 token 平均，令 mu=tr(F)/H：

L_F = 1/2 delta^T F delta

L_M = mu/2 ||delta||^2

L_hybrid = (L_F + L_M)/2 = 1/2 delta^T [(F+mu I)/2] delta。

若 F=U diag(s_i) U^T，则原 hybrid 把各方向的曲率 s_i 替换成 (s_i+mu)/2。这会提高弱方向的约束，也会降低强方向的曲率。它可能改变原 Fisher 对误差方向的优先级；这是一种可检验解释，尚未被证明是 KL 退化的原因。

## 候选：只补充低曲率方向的 MSE 约束

F_star = U diag(max(s_i,tau)) U^T

L_star = L_F + 1/2 sum_i (tau-s_i)_+ (u_i^T delta)^2。

该候选保留原 Fisher 的特征方向以及 s_i >= tau 的曲率，只对低曲率方向增加补偿。首轮固定 tau=mu，复用原 hybrid 的尺度约定，避免以参数搜索代替方法验证；这个选择本身仍是需要检验的设计假设。

这是一种经典谱下限修正的工程应用，不声称原创数学算法。参考 Higham 对满足最小特征值下限的最近对称矩阵的说明：
https://nhigham.com/2021/01/26/what-is-the-nearest-positive-semidefinite-matrix/

## 验证和停止条件

1. 数学与实现审查：先对称化 F；检查 PSD 数值误差、损失和梯度有限差分，以及强特征方向的曲率保持。F=mu I 时候选应退化为原 Fisher。不能把特征向量逐列符号变化当成结果差异。
2. 同配置对照：原 Fisher、旧 hybrid、谱补偿候选；所有样本、种子、优化器、步数与评估窗口完全相同。增加整体缩放的 Fisher 对照，以排除新损失总尺度、梯度截断和 Adam 数值项的影响。
3. 多层小配置：检查前中后层的真实 KL 方向导数和实际更新前后 KL。单层梯度余弦只能帮助诊断，不能取代最终量化 KL/PPL。
4. 通过后才做全模型配对；不允许根据评估结果选择 tau。若继续出现可重复的 KL 退化，就停止此候选。

## 成本与局限

每层一次对称特征分解和矩阵构造，随后缓存 F_star，沿用原来的二次型和反向图；不增加每个量化块的 Transformer 前向或反向次数。完整特征分解成本为 O(H^3)，临时矩阵存储为 O(H^2)，需要实测时间及峰值显存，不能事先保证大模型上开销很小。

低曲率可能代表任务本来就不敏感，而不是估计遗漏；谱补偿也可能降低性能。保留强 Fisher 方向不等于保证真实 KL 不升。该假设值得单独验证，但目前没有 work 的结果。
