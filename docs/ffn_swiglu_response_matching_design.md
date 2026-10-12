# SwiGLU 响应匹配：损失设计草案

状态：仅设计，未实现到量化循环，未启动复现或 GPU 实验。用户当前要求优先改进方法设计。成功指标为 teacher→quantized KL；PPL 为辅助指标。

## 先明确现有损失的边界

现有 REAL-Q full-Fisher 损失使用实际 Transformer 层输出差，已经穿过当前 FFN 的 SwiGLU。其反向传播也已经包含当前 gate/up 乘法的真实链式导数。普通 MSE 或 Fisher 谱修正均不能描述成“补上遗漏的 SwiGLU”。当前输出的交互误差不应再次被当作原方法没有计算的项。

论文附录 E(1)提出非线性代理/高阶 Taylor 是研究方向，不等于证明本仓库原 loss 漏掉当前块 SwiGLU 前向。这里提出一个补充约束的假设：在有限校准点匹配输出时，量化后的 FFN 对输入漂移的响应可能仍不同于原 FFN。

## 定义

令 M_theta(z)=W_d[SiLU(W_g z) * (W_u z)]，z 是进入 FFN 线性模块的归一化输入。M_0 为保留的原始 FP FFN，M_theta 为量化过程中的当前混合精度 FFN。

对同一校准样本，z_0 是原始 FP 路径的 FFN 输入，z_q 是当前路径的 FFN 输入；r=z_q-z_0。二者及 teacher 目标均 detach。只选择现有样本，新增的是损失计算使用的 FFN 输入端点，不增加或选择校准序列。

响应误差：

R_theta = [M_theta(z_q)-M_theta(z_0)] - [M_0(z_q)-M_0(z_0)]。

候选辅助项：L_response = (mu/2) E ||R_theta||²，mu=tr(F)/H。

首个完整候选目标：L_new = L_F + L_response，保留原 Fisher 项系数。mu 仅用于沿用当前输出损失的尺度约定，并不保证两项梯度等大；需记录梯度范数和截断比例，并设置整体尺度控制。暂不为结果寻找更优混合系数。

## 为什么与 SwiGLU 有关，又有什么没有解决

设 g=W_g z、u=W_u z，则精确方向导数为：

J_theta(z) r = W_d[(SiLU'(g) * u) * (W_g r) + SiLU(g) * (W_u r)]。

响应误差满足：

R_theta = integral_0^1 [J_theta(z_0+t r)-J_0(z_0+t r)] r dt。

因此它依赖沿真实输入漂移路径变化的 SiLU 门控、up 分支幅度、以及 down 混合。它不等于用一个固定矩阵对当前输出差重新加权，也不需要建立或储存完整 Jacobian。

响应匹配本身是通用思想，也可以用于线性网络，不能仅凭名字声称收益来自 SwiGLU 的高阶非线性。必须比较线性化响应 R_linear=[J_theta(z_0)-J_0(z_0)]r。R_theta-R_linear 才是两模型非线性余项的差。如果精确响应没有优于线性化响应，不能把结果归因于更好的高阶 SwiGLU 建模。

本目标并未直接估计真实下游 KL 的三阶/四阶导数，也没有解决所有后续层非线性。它是希望通过保持局部响应改善最终 KL 的工程假设。

## 实现范围与成本边界

- 保持 REAL-Q 逐模块、逐块更新范围：梯度仅更新当前模块尚未固定的列；其他模块不额外补偿，不迁移 Adam 状态。
- 在任何当前层量化发生之前保存 FP FFN 权重或必要的 teacher 响应。不能把已经量化部分权重的 FFN 当 teacher。
- FFN 优化期间上游权重固定，z_0/z_q 及 teacher 在两端点的响应可以按校准样本分批缓存。不可假定 Transformer 层输出缓存直接等于 FFN 输出缓存；残差路径需要明确分离。
- M_theta(z_q) 可复用当前前向中的 FFN 输出。新增 M_theta(z_0) 通常需要一次 FFN 前向及该分支的反向；teacher 分支只需无梯度计算并缓存。
- 不需要下游整网反传或完整 Hessian/Jacobian；但 FFN 是主要计算开销之一，不能宣称增加的成本必然很小。8 GB 下使用分批/顺序处理，测量额外图的显存占用和耗时后才能评价工程可行性。

## 设计风险与后续验证条件（本轮不运行）

1. z_q=z_0 时辅助项严格为零；同一 teacher/student FFN 时辅助项严格为零。FP 端点不是独立样本，也不应使用评测文本参与构建。
2. 教师响应差应使用同一 M_0 在两个端点计算，不可把两条不同权重/不同输入路径的输出差误写成 teacher 的输入响应。
3. 逐模块有限差分检查辅助梯度，核对冻结列无更新，teacher 与上游输入无梯度。
4. 与普通 hybrid、只有整体损失缩放的原 Fisher、线性化响应匹配比较，判断机制来自额外计算、尺度、还是非线性响应信息。最终以固定协议下独立评估的 KL 为准。
5. 强制保持 teacher 响应可能妨碍补偿上游量化误差，这是实质性风险。r 很小也可能令辅助项近零；不能未经证据就归一化放大或扫系数。
6. 即使有 KL 改善，还要确认相同 GPU 时间预算的原 REAL-Q 是否也能达到；正向梯度诊断或简单跑通均不是论文贡献。

## 工程参考

- Sobolev Training for Neural Networks，NeurIPS 2017：在函数值以外匹配导数的训练思路。https://papers.neurips.cc/paper_files/paper/2017/hash/758a06618c69880a6cee5314ee42d52f-Abstract.html
- Knowledge Transfer with Jacobian Matching，ICML 2018：Jacobian 匹配及其与带输入扰动的蒸馏的联系。https://proceedings.mlr.press/v80/srinivas18a.html

上述论文支持响应/导数匹配这一通用思路，没有验证本设计在 REAL-Q 或 SwiGLU W4 量化上的收益。这里的两端点、缓存与逐块集成是拟研究的工程适配，不声称新数学算法。
