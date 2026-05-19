# Objective:
我要测试在RL任务中使用supernet NAS搜索的可能性. 这里默认使用PPO, 测试gymnasium上带视觉环境.

这里supernet采用single-path weight slice设计, 类似OFA. 有有一个最大网络. 我希望supernet本身只包含backbone, 后面在接不同task head来实现actor & critic

分为三个阶段:
stage1: 采样数据
使用最大的supernet网络做backbone, 生成actor&critic(共享backbone)进行训练, 并记录训练中轨迹
再基于随机策略与环境进行采样, 得到另一批轨迹数据

stage2: 训练supernet, 进行表征学习
利用上面采样的数据, 在supernet上采样子网络, 进行表征学习. 其中supernet初始参数可以继承自上面训练
该过程数据监督学习, 对于每一批数据, 可以进行三明治法则知识蒸馏: 1个最大的网络(即supernet)作为teacher, 去蒸馏最小网络+ n个随机采样子网落

loss: 提供一个通用的API接口, 实现两种和state特性无关的表征学习loss
使用Dynamics Prediction:
Latent Dynamics Model (PBL, PlaNet style): 用 action-conditioned predictor `D_phi(z_t^m, u_t...u_{t+k-1}) -> z_hat_{t+k}`，目标为 `z_{t+k}^T = norm(g_T(f_T(s_{t+k})))`，loss 为 `L_dyn = Σ_k β_k * (2 - 2 * cos(z_hat_{t+k}, sg(z_{t+k}^T)))`

面向三明治法则知识蒸馏:
结合上面Latent Dynamics: 在latent层面做 cosine distance KD. 不使用MSE原因(小子网和大子网输出尺度可能不同。cosine 更关注方向/语义相似性，不强制数值尺度完全一致，更适合跨 subnet 蒸馏)

stage3:
subnet搜索策略这里就采用简单的EA算法, 将模型架构转为一个整数int list作为gene(因为subnet的搜索空间都是离散化的), 参考 @cnn_search_space.py

基于supernet的参数, 进行RL finetune来计算subnet在RL上的性能(fitness). 对于一个subnet, 从supernet上继承其架构上的参数, 然后初始化actor & ciritc head, 进行RL训练.

针对是否学习backbone部分参数, 提供一个输入参数--supernet_backbone-lr, 当<=0 时固定backbone参数, 当>0, 此处为设置一个独立的lr(一般来说默认值小于用于actor & critic head部分的lr)给backbone.


基于stable-baselines3和rl_zoo3的相关代码(已经安装到.venv中), 生成上述完整的代码并测试.

你需要先计划一个PLAN.md, 然后一步步执行.

---

# Done Condition:
每一轮检查是否真正完成goal的每一条(结合PLAN.md查看), 是否有遗漏和忽略的实现细节. 并进行smoke test, 或者运行完整的一次, 在验证到能成功运行后, ctrl+c终止.
验证的步骤不需要完整的跑完代码, 只需要保证每一步代码都是可以正常运行的.
