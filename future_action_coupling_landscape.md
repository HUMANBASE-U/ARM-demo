# Future Prediction 与 Action Prediction 耦合机制占坑评估

生成日期：2026-05-27  
目标读者：AI / ML 本科生，准备 Robotics / Embodied AI / World Model / VLA 方向研究选题

## Executive Summary

- **总判断：Yellow。** 这个方向已经被明显占坑，但还没有被完全讲透。不能再把“joint future video + action prediction 会不会提升 robot policy”当成新问题；但可以做“不同耦合机制为什么有效、什么时候有效、失败模式是什么”的小型机制研究。
- **最危险的占坑论文是 DreamZero、GR-1、UWM、Motus、LingBot-VA、FLARE、WoG、GigaWorld-Policy、PFD、Being-H0.7。** 它们已经覆盖了 joint video-action、shared backbone separate heads、latent future alignment、训练时 future / 推理时丢 future、video-first + IDM、future-conditioned correction 等关键路线。
- **joint video-action prediction 本身已经不是 Green。** GR-1 早在 2023/2024 就同时预测 action 和 future image；UWM、Motus、LingBot-VA、DreamZero 在 2025–2026 年把它扩展到大规模 WAM。
- **“future branch 到底有什么用”正在被研究，但还没有完全稳定。** FLARE 把它解释为 future latent representation alignment；WoG 把 future observation 压成 action condition；PFD 明确提出 future branch 不是单纯 regularizer，而是可以被蒸馏的 future-conditioned action correction。
- **video-first + inverse dynamics 也已经是成熟路线的一支。** UniPi、VPP、Say-Dream-Act、EVA、GEM-4D、AMPLIFY 都在不同形式上采用“先生成 / 表征未来，再用 IDM 或 action decoder 转动作”。
- **真正空缺不是“有没有 future”，而是“耦合程度的系统比较”。** 目前许多论文各自证明自己的架构有效，但很少在同一小 benchmark、同一数据预算、同一模型容量下比较 action-only、joint、video-first+IDM、privileged future、uncertainty-aware coupling。
- **对本科生最适合的切口是 controlled mechanism experiment。** 不建议训练 14B WAM 或大规模 VLA；建议在 2D manipulation / MetaWorld / ManiSkill / LIBERO 子任务上做小模型、统一数据、统一评估。
- **你提出的题目 “How Much Should a Robot Policy See Its Predicted Future?” 值得做，但必须缩小。** 最小新意应放在：未来预测质量、动作失败、OOD robustness、uncertainty calibration 之间的因果关系，而不是提出一个新的大模型。

结论颜色：

```text
Red:    joint video-action prediction as a broad idea
Yellow: coupling mechanism / ablation / uncertainty-aware use
Green:  undergraduate-scale controlled benchmark with clear diagnostics
```

## Research Question Definition

原始 vague idea：

> 在机器人 World Action Model / VLA / robot policy 中，future video prediction 和 action prediction 应该如何耦合？

可以压缩成以下 research questions：

**RQ1. Future prediction 到底通过什么机制提升 action prediction？**  
候选机制包括 representation regularization、dynamics supervision、implicit planning、future-conditioned correction、executability checking。

**RQ2. 在相同数据和模型容量下，哪种耦合方式更好？**  
比较 action-only、joint future-latent + action、video/latent-first + inverse dynamics、training-time privileged future、uncertainty-aware joint model。

**RQ3. Future branch 是否必须在 inference 时存在？**  
如果训练时预测未来、推理时丢掉 future branch 仍然有效，那它更像 representation shaping；如果显式 future condition 才有效，那它更像 planning / correction。

**RQ4. Future prediction error 是否能预测 action failure？**  
如果 future error 与失败高度相关，那么 future branch 可作为 risk / uncertainty signal；如果不相关，说明它可能只是在做无关视觉重建。

**RQ5. 耦合程度如何影响 OOD robustness 与 executability？**  
尤其在扰动 cube 位置、遮挡、背景变化、动力学变化、目标组合变化时，future-aware policy 是否更稳。

## Literature Search Map

### A. World Action Models / Joint Video-Action Models

- **GR-1: Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation**：输入语言、历史图像、机器人状态，同时预测 action 和 future images，是早期强相关 joint future-action policy。[arXiv](https://arxiv.org/abs/2312.13139)
- **Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets**：一个 multimodal diffusion transformer，通过 modality-specific diffusion timesteps 表示 policy、forward dynamics、inverse dynamics、video generation。[arXiv](https://arxiv.org/abs/2504.02792), [Project](https://weirdlabuw.github.io/uwm/)
- **Motus: A Unified Latent Action World Model**：MoT + UniDiffuser-style scheduler，同时支持 world model、VLA、IDM、video generation、video-action joint prediction。[arXiv](https://arxiv.org/abs/2512.13030), [Project](https://motus-robotics.github.io/motus)
- **LingBot-VA / Causal World Modeling for Robot Control**：共享 latent space，把 vision/action tokens 结合，学习 frame prediction 和 policy execution。[arXiv](https://arxiv.org/abs/2601.21998)
- **DreamZero: World Action Models are Zero-shot Policies**：14B WAM，基于 pretrained video diffusion backbone，jointly models video and action，强调 zero-shot physical generalization。[arXiv](https://arxiv.org/abs/2602.15922), [Project](https://dreamzero0.github.io/)
- **GigaWorld-Policy**：action-centered WAM，训练时动作预测 + 视频生成双监督，因果设计防止 future video token 影响 action token，推理可丢视频分支。[arXiv](https://arxiv.org/abs/2603.17240), [Project](https://gigaai-research.github.io/GigaWorld-Policy/)
- **MotuBrain**：Motus 后续大规模 WAM，支持 policy learning、world modeling、video generation、IDM、joint video-action prediction，并强调实时部署。[arXiv](https://arxiv.org/abs/2604.27792)

### B. Video World Model + Inverse Dynamics / Modular Approaches

- **UniPi**：把 policy 表示成 text-conditioned video generation，然后用 inverse dynamics model 抽取 low-level actions。[arXiv](https://arxiv.org/abs/2302.00111), [Google Research Blog](https://research.google/blog/unipi-learning-universal-policies-via-text-guided-video-generation/)
- **Video Prediction Policy (VPP)**：利用 video diffusion model 内部 predicted future representations，学习 implicit inverse dynamics model。[PMLR / ICML 2025](https://proceedings.mlr.press/v267/hu25g.html), [arXiv](https://arxiv.org/abs/2412.14803)
- **Say, Dream, and Act**：选择并适配 video generation model，快速生成 future videos，再训练 action model 利用 generated videos 和 real observations 修正空间错误。[arXiv](https://arxiv.org/abs/2602.10717)
- **EVA**：指出 video world model + IDM 的 executability gap，并用 IDM rewards 对 generated videos 做 RL post-training。[arXiv](https://arxiv.org/abs/2603.17808)
- **GEM-4D**：用 dense 4D correspondence supervision 改善 video world model 几何一致性，再用 IDM 转成 executable robot trajectories。[arXiv](https://arxiv.org/abs/2605.22882)

### C. Future Prediction as Auxiliary Objective

- **World Models / Dreamer / DayDreamer / TD-MPC2**：传统 model-based RL 已长期使用 latent future prediction / imagination 来服务 policy learning。[World Models](https://arxiv.org/abs/1803.10122), [DayDreamer](https://arxiv.org/abs/2206.14176), [TD-MPC2](https://www.tdmpc2.com/)
- **Future Prediction Can be a Strong Evidence of Good History Representation**：在 POMDP 中，future observation prediction accuracy 与 RL 表现强相关，支持 future prediction 作为 representation learning 信号。[arXiv](https://arxiv.org/abs/2402.07102)
- **FLARE**：不预测 pixel future，而对齐 future latent observation embedding，用少量 future tokens 帮 policy 学 implicit world modeling。[arXiv](https://arxiv.org/abs/2505.15659), [Project](https://research.nvidia.com/labs/gear/flare/)

### D. Privileged Future / Foresight Distillation / Future-Conditioned Correction

- **PFD: Privileged Foresight Distillation**：把 teacher 看到 true future 与 student 只看 current 的 action denoising 差异定义为 foresight residual，并蒸馏到 current-only adapter。[arXiv](https://arxiv.org/abs/2604.25859)
- **Being-H0.7**：训练时 posterior branch 使用 future observations，推理时丢弃 posterior，通过 prior branch 当前信息推断 future-aware latent queries。[arXiv](https://arxiv.org/abs/2605.00078), [Project](https://research.beingbeyond.com/being-h07)
- **World Guidance (WoG)**：把 future observations 映射到 compact condition space，并注入 action inference pipeline；训练时学习条件，推理时 self-guided。[arXiv](https://arxiv.org/abs/2602.22010)

### E. Executability Gap / Action Feasibility of Generated Videos

- **EVA**：明确命名 executability gap：视频看起来合理，但 IDM 解码出的动作违反刚体 / 运动学 / 平滑性约束。[arXiv](https://arxiv.org/abs/2603.17808)
- **GEM-4D**：强调普通 VWM 缺少 point-level motion consistency，导致 generated videos 难以执行。[arXiv](https://arxiv.org/abs/2605.22882)
- **GigaWorld-Policy**：指出 joint reasoning over visual dynamics and actions 会带来 inference overhead 和 representation entanglement，因此用 action-centered 因果设计降低依赖。[arXiv](https://arxiv.org/abs/2603.17240)

### F. Action Anticipation / Future Representation Synthesis Outside Robotics

- **Action anticipation in egocentric vision**：早期工作已研究从 observed frames 预测 future action / object / trajectory；它说明“预测未来帮助行动理解”不是新思想。[IJCV egocentric vision survey](https://link.springer.com/article/10.1007/s11263-024-02095-7)
- **TTPP**：human action anticipation 中 joint future feature/action prediction 已出现，表明 joint prediction 机制在 video understanding 中早已有根。[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0925231221001697)

### G. Model-Based RL / Latent Dynamics / Planning with Learned World Models

- **World Models (Ha & Schmidhuber)**：VAE + RNN world model + controller，在 learned model 内训练 / 评估 policy。[arXiv](https://arxiv.org/abs/1803.10122)
- **DayDreamer**：把 Dreamer world model 应用到真实机器人学习。[arXiv](https://arxiv.org/abs/2206.14176)
- **TD-MPC2**：decoder-free latent world model + MPC，在 MetaWorld / ManiSkill2 等连续控制任务上扩展。[Project](https://www.tdmpc2.com/), [arXiv](https://arxiv.org/abs/2310.16828)

### H. Uncertainty-Aware World Model / Risk-Aware Policy

- **RAMCO / Risk-Aware Model-Based Control**：Bayesian dynamics model + CVaR，用 uncertainty-aware dynamics 做 risk-sensitive planning。[Frontiers](https://www.frontiersin.org/articles/10.3389/frobt.2021.617839/full)
- **Mind the Uncertainty**：在 MBRL 中区分 epistemic 和 aleatoric uncertainty，并用于 risk-aware / active exploration。[arXiv](https://arxiv.org/abs/2309.05582)
- **Risk-Aware RL for Mobile Manipulation**：用 distributional RL teacher 训练 runtime-adjustable risk-sensitive visuomotor student。[arXiv](https://arxiv.org/abs/2603.04579)

## Occupancy Evaluation Table

| Paper | Year | Category | Coupling Type | Future Used at Inference? | Action Sees Future Explicitly? | Mechanism Studied? | Benchmarks | Main Finding | Evidence Strength | Close to My Idea 1-5 | Blocks My Idea? | Notes |
|---|---:|---|---|---|---|---|---|---|---|---:|---|---|
| World Models | 2018 | G | model-based latent dynamics | yes, latent imagination | indirectly | partial | CarRacing / VizDoom | learned world model can support policy learning | medium | 2 | no | 占据“想象未来用于控制”的老根，但不是 robot VLA coupling |
| DayDreamer | 2022 | G | latent dynamics + policy | yes | indirectly | performance-oriented | real robots | Dreamer-style world model can learn physical robots | medium | 2 | no | 证明 robotics world model 可行，不回答 video-action coupling |
| UniPi | 2023 | B | video-first + IDM | yes | yes, IDM sees generated video | partial | simulated manipulation / real robot demos | text-conditioned video can act as universal policy interface | medium | 4 | partially | 直接占据“先生成 future video 再 IDM” |
| Diffusion Policy | 2023 | action-only baseline | action-only diffusion | no | no | performance | 12 robot tasks | action diffusion is strong visuomotor baseline | strong | 2 | no | 你必须拿它或类似 action-only model 做 baseline |
| GR-1 | 2023/2024 | A/C | shared backbone + action/future heads | not necessarily full video rollout | no explicit generated future as input | partial | CALVIN / real robot | video generative pretraining + future image prediction improves manipulation | strong | 4 | partially | 很早占了 joint action + future image |
| RT-2 | 2023 | action-only VLA | action-only | no | no | performance | real robot | VLA transfers web knowledge to action | strong | 2 | no | 代表 direct VLA baseline |
| Octo | 2024 | action-only generalist policy | action-only diffusion/transformer | no | no | performance | Open X-Embodiment | open-source generalist policy | strong | 2 | no | undergraduate baseline 候选 |
| OpenVLA | 2024 | action-only VLA | action-only | no | no | performance | Open X / LIBERO | open 7B VLA action model | strong | 2 | no | 不研究 future，但必须对比 |
| Future Prediction as History Representation | 2024 | C/G | auxiliary future prediction | no direct robot future video | no | yes, representation correlation | POMDP RL | future prediction quality correlates with policy performance | medium | 3 | no | 支持机制实验的理论动机 |
| VPP | 2025 | B/C | predicted future representation + implicit IDM | yes, internal future reps | yes, via reps | partial | robot policy benchmarks | VDM future representations help action learning | strong | 5 | partially | 非常接近你的 video/latent-first + IDM |
| UWM | 2025 | A/B | unified diffusion; policy/forward/IDM/video modes | optional by mode | depends on mode | partial | DROID / real & sim | coupling video/action diffusion improves generalization | strong | 5 | partially | 强占坑，但大规模 |
| FLARE | 2025 | C/D | future latent alignment | no explicit future at inference | no | yes-ish | multitask sim / real GR-1 | future latent tokens improve policy with minimal overhead | strong | 5 | partially | 直接威胁“future as latent regularizer” |
| AMPLIFY | 2025 | B/F | action-free motion prior + IDM | yes, latent motion | yes, IDM sees predicted motion | partial | LIBERO / video datasets | decoupling motion prediction and action inference helps low-data regimes | medium-strong | 4 | partially | 强占 modular decoupling |
| Motus | 2025 | A/B | unified latent WAM + MoT | yes/optional by mode | yes in joint mode | partial | sim + real | one model supports video/action/IDM/joint modes | strong | 5 | partially | 占了 unified multi-mode WAM |
| LingBot-VA | 2026 | A | shared latent space + joint frame/action | yes, async / closed-loop | coupled tokens | partial | sim + real | frame prediction and policy execution jointly learned | strong | 5 | partially | 强占 joint WAM |
| DreamZero | 2026 | A | joint video-action WAM | latent future/action inference | jointly modeled | partial | AgiBot / DROID / real robots | WAM outperforms VLA zero-shot generalization | strong | 5 | partially/yes | broad idea 被它强占，但机制比较没完全结束 |
| WoG | 2026 | D | future condition space + action generation | predicted compact future condition | yes, via condition | yes-ish | sim + real | compact future conditions beat raw future prediction methods | strong | 5 | partially | 很接近“future branch 作用机制” |
| Say, Dream, and Act | 2026 | B | video-first + action model | yes | yes | partial | instruction robot manipulation | generated videos support spatial correction | medium | 4 | partially | 占 modular video-conditioned action |
| EVA | 2026 | E | video-first + IDM reward alignment | yes, generated video evaluated by IDM | IDM sees generated video | yes, executability | RoboTwin / real bimanual | executability gap can train video model | medium | 4 | partially | 很适合你扩展 risk/executability |
| GigaWorld-Policy | 2026 | A/D | action-centered WAM; future video optional | no, can drop video branch | causal mask prevents leakage | yes-ish | real robots / RoboTwin 2.0 | train complex, infer simple improves speed and performance | strong | 5 | partially/yes | 强占“推理时不看未来”的路线 |
| PFD | 2026 | D | privileged future distillation | no | teacher yes, student no | yes | LIBERO / RoboTwin | future is compressible action-denoising correction, not just regularizer | medium | 5 | partially/yes | 最威胁你的机制 novelty |
| Being-H0.7 | 2026 | D | posterior future branch -> prior latent | no | training posterior yes | yes-ish | six sim benchmarks + real | latent future-aware reasoning without rollout | medium-strong | 5 | partially | 占 train-time future / inference current-only |
| GEM-4D | 2026 | B/E | geometry video + IDM | yes | yes | executability/geometry | sim + real | 4D correspondence improves executable video rollouts | medium | 4 | partially | 占 video executability alignment |
| RAMCO / Risk-Aware MBC | 2021 | H | uncertainty-aware dynamics planning | yes, model rollout | no video | yes | walking robot model | CVaR over model uncertainty improves risk-sensitive control | medium | 3 | no | 给你 uncertainty 切口理论支持 |
| Mind the Uncertainty | 2023 | H | uncertainty-aware MBRL | yes | no video | yes | safety-critical control | separating epistemic/aleatoric matters | medium | 3 | no | 与 WAM coupling 尚未完全结合 |

## Timeline of Research Progress

### 1. Old world models / model-based RL

早期问题是：能不能学一个内部环境模型，然后在模型里规划或训练 policy？

```text
o_t, a_t -> world model -> predicted future state
predicted future -> planning / policy learning
```

World Models、Dreamer、DayDreamer、TD-MPC2 解决了“未来预测可用于控制”的基础问题。没解决的是：语言任务、真实机器人多任务、web video pretraining、action-free video、VLA 中 video/action 如何耦合。

### 2. Future prediction as auxiliary loss

这一阶段开始把 future prediction 作为 representation shaping：

```text
policy loss + future prediction loss
```

直觉是：如果 latent 必须预测未来，它就不能只记当前外观，必须包含 dynamics-relevant information。GR-1 和 FLARE 都属于这个思想的机器人版本。没解决的是：future loss 提升 action 的原因到底是 regularization、planning 还是 correction。

### 3. VLA and diffusion policy

RT-2、OpenVLA、Octo、π0、Diffusion Policy 把重点放在：

```text
current observation + language -> action
```

它们强大、直接、推理快，但缺点是 action supervision 稀疏，容易学 shortcut，对 OOD physical motion 泛化有限。这自然引出：能不能用 video / future dynamics 给 action policy 更密集的物理监督？

### 4. Video world models for robotics

UniPi、VPP、Say-Dream-Act 等把 video generation 作为 plan：

```text
current image + language -> future video / future latent
future video / latent -> IDM / action model -> robot action
```

优势是可解释、可利用 action-free video；缺点是生成慢、误差级联、generated video 可能不可执行。

### 5. World Action Models

GR-1、UWM、Motus、LingBot-VA、DreamZero、MotuBrain 把 video 和 action 放入统一模型：

```text
current obs + language -> future visual tokens + action tokens
```

这个阶段解决了“joint modeling 是否能 scale”的问题。没完全解决的是：不同耦合程度在小模型、低数据、OOD、失败诊断下的机制差异。

### 6. Privileged future / executability alignment

GigaWorld-Policy、PFD、Being-H0.7、WoG、EVA 开始针对机制和部署成本：

```text
训练时用 future
推理时尽量不生成 full future video
把 future 压缩成 latent condition / residual / reward / executability signal
```

这说明研究前沿正在从“是否预测未来”转向“预测什么未来、未来如何进入 action、推理时要不要保留未来分支”。

## Mechanism Analysis

### Future Prediction as Representation Regularizer

机制：

```text
action-only policy:
o_t -> z_t -> a_t

future-regularized policy:
o_t -> z_t -> a_t
          -> predict z_{t+k} / image_{t+k}
```

future loss 迫使 `z_t` 包含位置、速度、接触、遮挡、目标进度等动态信息。它可能不直接“规划”，但会让 representation 更适合 action。

危险点：如果 future prediction 是 pixel reconstruction，模型可能浪费容量预测背景纹理；这也是 FLARE / WoG / Being-H0.7 转向 latent / condition space 的原因。

### Future Prediction as Implicit Planner

如果模型在推理时真的生成未来：

```text
candidate future -> evaluate -> choose action
```

那它更像 planner。UniPi 和 Say-Dream-Act 更接近这一类。优点是可解释；缺点是慢、误差级联，并且“好看的视频”不等于“机器人能执行”。

### Future Prediction as Dense Dynamics Supervision

机器人 action label 稀疏：

```text
每个 timestep 只有一个 action target
```

但未来视频 / latent 提供更密集的监督：

```text
未来每一帧、每个 latent token、每个 object motion 都是 supervision
```

这对低数据机器人学习很重要。GR-1、UWM、DreamZero 都借助 dense video signal 改善动作泛化。

### Future Prediction as Future-Conditioned Correction

PFD 的核心观点是：

```text
teacher action prediction with true future
-
student action prediction with current only
= foresight residual
```

这说明 future branch 不只是 regularizer，而是能给 action denoising 一个方向性 correction。这个机制非常接近你的核心问题，也是目前最威胁 novelty 的工作。

### Video-First Planning + Inverse Dynamics

流程：

```text
o_t + goal -> future video / latent
future video / latent + current state -> IDM -> action sequence
```

优点：

- 模块清楚；
- 可利用 action-free videos；
- future plan 可视化；
- embodiment 可以通过 IDM 适配。

缺点：

- video generation 慢；
- future video error 会传给 IDM；
- generated video 未必符合 robot kinematics；
- IDM 对 OOD generated video 可能不稳。

### Executability Gap

EVA 给这个问题一个很清楚的名字：

```text
visually plausible future != physically executable future
```

比如视频里 cube 平滑移动到目标，但实际 robot action 需要穿过桌面、超出关节限制、夹爪没抓住。这个 gap 是你做小项目时很好的 evaluation 切口。

### Train-Time Future vs Inference-Time Future

两种路线：

```text
路线 1:
训练时预测未来，推理时也生成未来

路线 2:
训练时用未来监督，推理时只看当前
```

路线 1 更像 planning，解释性强但慢。路线 2 更像 representation shaping / distillation，部署快但机制难证明。GigaWorld-Policy、PFD、Being-H0.7 都在推路线 2。

### Shared Backbone vs Separate Model

共享 backbone：

```text
shared encoder / transformer
  -> action head
  -> future head
```

优点是特征共享、训练高效、动作受 future supervision 影响。缺点是目标冲突：video prediction 关心外观细节，action prediction 关心可控动态。

分离模型：

```text
video model -> IDM / action model
```

优点是模块清楚、可单独替换。缺点是 interface bottleneck 和 error cascading。

## Gap Analysis

### Gap 1: 同一 benchmark 下的 coupling degree 系统比较不足

已有论文做到：各自提出 action-only、joint、latent future、video+IDM、privileged future。  
缺什么：同一数据、同一模型容量、同一任务上的 controlled comparison。  
研究价值：能回答“到底应该耦合到什么程度”。  
本科生可做：可以。  
算力：1 张 12–24GB GPU 足够做小规模 latent/image 64x64。  
最小实验：MetaWorld pick-place / ManiSkill PickCube 上比较 5 个小模型。

### Gap 2: Future prediction quality 与 action success 的因果关系不清楚

已有论文做到：报告 future quality 或 success rate。  
缺什么：future error 是否预测 action failure、是否 OOD 时更相关。  
研究价值：决定 future branch 能不能做 risk signal。  
本科生可做：很适合。  
算力：低到中等。  
最小实验：记录每步 future MSE / latent error / uncertainty，与之后 k 步失败或 recovery 相关性。

### Gap 3: Generated future 的 executability 小尺度诊断不足

已有论文做到：EVA / GEM-4D 在大模型上定义并优化 executability。  
缺什么：简单任务中 executability gap 的可解释可视化指标。  
研究价值：连接 world model 和 safe execution。  
本科生可做：可以。  
算力：中等。  
最小实验：用 IDM 解码 generated future，统计 action jerk、workspace violation、IK failure、gripper mismatch。

### Gap 4: Privileged future 是否只是更强 supervision，还是真正 correction？

已有论文做到：PFD 已经提出 foresight residual。  
缺什么：更小、更可复现的 toy-to-robot benchmark，验证不同扰动下 residual 何时有效。  
研究价值：机制清楚，适合 workshop。  
本科生可做：有挑战但可做。  
算力：中等。  
最小实验：teacher sees future latent，student current-only，adapter distills correction；比较 perturbation recovery。

### Gap 5: Uncertainty-aware coupling 还没有和 WAM 主流路线充分结合

已有论文做到：risk-aware MBRL / uncertainty-aware planning 已有；WAM 论文多关注性能和速度。  
缺什么：future prediction uncertainty 如何调节 action confidence / stop / replan。  
研究价值：与你的 uncertainty-aware safety 背景高度匹配。  
本科生可做：非常适合小 demo。  
算力：低到中等。  
最小实验：ensemble world model 或 dropout future head，预测 risk / action failure，超过阈值 stop。

## Undergraduate Feasibility Assessment

### 不建议做

| Direction | 为什么不建议 |
|---|---|
| 训练大规模 WAM / VLA | 需要海量 robot/video data、多机 GPU、工程复杂度高 |
| 复现 DreamZero / Motus / MotuBrain | 参数规模和数据管线超出本科生合理范围 |
| 做 web-scale video pretraining | 数据清洗、版权、算力、训练稳定性都很重 |
| 真实机器人长期实验 | 如果没有实验室资源，周期和硬件风险太大 |

### 可以做但风险大

| Direction | 风险 |
|---|---|
| LIBERO 上复现 FLARE / PFD | 代码和依赖可能复杂，benchmark 训练耗时 |
| video-first + IDM with diffusion video | 生成视频慢，质量差时很难 debug |
| 大模型 VLA finetuning | 显存和数据格式容易卡住 |

### 最推荐的小切口

| Candidate | Novelty | Feasibility | Compute Cost | Engineering Difficulty | Publication Potential | Advisor Appeal | PhD Value |
|---|---:|---:|---:|---:|---:|---:|---:|
| Coupling degree controlled benchmark | 4 | 5 | 2 | 3 | 3 | 4 | 5 |
| Future error as action failure/risk predictor | 4 | 5 | 2 | 3 | 3 | 5 | 5 |
| Privileged future residual on small manipulation | 3 | 4 | 3 | 4 | 3 | 4 | 4 |
| Video-first + IDM vs joint latent policy | 3 | 3 | 3 | 4 | 3 | 4 | 4 |
| Uncertainty-aware future-action coupling | 4 | 4 | 3 | 3 | 4 | 5 | 5 |

Top 3 推荐：

1. **How Much Should a Robot Policy See Its Predicted Future?**  
   做同一任务下 action-only / joint / video-first / privileged / uncertainty 的小规模比较。
2. **Can Future Prediction Error Predict Robot Action Failure?**  
   把 future branch 从 performance module 变成 risk diagnostic module。
3. **Privileged Future as Distillable Action Correction in Small Robot Manipulation**  
   做 PFD 思想的小型可复现版本，强调 mechanism 而不是 SOTA。

## Final Recommendation

是否值得继续：**值得，但不能做大而泛的 WAM。**

应该缩小为：

```text
在一个小型 manipulation benchmark 中，
系统比较 policy 对 predicted/privileged future 的可见程度，
并分析 future prediction quality / uncertainty 与 action failure 的关系。
```

第一个 demo：

```text
ManiSkill PickCube 或 MetaWorld PickPlace
64x64 RGB 或 latent state
训练 5 个模型：
1. action-only BC / diffusion policy 小版
2. shared encoder + action head + future latent head
3. future latent first + IDM
4. privileged future teacher -> current-only student
5. joint future head + uncertainty / ensemble risk head
```

第一个月应该读：

1. GR-1
2. UniPi
3. VPP
4. FLARE
5. DreamZero
6. WoG
7. GigaWorld-Policy
8. PFD
9. EVA
10. AMPLIFY

第一个月应该复现：

- 不是复现 DreamZero；
- 而是复现一个小 action-conditioned world model：

```text
o_t, a_t -> predict z_{t+1}
z_t -> action
```

然后加 auxiliary future loss，看 action success / OOD 是否提升。

三个月内能做出：

- 一个可运行 benchmark；
- 5 个 coupling baselines；
- success / OOD success / latency / smoothness / future error / failure correlation；
- 一组可视化：future prediction 越差，动作越容易失败吗？
- 一个 workshop-style paper draft。

不要碰：

- 从零训练大视频 diffusion；
- 复现 14B WAM；
- 真实机器人作为第一阶段；
- 只做“我也 joint predict future and action”的重复工作。

## 核心判断题回答

### Question 1: Joint video-action prediction 本身是不是已经被占坑？

**回答：部分是，接近“是”。**

GR-1、UWM、Motus、LingBot-VA、DreamZero、MotuBrain 已经做了 joint video-action / world-action modeling。DreamZero 明确把 WAM 定义为 jointly modeling future video states and robot actions；UWM / Motus 还支持多种 query mode。因此 broad idea 已经被占。

但没有完全堵死的是：小模型、同一 benchmark、同一数据预算下的 mechanism comparison。

### Question 2: 系统比较不同解耦程度是否已经有人做了？

| Coupling | 覆盖程度 | 判断 |
|---|---|---|
| action-only | 很强 | Diffusion Policy、RT-2、OpenVLA、Octo、π0 |
| joint video-action | 很强 | GR-1、UWM、Motus、LingBot-VA、DreamZero |
| shared backbone separate heads | 中到强 | GR-1、FLARE、GigaWorld-Policy |
| video-first + IDM | 强 | UniPi、VPP、Say-Dream-Act、EVA、GEM-4D |
| privileged future distillation | 新但很相关 | PFD、Being-H0.7 |
| uncertainty-aware coupling | 弱到中 | risk-aware MBRL 多，WAM-specific 少 |

系统比较：**没有被完全做透。** 这是你的机会。

### Question 3: Future branch 的机制是否已经被讲透？

| Hypothesis | 证据 | 判断 |
|---|---|---|
| regularization | GR-1 / future loss / GigaWorld 推理丢视频分支仍有效 | 部分支持 |
| representation shaping | FLARE / Being-H0.7 | 强支持 |
| implicit planning | UniPi / Say-Dream-Act / VPP | 支持，但成本高 |
| future-conditioned correction | PFD | 新强证据 |
| executability checking | EVA / GEM-4D | 支持，但仍新 |

结论：**没有完全讲透，但已经有人开始直击机制。**

### Question 4: 先预测 future video，再让 action expert 输出动作，会不会更好？

| 维度 | video-first + IDM |
|---|---|
| 可解释性 | 强，可以看 future plan |
| 泛化 | 对 language / action-free video 有优势 |
| 误差级联 | 高，video 错会带坏 IDM |
| 推理延迟 | 高，尤其 diffusion video |
| 可执行性 | 不保证，存在 executability gap |
| 训练难度 | 模块化但 pipeline 长 |
| 数据需求 | 可用 action-free video，但 IDM 仍需 action data |
| 适合任务 | 长 horizon、语言组合泛化、需要可视化 plan 的任务 |

它不一定更好。对实时 manipulation，latent future / train-time future / action-centered WAM 可能更实际。

### Question 5: 这个方向适不适合本科生？

**大模型方向不适合；小规模机制实验适合。**

需要：

- benchmark：MetaWorld / ManiSkill / LIBERO 子集；
- GPU：单张 12–24GB 可做 latent 小模型；
- 时间：1–2 个月 demo，3 个月可形成完整实验；
- 价值：适合 RA 面试、PhD application、workshop paper。

## 三个研究路线建议

### Direction A: Safe Small Project

**Research question**：Future prediction error 能否预测 robot action failure？  
**Why open**：WAM 论文报告 success，但很少把 future error 当 failure/risk diagnostic。  
**Minimum experiment**：ManiSkill PickCube，训练 action-conditioned latent world model + action policy，记录 future error 与失败关系。  
**Dataset / simulator**：ManiSkill / MetaWorld scripted demos。  
**Baselines**：action-only, future auxiliary, ensemble uncertainty。  
**Metrics**：success, OOD success, future MSE, failure correlation, AUROC risk prediction。  
**Risks**：任务太简单时相关性不明显。  
**Expected contribution**：future prediction can be used as safety signal。  
**PhD value**：连接 world model + uncertainty-aware robot execution。

### Direction B: Medium Project

**Research question**：How much future should a robot policy see?  
**Why open**：已有论文做各自架构，但缺统一小规模机制比较。  
**Minimum experiment**：比较 action-only、joint future-latent、video/latent-first+IDM、privileged future、uncertainty head。  
**Dataset / simulator**：LIBERO-spatial 或 ManiSkill PickCube / StackCube。  
**Baselines**：Diffusion Policy 小版、BC-RNN、world-model auxiliary。  
**Metrics**：success, OOD, smoothness, latency, future quality, recovery after perturbation。  
**Risks**：工程量较大，需要严格控制模型容量。  
**Expected contribution**：coupling degree design guide。  
**PhD value**：研究问题清楚，能在面试中讲机制。

### Direction C: Ambitious Project

**Research question**：Can uncertainty-aware future latent prediction improve safe robot execution under OOD perturbations?  
**Why open**：uncertainty-aware MBRL 和 WAM 还没有充分结合。  
**Minimum experiment**：ensemble future latent model + risk head + stop/replan policy。  
**Dataset / simulator**：ManiSkill + obstacle / forbidden zone / perturbations。  
**Baselines**：action-only, deterministic future, ensemble risk, CVaR risk。  
**Metrics**：unsafe action rate, false stop rate, success, risk calibration ECE, OOD AUROC。  
**Risks**：risk label 设计要严谨。  
**Expected contribution**：future uncertainty as execution guard。  
**PhD value**：非常贴合 safe embodied AI。

# Final Decision

## Is the topic occupied?

Red / Yellow / Green: **Yellow**

Broad joint video-action prediction 已经接近 Red；但小规模机制比较、failure correlation、uncertainty-aware coupling 仍是 Yellow/Green。

## Best narrowed research question

**How much predicted or privileged future information should a robot policy use, and when does that future information improve robustness rather than merely regularize representation learning?**

## Why this is still worth studying

因为现在的 WAM 论文大多证明“我的架构有效”，但没有把 action-only、joint、video-first、privileged、uncertainty 在同一小环境里做成清楚的机制对照。这个问题对实际 robot policy 设计很重要：推理时生成 future 很贵，如果训练时 future 已经足够，就没必要上线时 dream video。

## Why this may be too hard

如果你试图复现 DreamZero / Motus / GigaWorld 级别模型，会被算力、数据、工程管线卡死。另一个风险是小任务太 toy，导致 future branch 的优势不明显。

## Best undergraduate-scale experiment

ManiSkill / MetaWorld 上做 5-way coupling ablation：

```text
1. action-only
2. joint future-latent + action
3. latent/video-first + IDM
4. privileged future teacher -> current-only student
5. joint future + uncertainty/risk head
```

重点不追 SOTA，而追：

```text
future error 是否预测 action failure
OOD 下哪种 coupling 最稳
推理 latency 与 success 的 tradeoff
```

## Papers I must read first

1. GR-1: Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation
2. UniPi: Learning Universal Policies via Text-Guided Video Generation
3. VPP: Video Prediction Policy
4. FLARE: Robot Learning with Implicit World Modeling
5. DreamZero: World Action Models are Zero-shot Policies
6. World Guidance
7. GigaWorld-Policy
8. Privileged Foresight Distillation
9. EVA
10. AMPLIFY

## Papers that most threaten novelty

1. Privileged Foresight Distillation
2. GigaWorld-Policy
3. FLARE
4. DreamZero
5. World Guidance

## Recommended next 7 days

Day 1: 读 GR-1、UniPi，只画 data flow：future 在哪里，action 在哪里。  
Day 2: 读 FLARE、VPP，重点看 future latent 如何进入 action policy。  
Day 3: 读 DreamZero、GigaWorld-Policy，记录它们如何避免推理时 video overhead。  
Day 4: 读 PFD、Being-H0.7，整理 privileged future / train-time future 机制。  
Day 5: 读 EVA、AMPLIFY，整理 video-first + IDM 和 executability gap。  
Day 6: 选一个环境，写数据收集脚本：`obs_t, action_t, obs_{t+1:t+k}, success/failure`。  
Day 7: 实现 action-only baseline 和 future-latent auxiliary baseline。

## Final advice

你应该继续深挖，但不要把目标设成“我提出一个新的 World Action Model”。这个坑的大方向已经被大组和大算力快速占住了。你的机会是做一个干净、可信、机制导向的小实验，回答“大模型论文没有系统回答的问题”：future branch 到底什么时候帮 action，帮在哪里，能不能预测失败，推理时是否值得保留。最推荐先做 ManiSkill / MetaWorld 的小规模 ablation，把 `future prediction quality -> action failure / OOD robustness / uncertainty` 这条链证明清楚。这个方向很适合作为 RA 面试和 PhD 申请项目，因为它显示你不是只追热点，而是在拆解热点背后的机制。
