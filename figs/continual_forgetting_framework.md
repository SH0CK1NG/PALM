# 连续学习（防遗忘/可控遗忘）总体框架图

下图给出了典型的连续学习（含选择性遗忘/可控遗忘）训练与评测流程，覆盖数据流、训练组件、防遗忘策略、目标函数以及指标体系。

```mermaid
flowchart LR
  %% 数据流与任务序列
  subgraph D[数据流 / 任务序列]
    T1[任务 1 / 数据块 1]
    T2[任务 2 / 数据块 2]
    TT[... 任务 t ...]
    TN[任务 N / 数据块 N]
  end

  %% 模型结构
  subgraph M[模型]
    BB[共享骨干 Backbone]
    subgraph Heads[任务/类别头 Heads]
      H1[Head@T1]
      Ht[Head@Tt]
      HN[Head@TN]
    end
    ADP[可选: 适配器/提示 Prompt/LoRA]
  end

  %% 记忆与教师模型
  subgraph K[知识与记忆]
    MEM["经验回放缓存 <br/>(样本/特征/原型)"]
    TM["教师模型 / 旧参数快照"]
  end

  %% 策略族
  subgraph S[防遗忘策略族]
    S1[回放类: ER/DER/iCaRL/原型]
    S2[正则类: LwF/EWC/SI/MAS]
    S3[结构类: 冻结/扩展/适配器/提示]
    S4[数据/损失重加权: 重要样本/难例挖掘]
    S5[可控遗忘/合规删除: 目标类/样本的反向蒸馏/惩罚]
  end

  %% 训练与优化
  subgraph Train[训练循环（t = 1..N）]
    MIX[混合批次: 当前任务数据 + 回放数据]
    Ltask["任务损失 L_task"]
    Ldist["蒸馏损失 L_distill<br/>(与教师/旧模型对齐)"]
    Lreg["正则损失 L_reg<br/>(参数重要性/约束)"]
    Lreplay["回放一致性 L_replay<br/>(特征/原型/Logits)"]
    Lunlearn["可选: 遗忘损失 L_unlearn<br/>(对指定类/样本抑制)"]
    SUM["总损失: L = L_task + λ1 L_distill + λ2 L_reg + λ3 L_replay + λ4 L_unlearn"]
    OPT["优化器更新 θ → θ'"]
  end

  %% 评测
  subgraph Eval[评测与监控]
    ACC[平均准确率 ACC]
    BWT[向后迁移 BWT]
    FWT[向前迁移 FWT]
    FORG[遗忘度 F]
    MEMC[内存/算力开销]
    ROB[鲁棒性/OOD/漂移]
  end

  %% 连接关系
  D -->|顺序到达| T1 --> T2 --> TT --> TN
  T1 & T2 & TT & TN --> MIX
  MIX --> BB --> Heads
  BB -. 可选 .- ADP
  MEM --> MIX
  TM --> Ldist
  S1 --> Lreplay
  S2 --> Lreg
  S3 --> ADP
  S4 --> Ltask
  S5 --> Lunlearn
  Ltask --> SUM
  Ldist --> SUM
  Lreg --> SUM
  Lreplay --> SUM
  Lunlearn --> SUM
  SUM --> OPT --> BB
  SUM --> OPT --> Heads
  ADP --> OPT
  BB -->|特征/原型| MEM
  Heads -->|Logits/原型| MEM
  BB & Heads -->|周期性/阶段性评测| Eval
  Eval --> ACC & BWT & FWT & FORG & MEMC & ROB
```

## 图例与说明（简）

- 回放类（S1）：使用缓存的样本/特征/原型来稳定旧任务表现（ER/DER/iCaRL 等）。
- 正则类（S2）：通过参数重要性或知识蒸馏抑制对旧知识的有害更新（EWC/LwF/SI/MAS）。
- 结构类（S3）：通过冻结、网络扩展、适配器/提示（Prompt/LoRA）等隔离干扰。
- 重加权（S4）：对重要样本/困难样本或类别进行加权以提升稳定性。
- 可控遗忘（S5）：针对合规/需求的类或样本进行“反向蒸馏/惩罚”以主动遗忘。

### 常用指标

- ACC：所有已学任务的平均准确率。
- BWT：新任务学习对旧任务的影响（负值代表遗忘）。
- FWT：先学任务对后续未见任务的正向迁移。
- F（Forgetting）：遗忘度；衡量各任务从最优点到当前点的性能下降。
- 资源：额外内存与计算开销；以及鲁棒性（OOD/漂移）。

> 使用方法：本文件为 Mermaid 图，Cursor/VS Code/Markdown 预览或 Git 平台通常可直接渲染；若需 PNG/SVG，可使用 mermaid-cli 进行导出。


