# 电缆表面缺陷检测：方向结构研究原型

本仓库提供一个用于论文前期实验的检测网络原型，以及在同一数据划分上比较基线、方向分支和结构张量门控的最小实验框架。**这是待验证的研究假设，不是已经证明的新颖方法，也不声称达到 SOTA。**

## 方法与对照

以 YOLO11 的 P3 检测特征为实验插入位置，分别测试：

| 模式 | 作用 |
| --- | --- |
| `baseline` | 不插入新模块。 |
| `isotropic` | 插入普通深度可分离方形卷积与残差，检查额外模块容量的影响。 |
| `uniform` | 使用水平、垂直、两条对角线四个深度方向分支，等权融合。 |
| `tensor` | 使用与 `uniform` 相同的方向分支，根据局部结构张量生成方向权重。 |
| `contrast` | 根据局部结构与较大邻域结构的差异，对相同方向分支进行路由。 |
| `hybrid` | 将普通方形卷积与局部张量路由分支结合，检验二者是否互补。 |

方向组件使用特征图的局部梯度估计双角方向分量；弱结构区域的方向权重趋近均匀。`contrast` 的假设是局部损伤与周围纹理的方向差异能够提供额外线索，但与背景方向一致的真实划痕也可能受到影响，必须通过实验证实。输出通过可学习残差系数与原特征相加。消融必须区分“加了参数”“加了方向分支”“加了张量门控”以及“局部—邻域差异”的影响；这些模式的参数量并不完全相同，不能把差异全部归因于单一因素。

代码保留已有的 YOLO11 检测头和预训练骨干，不是从零设计的全新检测器。结构张量、方向卷积及细长缺陷建模均有相关先例；只有在进一步查重并经严格实验验证后，才能判断是否具有足够的论文创新性。

## 数据与隐私

本仓库不提供数据、图片、标签、划分文件、权重、模型输出或训练日志。使用者需要自行提供带有 `images/train`、`labels/train` 和 `train_ready.txt` 的 YOLO 格式数据集，以及本地分组文件和可信的预训练权重。

分组文件由 `folds`（三个图片文件名 stem 列表）和 `groups`（包含 `fold`、`stems` 的列表）组成。代码在生成训练清单前检查图片不跨组、训练清单完整、标签类别及边界有效，并只使用 `images/train` 中的图像。根据文件名、时间或图像相似性推断的组并不保证物理电缆段独立，论文最终划分应由真实电缆段、采集批次和拍摄条件确定。

注意：人工生成或模型辅助修订的标签不能自动视为独立金标。验证集用于调参与模型选择，不能同时作为独立测试集。

## 安装与运行

建议使用单独的 Python 环境，并根据硬件先安装适合的 PyTorch；随后安装依赖：

```powershell
python -m pip install -e .
```

以下示例中的路径均为占位符，请指向本机文件；每个实验的 `--work` 必须是不存在的新目录。运行过程会禁用网络连接，必须提前准备本地权重。

```powershell
python -m cable_research.train --dataset DATASET_ROOT --fold-plan PRIVATE_PLAN.json --weights LOCAL_YOLO11N.pt --work ./local/baseline --mode baseline --fold 2 --seed 42 --epochs 25 --imgsz 640 --batch 4
```

将 `--mode baseline` 改为 `isotropic`、`uniform`、`tensor`、`contrast` 或 `hybrid`，并为 `--work` 指定不同的新目录，以相同超参数开展消融。`--sites p3p4` 可同时在 P3 和 P4 插入模块；`--channel-gate` 可组合 SE 风格的通道门控。位置或模块组合变化必须与参数量、计算量及单因素版本一起比较。每次训练仅保存本地 `results.csv` 和检查点，不会自动上传。

也可用 `python -m cable_research.grid` 构造模式、位置、通道门控及随机种子的有限组合。默认只打印计划，不启动训练；只有显式添加 `--execute` 才会依次运行，最多 24 组。重复在同一验证集上搜索会产生选模偏差，必须保留未用于搜索的独立测试集。

作为额外的跨架构参考，`baseline` 模式也可使用受支持的 YOLO26 本地权重，或使用 `--family rtdetr` 和本地 RT-DETR 权重；这些不是同一骨干上的模块消融，不同预训练来源、批量和分辨率也必须报告。下例测量本地可信检查点的网络前向延迟，不包含预处理、后处理或磁盘 I/O：

RT-DETR 的 CUDA 可变形注意力包含无确定性反向实现的算子；即使固定随机种子并请求确定性运行，也可能出现警告和结果波动，不能声称逐位复现。参见 [RT-DETR 官方说明](https://docs.ultralytics.com/models/rtdetr/)。

为加快探索，默认使用 `--cache ram`；Ultralytics 会提示该缓存方式可能影响确定性。更严格的复现实验可设置 `--cache none`，并记录环境、运行多次和报告波动；关闭缓存也不能保证所有 CUDA 算子完全确定。

训练提供 `--optimizer` 与 `--lr0` 以检验不同架构的训练配方；与网络结构对照不同，改变优化器或学习率属于另一因素，需要单独记录和消融。

YOLO 的 `baseline` 模式还可用 `--architecture` 指定已安装的架构 YAML（例如带 P2 输出的结构），随后从本地权重迁移兼容参数。此时不是全部层都能迁移，必须记录实际迁移数量，不可与完整预训练模型当作严格的单因素比较。

```powershell
python -m cable_research.benchmark --weights LOCAL_MODEL.pt --out ./local/latency.json --device cuda:0 --imgsz 640
```

要用同一验证协议提取各类别 AP，可使用下列命令。输入的数据配置仍须自行确认分组及标签质量；重复使用它选模不能构成独立测试。

```powershell
python -m cable_research.evaluate --weights LOCAL_MODEL.pt --data VALIDATION_CONFIG.yaml --work ./local/evaluation --imgsz 640 --batch 4
```

```powershell
python -m unittest discover -s tests -v
python -m ruff check .
```

## 论文验证最低要求

1. 在分组开发集上固定对照、训练预算与超参数，记录每一组实验，而不是只挑最高分。
2. 在此前没有用于标签生成、训练、调参或选模的独立电缆段上评估；每类缺陷均需足够正样本，并复核漏标。
3. 报告分类型 AP、mAP50、mAP50–95、召回率、参数量、显存和相同硬件条件下的延迟；多个种子和电缆段级置信区间应一并报告。
4. 与适合的强基线及公开数据集比较，保持同一划分、标注、输入分辨率、评测实现与训练预算；不同论文的分数不能直接拼成同一排行榜。
5. 公开论文中明确训练数据、标签来源、潜在近重复、失败案例和局限；在真实独立基准上确认之后再讨论 SOTA。

## 参考工作及许可

- [工业微小缺陷方向归纳偏置与特征重建，2026](https://www.mdpi.com/2076-3417/16/22/11181)。
- [电缆小目标缺陷检测 YOLO-ESBD，2026](https://www.mdpi.com/2079-9292/15/10/1979)。
- [SFEDet：空间—频率与边缘引导缺陷检测，2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13562717/)。
- [RF-DETR：实时检测器架构搜索，ICLR 2026](https://arxiv.org/abs/2511.09554)。
- [结构张量用于旋转目标检测，2024](https://arxiv.org/abs/2411.10497)。
- [Squeeze-and-Excitation Networks，2018](https://arxiv.org/abs/1709.01507)。
- [表面缺陷检测可复现和统计评测，2025](https://link.springer.com/article/10.1007/s10845-025-02672-8)。

使用者应自行确认数据及预训练权重的授权。Ultralytics 及其权重适用其自身许可条款，不能假设与本仓库代码的许可相同。
