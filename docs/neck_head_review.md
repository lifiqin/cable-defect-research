# 面向划痕的 Neck 与检测头：文献和实验边界

检索日期：2026-11-27。以下是与当前任务相近的一手资料，不是完整查新，也不能把不同数据集的论文分数直接比较。

| 文献 | 可借鉴的问题与机制 | 与本仓库实验的关系 |
| --- | --- | --- |
| [Cable Defect Detection Using an Improved RT-DETR Model](https://www.mdpi.com/1424-8220/26/19/6006) | 在电缆缺陷 RT-DETR 中使用上下文引导下采样，关注长划痕等类别。 | `--family rtdetr --neck context` 将预训练下采样与局部/膨胀上下文残差结合；**不是**该论文完整复现。 |
| [YOLO-ESBD 电缆小缺陷检测](https://www.mdpi.com/2079-9292/15/10/1979) | 使用特征融合、细节保持与动态检测头。 | 当前加权/自适应拼接仅受加权融合启发，任务门控头仅做分支门控，均**不等同**原论文 BiFPN 或 DyHead。 |
| [ATFL-Swin-YOLO 高分辨率检测头](https://www.nature.com/articles/s41598-026-61060-1) | 通过 P2 浅层检测头及其他组件关注细小缺陷。 | `--head p2` 采用 P2–P5 四尺度拓扑；没有复现该文 Swin 与损失设计，且新增检测头不能完整继承预训练权重。 |
| [Dynamic Head 原论文](https://openaccess.thecvf.com/content/CVPR2021/html/Dai_Dynamic_Head_Unifying_Object_Detection_Heads_With_Attentions_CVPR_2021_paper.html) | 从尺度、空间和任务三个维度设计检测头。 | `taskgate` 仅对分类/定位分支分别施加轻量通道门控，**不是**完整 DyHead，也不能据此声明其效果。 |
| [EfficientDet / BiFPN](https://arxiv.org/abs/1911.09070) | 通过带权的双向金字塔融合多尺度特征。 | `weighted` 与 `adaptive` 改变原始拼接节点输入权重，保留原拓扑，不是完整 BiFPN。 |
| [TEEP-RCNN 划痕及表面缺陷检测](https://arxiv.org/abs/2609.28077) | 两阶段 Faster R-CNN 与纹理、边缘和注意力结合；为预印本。 | 适合作为非 YOLO 两阶段路线的对照与后续研究方向，是否复现以实际运行记录为准。 |
| [DPN-Detector 双金字塔及双头，2023](https://www.mdpi.com/2076-3417/13/2/1054) | 在 Faster R-CNN 中结合多尺度金字塔与区分分类、回归的双头，研究钢板表面缺陷。 | 本仓库仅用简化上下文 FPN 与分类/回归适配分支，未完整复现 DPN 或其双头；“两阶段加金字塔和双头”本身已有先例。 |
| [AMFNet，2025](https://link.springer.com/article/10.1007/s10845-025-02613-5) | 基于 Faster R-CNN ResNet50-FPN，融合多层特征和空间上下文。 | 为两阶段 Neck 设计提供参考，当前没有实现该方法。 |
| [SA-FPN，2025](https://www.sciencedirect.com/science/article/pii/S0263224125003781) | 用尺度感知注意力融合表面缺陷的多尺度特征。 | 与 Neck 重加权的设计目标有关，但本仓库未复现 SA-FPN。 |
| [RF-DETR，ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5bed8703db85ab27dc32f6a42f8fbdb6-Abstract-Conference.html) | Transformer 检测和不同预训练特征提供非 YOLO 对照。 | 可使用官方 Nano 权重、指定 Python 环境和本地离线训练；训练配方、后处理、评估实现可能与 YOLO 不同。 |
| [WPFormer，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Yan_Wavelet_and_Prototype_Augmented_Query-based_Transformer_for_Pixel-level_Surface_Defect_CVPR_2025_paper.html) | 小波、原型与查询式 Transformer 用于像素级表面缺陷识别。 | 面向细长缺陷的可选研究方向；现有框标注不是像素级真值，不能声称复现该分割方法。 |

## 因素设计

- **Neck**：标准拼接、可学习静态加权拼接、按图像统计量自适应加权拼接，以及依据局部与邻域方向差异进行逐位置路由的 `geometry` 候选。后者应排除正常线状纹理和反光干扰，并与方向无关的路由基线比较；不声称已证明创新性。
- **检测头**：标准 YOLO11 解耦头、分类/回归分支独立门控、P2–P5 四尺度头。
- **非 YOLO**：RT-DETR 标准 Neck 与上下文残差式下采样、独立的 RF-DETR 基线、Faster R-CNN 两阶段基线，以及该两阶段模型内上下文 FPN 与任务解耦适配器的单因素和组合。
- 先在同一开发折检查单因素，再比较合理组合；所有结果必须保留，包括负结果。P2 和跨框架试验存在不同的预训练转移、分辨率、优化器或批量，不能当成严格同条件消融。
- 分组由拍摄线索近似构造、标签含模型辅助修订；这些验证实验不是独立金标测试。应优先关注 `scratch` 的分类别 AP、召回和误报，而不是只比较总体 mAP。
