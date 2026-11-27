# 细微缺陷：输入分辨率与切片评估线索

以下方法用于研究假设生成，不能将其他数据集论文的 AP 直接当成本项目效果。检索日期：2026-11-27。

1. [SAHI：Slicing Aided Hyper Inference and Fine-tuning](https://arxiv.org/abs/2202.06934) 将高分辨率图像切片用于小目标检测，也研究切片微调。本仓库 `sliced_eval` 是全图、重叠切片及类别感知 NMS 的简化对照，不是 SAHI 官方复现，也没有实现切片训练。
2. [When Does Sliced Inference Pay Off?](https://arxiv.org/abs/2611.12657)（2026 预印本）指出：切片收益可能来自有效输入分辨率，并且合并方法会影响结果。实验应比较相同权重的全图 640、全图高分辨率、仅切片、全图加切片，不能只比较切片与低分辨率基线；本仓库没有复现该文 SCAHI。
3. [WPFormer](https://openaccess.thecvf.com/content/CVPR2025/html/Yan_Wavelet_and_Prototype_Augmented_Query-based_Transformer_for_Pixel-level_Surface_Defect_CVPR_2025_paper.html) 利用小波和原型查询研究像素级表面缺陷，但只有矩形框时不能宣称已复现像素级监督。
4. [Noise-Aware Evaluation of Object Detectors](https://openaccess.thecvf.com/content/WACV2025/papers/Llerena_Noise-Aware_Evaluation_of_Object_Detectors_WACV_2025_paper.pdf) 讨论漏标、误标和边界误差对检测评测的影响。模型辅助修订后的验证标注仍需独立人工复核；切片和后处理不能替代真实金标。
5. [CableInspect-AD，NeurIPS 2024](https://mila-iqia.github.io/cableinspect-ad/) 含多种电缆、长划痕及框/掩膜标注，可考虑作为外部对照或预训练来源。其数据采用 CC BY-NC-SA 4.0，使用及再发布前须核对具体限制；不同电缆材质和成像条件带来的域差不能忽略。
6. [Ultralytics Mosaic 文档](https://docs.ultralytics.com/guides/yolo-data-augmentation) 支持控制拼图增强概率及训练末期关闭时机。Mosaic 的帮助或伤害取决于目标尺寸、上下文和数据，需作为独立训练因素比较，不能预先断言结论。

## 实验边界

- 先按原始图像和可识别的物理实体/采集批次分组，再生成训练切片；同一原图的切片不可跨训练与验证。现阶段只有近似分组，不能当成最终独立测试。
- 验证仍在整张原图的原始框上计算 AP。对长划痕，切片可能将同一目标截断为多段并形成伪重复框；全图预测提供上下文，但 NMS 可能误删相邻目标或无法拼接跨切片的细长目标，需报告类别 AP 和失败案例。
- 用户要求每条可区分划痕分别标框时，相邻目标的重叠框可能受到 NMS 阈值影响；应固定权重单独比较不同阈值，避免将后处理收益当成网络创新。
- 采用相同检测权重评估全图分辨率和推理策略；后续单独训练高分辨率模型时，必须记录 batch、显存和预训练差异。
- 多模型 NMS 集成可以检验不同检测器的互补性，但属于额外的推理计算，必须与单模型区分，不能当作单个模型的网络结构创新。
- `mAP50` 80% 是研究目标，不是承诺。还需报告严格定位指标 `mAP50–95`、划痕/孔洞分类别 AP、误报、漏检、真实独立测试和多个种子。
