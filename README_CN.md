# Daily-GasTurbineML

> 燃气轮机 + 物理信息神经网络 / 深度学习领域每日文献追踪。

[English Version](README.md)

## 作者

**熊雄 (Xiong Xiong)**

- 西北工业大学 (NWPU)
- 研究方向：AI4PDE、物理信息深度学习、数据驱动发现
- 邮箱：xiongxiongnwpu@mail.nwpu.edu.cn
- [Google Scholar](https://scholar.google.com/citations?user=j1M9tkwAAAAJ&hl=zh-CN&oi=sra)
- [ResearchGate](https://www.researchgate.net/profile/Xiong-Xiong-19?ev=hdr_xprf)
- [Physics-informed-vibe-coding](https://github.com/xgxgnpu/Physics-informed-vibe-coding) — 首个专注于 PINN Vibe Coding 研究的开源仓库。

---


## 概述

本仓库独立追踪 **机器学习与物理信息方法用于燃气轮机及叶轮机械** 的最新研究，涵盖6个类别，共 **106 篇论文**。

> **快速导航** — 点击下方任意类别跳转：
>
> [1. PINN-燃气轮机](#1-pinn--燃气轮机与叶轮机械) | [2. 代理模型](#2-代理模型--性能预测) | [3. 深度学习-气动](#3-深度学习--叶轮机内流与气动) | [4. CFD-ML](#4-cfd-与机器学习融合) | [5. 燃烧](#5-燃烧室机器学习) | [6. 健康监测](#6-健康监测与故障诊断)

---

### 1. PINN — 燃气轮机与叶轮机械

物理信息神经网络直接应用于燃气轮机与叶轮机械仿真：

- 压气机/涡轮叶片气动的PINN方法
- 叶片流道中基于物理约束的流场预测
- 叶轮机械中的反问题与参数识别
- 融合CFD数据与物理定律的多保真度PINN

### 2. 代理模型 — 性能预测

替代昂贵仿真的数据驱动代理模型：

- 燃气轮机循环分析的神经网络代理
- 多工况压气机/涡轮特性图建模
- 偏设计点性能的高斯过程和深度学习方法
- 面向发动机仿真加速的降阶模型

### 3. 深度学习 — 叶轮机内流与气动

深度学习用于叶轮机械气动分析与设计：

- CNN/Transformer用于叶片流场预测
- 生成模型用于涡轮叶片形状优化
- 图神经网络用于叶栅气动分析
- 基于注意力机制的尾迹干涉和非定常流动

### 4. CFD 与机器学习融合

机器学习增强或替代叶轮机械CFD：

- ML增强湍流封闭模型（RANS、LES）
- 叶轮机械CFD的物理信息数据同化
- 快速流场预测的神经算子
- 叶片流道二次流的ML修正RANS

### 5. 燃烧室机器学习

燃气轮机燃烧室与热管理的机器学习：

- 点火和火焰动力学的神经网络燃烧模型
- 基于ML的NOx与排放预测
- 涡轮冷却设计与优化的深度学习方法
- 稀疏传感器数据的温度场重建

### 6. 健康监测与故障诊断

深度学习用于燃气轮机状态监测、故障检测和预测性维护：

- LSTM/CNN用于燃气轮机退化预测
- 振动和温度传感器数据的异常检测
- 涡轮部件剩余使用寿命估计
- 跨发动机故障诊断的迁移学习

---

**论文总数**: 106

## 使用方法

```bash
pip install arxiv requests

# 检索最新论文（每类10篇）
python fetch_today.py

# 指定每类数量
python fetch_today.py --per_category 15

# 追加模式（不覆盖现有）
python fetch_today.py --append 10

# 跳过代码搜索（更快）
python fetch_today.py --no_code_search
```

## 许可

MIT
