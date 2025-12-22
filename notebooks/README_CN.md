# CatBoost 银行营销数据分析 (CatBoost Analysis for Bank Marketing)

本项目使用 **CatBoost**，基于银行营销数据集预测客户是否会订阅银行服务（定期存款），并采用 **SHAP** 进行模型解释以及 **DiCE** 进行反事实分析， 帮助销售代表更好地理解客户行为（如更改那些变量可以影响预测结果），优化营销策略。

## 分析概述 (Analysis Overview)

本项目侧重于从银行营销数据集中从三个核心维度挖掘可操作的洞察：

1.  **模型训练 (Model Training - CatBoost)**
    *   **目的**：构建一个稳健的预测模型，准确识别可能订阅定期存款的客户。
    *   **目标**：通过定位高概率潜在客户来优化营销效率。

2.  **可解释性 (Explainability - SHAP)**
    *   **目的**：打开模型的“黑盒”，理解模型*为什么*做出特定的预测。
    *   **目标**：揭示客户行为的关键驱动因素（例如：通话时长、账户余额、联系次数等），并验证模型逻辑是否符合领域知识。

3.  **反事实分析 (Counterfactual Analysis - DiCE)**
    *   **目的**：探索“如果特征值改变，预测结果会如何变化” (what-if) 的情景，生成多样化的反事实解释。
    *   **目标**：为销售代表提供可操作的建议（例如：*“如果我们在5月而不是10月联系这位客户，他可能会订阅”*）。

## Notebook 概览 (Notebook Overview)

分析内容包含在 [catboost_analysis.ipynb](catboost_analysis.ipynb) 中，涵盖以下关键阶段：

### 1. 数据加载与预处理 (Data Loading & Preprocessing)
- 加载数据集，手动处理 ARFF 的表头。
- 清洗分类字符串列（去除引号）。
- 将目标变量 (`y`) 编码为二进制格式 (0/1)。
- 自动识别分类特征，供 CatBoost 原生处理。

### 2. 模型训练 (Model Training)
- **模型**：`CatBoostClassifier`
- **参数**：
  - `iterations`: 500
  - `learning_rate`: 0.1
  - `depth`: 6
  - `loss_function`: Logloss
  - `eval_metric`: AUC
- **验证**：使用 20% 的分层测试集来监控性能，并通过 Early Stopping 防止过拟合。

### 3. 评估 (Evaluation)
模型取得了优异的性能指标：
- **准确率 (Accuracy)**: ~91.1%
- **ROC AUC**: ~0.936
- **指标**：详细的分类报告 (Classification Report) 和混淆矩阵 (Confusion Matrix)。
- **阈值调优**：包含寻找最佳分类阈值的逻辑（例如：最大化 F1-score）。

### 4. 可解释性与说明性 (Interpretability & Explainability)
本 Notebook 的一个重点是理解预测背后的*原因*：
- **特征重要性 (Feature Importance)**：可视化 CatBoost 默认的特征重要性评分。
- **SHAP (SHapley Additive exPlanations)**：
  - **全局重要性**：使用蜂群图 (Beeswarm plot) 展示特征如何影响整个数据集的模型输出。
  - **局部解释**：使用瀑布图 (Waterfall plots) 解释单个预测（例如：为什么某个特定客户被归类为低风险）。
- **反事实分析 (Counterfactual Analysis - DiCE)**：
  - 使用 **DiCE (Diverse Counterfactual Explanations)** 生成“假设性”场景。
  - 生成可操作的改变建议（例如：改变联系方式或月份），从而将预测结果从“否”变为“是”。

## 依赖 (Requirements)
- `catboost`
- `shap`
- `dice-ml`
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`
