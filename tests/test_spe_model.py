#!/usr/bin/env python3
"""SPE 模型单元测试"""
import numpy as np
from sklearn.datasets import make_classification
from src.models.spe import SelfPacedEnsemble, SPEAdapter
from src.core.contracts import TrainingData
from src.models.registry import MODEL_REGISTRY
import pandas as pd

# 1. 验证适配器注册
adapter = MODEL_REGISTRY.create('spe')
assert adapter.name == 'spe'
assert adapter.artifact_filename == 'model_spe.pkl'
assert adapter.supports_constraints == False
print('✓ 适配器注册成功')

# 2. 验证 build() 返回正确的实例
model = adapter.build({'n_estimators': 3, 'k_bins': 3, 'base_estimator': 'decision_tree'}, 42)
assert hasattr(model, 'fit')
assert hasattr(model, 'predict')
assert hasattr(model, 'predict_proba')
assert model.n_estimators == 3
assert model.k_bins == 3
print('✓ build() 返回正确的 SelfPacedEnsemble 实例')

# 3. 验证配置校验
adapter.validate_config({'n_estimators': 5, 'k_bins': 3})
print('✓ validate_config() 通过')

try:
    adapter.validate_config({'n_estimators': 0, 'k_bins': 3})
    assert False, '应抛出异常'
except ValueError:
    print('✓ validate_config() 正确拒绝无效参数')

# 4. 验证模型训练和预测
X, y = make_classification(n_samples=200, n_features=10, weights=[0.9, 0.1], random_state=42)
model = SelfPacedEnsemble(n_estimators=5, k_bins=3, random_state=42)
model.fit(X, y)
pred = model.predict(X)
proba = model.predict_proba(X)
assert pred.shape == (200,)
assert proba.shape == (200, 2)
assert np.all((proba >= 0) & (proba <= 1))
print(f'✓ 模型训练和预测成功 (accuracy={np.mean(pred==y):.3f})')

# 5. 验证 sample_weight 透传
sw = np.ones(200)
model2 = SelfPacedEnsemble(n_estimators=3, k_bins=3, random_state=42)
model2.fit(X, y, sample_weight=sw)
print('✓ sample_weight 透传正常')

# 6. 验证 get_params/set_params (BayesSearchCV 兼容性)
# 嵌套参数仅在存在真实基分类器时可见（与 sklearn 约定一致）
# 使用适配器构建的模型（含 DecisionTreeClassifier 基分类器）来验证
tuned = adapter.build(
    {'n_estimators': 3, 'k_bins': 3, 'base_estimator': 'decision_tree',
     'base_estimator__max_depth': 7}, 42)
params = tuned.get_params(deep=True)
assert 'n_estimators' in params
assert 'k_bins' in params
assert 'base_estimator__max_depth' in params
assert params['base_estimator__max_depth'] == 7
tuned.set_params(base_estimator__max_depth=10)
assert tuned.base_estimator.max_depth == 10
print('✓ get_params/set_params 嵌套参数支持正常')

# 7. 验证 fit_params 返回正确格式
data = TrainingData(
    features=pd.DataFrame(X),
    labels=pd.Series(y),
    sample_weight=pd.Series(np.ones(200))
)
fit_kwargs = adapter.fit_params(data)
assert 'sample_weight' in fit_kwargs
print('✓ fit_params() 返回正确格式')

print()
print('所有测试通过!')