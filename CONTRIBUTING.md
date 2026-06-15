# 贡献指南

感谢你对 MedVision 项目的关注！

## 如何贡献

### 报告 Bug
1. 在 GitHub Issues 中创建新 Issue
2. 描述问题现象、复现步骤、环境信息
3. 附上错误日志或截图

### 提交代码
1. Fork 本仓库
2. 创建功能分支：`git checkout -b feature/your-feature`
3. 编写代码 + 测试
4. 提交：`git commit -m "feat: 描述"`
5. 推送：`git push origin feature/your-feature`
6. 创建 Pull Request

### 代码规范
- Python: PEP 8 风格
- 每个函数需要 docstring
- 关键参数需要类型标注
- 新功能需要配套测试

### 提交信息格式
```
feat: 新功能
fix: 修复bug
docs: 文档更新
test: 测试相关
refactor: 代码重构
```

### 开发环境
```bash
pip install -r requirements.txt
pip install pytest  # 测试框架
python3 -m pytest tests/  # 运行测试
```

## 问题反馈

如有问题，请在 GitHub Issues 中反馈，包含：
- 问题描述
- 复现步骤
- 系统环境 (OS, Python version, hardware)
- 错误日志
