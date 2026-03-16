# GitHub 上传指南

本指南帮助您将整理好的项目上传到 GitHub。

---

## 方法一：使用 Git 命令行

### 1. 准备工作

确保已安装：
- [Git](https://git-scm.com/downloads) (Windows/Linux/Mac)
- [Git LFS](https://git-lfs.github.com/) (用于大文件)

### 2. 创建 GitHub 仓库

1. 访问 https://github.com/new
2. 填写仓库信息：
   - **Repository name**: `ultra-lsnt` (或其他)
   - **Description**: Physics-Informed Mixture-of-Experts for Ultra-Lightweight Spatio-Temporal Forecasting
   - **Public/Private**: 选择公开或私有
   - **Initialize**: 不要勾选任何初始化选项
3. 点击 **Create repository**

### 3. 本地初始化

打开终端/PowerShell：

```bash
# 进入项目目录
cd f:\Ultra-LSNT-Paper

# 初始化Git仓库
git init

# 配置Git (如果尚未配置)
git config user.name "Your Name"
git config user.email "your.email@example.com"

# 添加远程仓库地址
git remote add origin https://github.com/yourusername/ultra-lsnt.git
```

### 4. 配置 Git LFS (推荐)

由于数据文件较大，使用 Git LFS：

```bash
# 初始化Git LFS
git lfs install

# 追踪大文件
git lfs track "data/processed/*.csv"
git lfs track "*.pkl"
git lfs track "*.h5"
git lfs track "*.pth"
git lfs track "checkpoints/*"

# 添加追踪配置
git add .gitattributes
git commit -m "Configure Git LFS for large files"
```

### 5. 添加文件并提交

```bash
# 添加所有文件
git add .

# 查看状态
git status

# 提交
git commit -m "Initial commit: Ultra-LSNT paper code and experiments

- Core model implementation (Ultra-LSNT v4.0)
- 14 baseline models (DLinear, PatchTST, etc.)
- 30+ experiment scripts
- 4 datasets (Wind CN/US, Air Quality, GEFCom)
- Complete experimental results
- Paper LaTeX source and PDF"
```

### 6. 推送到 GitHub

```bash
# 推送到main分支
git branch -M main
git push -u origin main
```

---

## 方法二：使用 GitHub Desktop (推荐新手)

### 1. 下载安装

- 下载 [GitHub Desktop](https://desktop.github.com/)
- 安装并登录GitHub账号

### 2. 添加本地仓库

1. 打开 GitHub Desktop
2. 点击 **File** → **Add local repository**
3. 选择 `f:\Ultra-LSNT-Paper` 文件夹
4. 点击 **Add Repository**

### 3. 发布到 GitHub

1. 在 GitHub Desktop 中，点击 **Publish repository**
2. 填写仓库名称: `ultra-lsnt`
3. 添加描述
4. 选择 **Public** 或 **Private**
5. 点击 **Publish Repository**

---

## 方法三：使用 VS Code

### 1. 打开项目

1. 在 VS Code 中打开 `f:\Ultra-LSNT-Paper` 文件夹
2. 安装 Git 扩展（如果未安装）

### 2. 初始化仓库

1. 点击左侧活动栏的 **源代码管理** 图标
2. 点击 **初始化仓库**

### 3. 提交和推送

1. 在源代码管理面板中，输入提交消息
2. 点击 **提交**
3. 点击 **发布到 GitHub**
4. 登录GitHub账号并选择仓库名称
5. 点击 **发布**

---

## 大文件处理

如果数据文件超过 GitHub 限制 (100MB)：

### 选项1：使用 Git LFS (推荐)

已包含在 `.gitattributes` 中，Git LFS 会自动处理。

### 选项2：压缩数据文件

```bash
# 压缩数据
cd data/processed
tar -czvf data_archive.tar.gz *.csv

# 上传到GitHub Release (见下文)
```

### 选项3：使用外部存储

将大文件上传到云存储，在README中提供链接：
- [Google Drive](https://drive.google.com)
- [OneDrive](https://onedrive.live.com)
- [Dropbox](https://dropbox.com)

---

## 上传后配置

### 1. 添加仓库描述

在GitHub网页上：
1. 进入仓库页面
2. 点击右侧 **About** 旁边的齿轮
3. 添加描述、网址、主题标签

### 2. 启用 Issues

设置 → 勾选 **Issues**

### 3. 添加 Topics (主题标签)

建议添加：
- `deep-learning`
- `time-series`
- `mixture-of-experts`
- `wind-power-forecasting`
- `pytorch`
- `edge-computing`

### 4. 设置 README 显示

确保 `README.md` 正确显示在仓库首页。

---

## 创建 Release (可选)

为项目创建发布版本：

1. 在GitHub仓库页面，点击右侧 **Releases**
2. 点击 **Create a new release**
3. 填写版本号: `v1.0.0`
4. 添加发布说明
5. 上传数据文件附件（如果超过100MB）
6. 点击 **Publish release**

---

## 验证上传

### 检查清单

- [ ] 所有代码文件已上传
- [ ] README正确显示
- [ ] 数据集可访问
- [ ] 图表可查看
- [ ] 许可证文件存在
- [ ] 依赖文件存在

### 克隆测试

```bash
# 在新位置克隆测试
cd /tmp
git clone https://github.com/yourusername/ultra-lsnt.git
cd ultra-lsnt

# 检查文件完整性
ls -la
ls src/models/
ls data/processed/
```

---

## 常见问题

### Q: 推送失败，提示文件太大？

**A**: 使用 Git LFS 或从Git历史中移除大文件：

```bash
# 移除大文件
git rm --cached data/processed/large_file.csv
git commit -m "Remove large file"

# 添加到.gitignore
echo "data/processed/large_file.csv" >> .gitignore
```

### Q: 如何更新已上传的仓库？

**A**:

```bash
# 添加修改
git add .
git commit -m "Update description"
git push
```

### Q: 如何添加协作者？

**A**: 在GitHub仓库页面 → Settings → Manage access → Invite a collaborator

---

## 下一步

上传后，您可以：

1. **分享仓库链接** 给他人
2. **撰写教程** 帮助用户复现实验
3. **响应 Issues** 解决用户问题
4. **持续更新** 改进代码和文档
5. **添加 CI/CD** 自动化测试

---

**恭喜！** 您的论文代码已成功上传到 GitHub！ 🎉
