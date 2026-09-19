# 如何把仓库发布出去（3 分钟）

## 方式 A：GitHub（推荐，评审最方便）

1. 打开 https://github.com/new ，仓库名建议 `jstars-lightweight-aerial-detection`，**不要**勾选 "Add a README"。
2. 在本目录打开 PowerShell，执行（把 `<你的用户名>` 换成你的 GitHub 用户名）：

```powershell
cd "D:\Research\02_Vault\30_Projects\轻量小样本目标检测\repo_jstars"
git remote add origin https://github.com/<你的用户名>/jstars-lightweight-aerial-detection.git
git branch -M main
git push -u origin main
```

3. 推送成功后，仓库公开地址就是：
   `https://github.com/<你的用户名>/jstars-lightweight-aerial-detection`

## 方式 B：Zenodo（可拿 DOI，适合"数据可用性声明"）

1. 把本目录压缩为 zip。
2. 打开 https://zenodo.org/ ，用 GitHub 或邮箱登录 → New upload → 上传 zip → 填写标题/作者 → Publish。
3. Zenodo 会给出一个 DOI（形如 `10.5281/zenodo.xxxxxxx`），在论文里引用这个 DOI 更正式。

## 完成后

把最终 URL（或 DOI）告诉我，我会写进稿件的 Data and Code Availability 声明里。
