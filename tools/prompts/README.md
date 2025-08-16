# 🤖 Prompt 模板庫

這個資料夾包含了所有用於 NotebookLM 輔助學習的 Prompt 模板，按功能分類組織。

## 📁 資料夾結構

```
prompts/
├── README.md                    # 本說明文件
├── paper-analysis/              # 論文分析相關 Prompts
│   ├── structure_analysis.md    # 分析論文結構
│   ├── contribution_extraction.md # 提取主要貢獻
│   ├── methodology_breakdown.md # 方法論分解
│   └── experiment_analysis.md   # 實驗分析
├── note-generation/             # 筆記生成相關 Prompts
│   ├── english_notes.md         # 英文筆記生成
│   ├── chinese_notes.md         # 中文筆記生成
│   ├── summary_generation.md    # 摘要生成
│   └── guide_writing.md         # 導讀撰寫
├── understanding/               # 理解深化相關 Prompts
│   ├── concept_clarification.md # 概念釐清
│   ├── connection_finding.md    # 關聯性發現
│   ├── critical_thinking.md     # 批判性思考
│   └── application_exploration.md # 應用探索
├── workflow/                    # 工作流程相關 Prompts
│   ├── quality_check.md         # 品質檢查
│   ├── completeness_review.md   # 完整性檢視
│   └── final_review.md          # 最終審核
└── markdown-helper/             # Markdown 助手相關 Prompts
    ├── format_optimization.md   # 格式優化
    ├── toc_generation.md        # 目錄生成
    ├── document_beautification.md # 文檔美化
    └── translation_formatting.md # 翻譯格式化
```

## 🎯 使用指南

### 基本使用步驟
1. **選擇適當分類** - 根據當前需求選擇對應資料夾
2. **選擇具體 Prompt** - 在分類中找到最符合需求的模板
3. **客製化調整** - 根據具體論文特點調整 Prompt 內容
4. **NotebookLM 應用** - 將調整後的 Prompt 複製到 NotebookLM
5. **結果優化** - 根據輸出品質持續優化 Prompt

### Prompt 命名規範
- 使用小寫字母和底線
- 名稱應清楚說明 Prompt 的用途
- 檔案擴展名統一使用 `.md`

### 版本管理
- 重大修改時保留舊版本作為備份
- 在檔案頭部記錄版本資訊和更新日期
- 說明修改原因和預期改善效果

## 📝 Prompt 撰寫原則

### 結構化設計
1. **明確目標** - 清楚說明期望的輸出內容
2. **具體指令** - 提供明確的操作步驟
3. **格式要求** - 指定輸出的格式和結構
4. **品質標準** - 說明期望的品質水準

### 適應性考量
- **論文類型** - 考慮不同類型論文的特殊需求
- **複雜程度** - 根據論文複雜度調整 Prompt 深度
- **個人偏好** - 保留個人化調整的空間

### 效果優化
- **迭代改進** - 根據使用經驗持續優化
- **A/B 測試** - 比較不同版本的效果
- **效果記錄** - 記錄使用效果和改進建議

## 🔧 維護指南

### 定期檢視
- 每月檢視一次 Prompt 的使用效果
- 根據 NotebookLM 更新調整 Prompt 策略
- 收集使用回饋並進行改善

### 新增 Prompt
- 遵循現有的分類體系
- 確保與現有 Prompt 不重複
- 提供清楚的使用說明

### 品質控制
- 新增的 Prompt 需經過實際測試
- 確保輸出品質符合標準
- 定期清理不再使用的 Prompt

---

*透過這些精心設計的 Prompt 模板，我們可以更有效地利用 NotebookLM 進行 NLP 論文的深度學習。*