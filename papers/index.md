# 論文索引

本索引記錄所有已閱讀的 NLP 相關論文，按年份和領域分類。

## 2025 年

### Natural Language Processing (NLP)
- [Provence: 用於檢索增強生成的高效且強健的上下文剪枝技術](2025/nlp/provence-efficient-and-robust-context-pruning-for-retrieval-augmented-generation/provence_zh.md) - RAG 系統的上下文裁剪器
  - 英文版本: [English](2025/nlp/provence-efficient-and-robust-context-pruning-for-retrieval-augmented-generation/provence_en.md)
  - 導讀: [中文導讀](../guides/2025/provence_guide.md)
- [CoALA: 語言代理的認知架構](2025/nlp/cognitive-architectures-for-language-agents/coala_zh.md) - 語言代理的統一概念框架
  - 英文版本: [English](2025/nlp/cognitive-architectures-for-language-agents/coala_en.md)
  - 導讀: [中文導讀](../guides/2025/coala_guide.md)
- [Reflexion: 帶有語言強化學習的語言代理](2025/nlp/reflexion-language-agents-with-verbal-reinforcement-learning/reflexion_zh.md) - 透過口頭反思進行強化學習的創新框架
  - 英文版本: [English](2025/nlp/reflexion-language-agents-with-verbal-reinforcement-learning/reflexion_en.md)
  - 導讀: [中文導讀](../guides/2025/reflexion_guide.md)
  - 總結筆記: [Summary](2025/nlp/reflexion-language-agents-with-verbal-reinforcement-learning/reflexion_summary.md)
- [遞迴式摘要功能使大型語言模型具備長期對話記憶](2025/nlp/recursively-summarizing-enables-long-term-dialogue-memory-in-large-language-models/recursive-memory_zh.md) - 透過遞迴摘要增強LLM長期對話能力
  - 英文版本: [English](2025/nlp/recursively-summarizing-enables-long-term-dialogue-memory-in-large-language-models/recursive-memory_en.md)
  - 導讀: [中文導讀](../guides/2025/recursive-memory_guide.md)
- [LEANN：低儲存向量索引](2025/nlp/leann-low-storage-vector-index/leann_zh.md) - 針對資源受限的個人裝置優化的儲存高效近似最近鄰（ANN）搜尋索引
  - 英文版本: [English](2025/nlp/leann-low-storage-vector-index/leann_en.md)
  - 導讀: [中文導讀](../guides/2025/leann_guide.md)
- [LAG: 邏輯增強生成](2025/nlp/lag-logic-augmented-generation-from-a-cartesian-perspective/lag_zh.md) - 從笛卡爾視角看知識增強的新範式
  - 英文版本: [English](2025/nlp/lag-logic-augmented-generation-from-a-cartesian-perspective/lag_en.md)
  - 導讀: [中文導讀](../guides/2025/lag_guide.md)

### Others
<!-- 在此添加其他領域相關論文 -->

---

## 如何添加新論文

每篇論文包含以下檔案：
1. **英文筆記**: `論文名稱_en.md`
2. **中文筆記**: `論文名稱_zh.md` 
3. **中文導讀**: 存放於 `guides/` 資料夾
4. **讀書心得**: 可包含在導讀中或單獨建檔

請按照以下格式添加索引項目：
```markdown
- [論文標題](相對路徑/論文名稱_zh.md) - 簡短描述
  - 英文版本: [English](相對路徑/論文名稱_en.md)
  - 導讀: [中文導讀](../guides/年份/論文名稱_guide.md)
```