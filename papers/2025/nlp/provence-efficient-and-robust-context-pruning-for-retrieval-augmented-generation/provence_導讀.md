# Provence：用於檢索增強生成的高效且穩健的上下文修剪

## 1. 論文基本資訊

- **標題**：
  - 中文：Provence：用於檢索增強生成的高效且穩健的上下文修剪
  - 英文：*Provence: efficient and robust context pruning for retrieval-augmented generation*
- **作者**：Nadezhda Chirkova, Thibault Formal, Vassilina Nikoulina, Stéphane Clinchant
- **發表機構**：NAVER LABS Europe, Grenoble, France
- **年份**：2024年

---

## 2. 研究背景

### **解決什麼問題**

- 檢索增強生成（RAG）存在**長上下文造成的計算開銷**問題，並可能傳播**不相關的檢索資訊**。
- 現有的上下文修剪方法缺乏一個**既高效又穩健的通用模型**，以應對多種情境，例如：
  - 上下文包含不同數量的相關資訊
  - 上下文長度不一
  - 在不同領域進行評估
- 部分方法**效率不足**，例如使用龐大的LLM作為基礎模型，或需要耗時的自迴歸生成。
- 大多數現有工作針對單一數據集訓練修剪器，缺乏跨多個領域的**穩健性**或**可轉移性**。

### **為什麼重要**

- 上下文修剪能透過移除不相關內容，**減少上下文長度**，從而**加快LLM的生成速度**。
- 它還有助於**避免生成不真實的資訊**，並提供參考資料。
- 修剪模組可作為**隨插即用**（plug-and-play）元件與任何生成器LLM結合，提升RAG管線的透明度。
- 透過存取領域特定的數據庫，可讓LLM無需微調即可推論先前未知的知識。

---

## 3. 主要貢獻

### **創新點**

1.  引入 **Provence (Pruning and Reranking Of retrieVEd relevaNt ContExts)**，一個用於問答的**高效且穩健的上下文修剪器**。
2.  Provence 能夠**動態地檢測**給定上下文所需的修剪量，並**開箱即用**於各種領域。
3.  將上下文修剪任務**重新定義為序列標記**（sequence labeling）問題。
4.  **統一了上下文修剪與上下文重排序**（reranking）的能力，將這兩個功能結合在單一模型中。
5.  透過在**多樣化數據**上訓練，提升了模型的穩健性。

### **技術特色**

- 採用**查詢依賴的句子級修剪**（query-dependent sentence-level pruning），移除與查詢不相關的語義單元（句子）。
- 以**提取式**（extractive）方式進行修剪，選擇相關文本而非生成新摘要。
- **高效性**：基於輕量級的**DeBERTa模型**微調，比基於大型LLM的方法更高效。
- 當與重排序功能統一時，**在標準RAG管線中幾乎沒有額外成本**。
- 能夠偵測相關句子的數量及其在上下文中的任何位置。
- 對不同上下文長度表現出**穩健性**。

---

## 4. 方法概述

### **核心方法**

- **問題建模**：將上下文修剪視為一個**序列標記任務**。模型輸入查詢-上下文對，輸出**逐詞元（per-token）的二元標籤**，指示每個詞元是否應被保留。
- **模型架構**：使用 **DeBERTa模型**進行微調。統一模型從預訓練的**跨編碼器重排序器**開始。
- **與重排序器的統一**：
  - 將重排序和上下文修剪合併為一個模型，具備**兩個不同的任務頭**。
  - 重排序頭輸出BOS（Begin of Sequence）詞元的相關性分數。
  - 修剪頭輸出上下文詞元的逐詞元預測。
  - 這種統一使得**上下文修剪的計算開銷幾乎為零**。
- **訓練數據**：使用來自MS MARCO和Natural Questions的數據。MS MARCO文檔被分割成包含N個連續句子的段落（N為1到10之間的隨機整數），以確保對可變上下文長度的穩健性。
- **「銀標籤」（Silver Labels）生成**：
  - 透過提示 **Llama-3-8B-Instruct** LLM來生成訓練標籤。
  - 該LLM被指示僅使用提供的上下文回答問題，並引用所有相關句子。
- **推理**：
  - 將查詢和檢索到的段落串聯送入 Provence。
  - 模型輸出每個詞元被包含在最終上下文中的概率。
  - 透過設定閾值 `T` 來二值化詞元概率。
  - 應用「**句子四捨五入**」（sentence rounding）程序：如果一個句子中超過50%的詞元被保留，則選擇整個句子。

### **架構圖解**

- **推論 (Inference)**：
  1.  使用者問題經由 `Retriever` 和 `Reranker` 處理，得到 `Retrieved context`。
  2.  `Retrieved context` 進入 `Provence` 進行 `Pruned context`。
  3.  `Pruned context` 傳給 `Generator LLM` 以產生 `Generated response`。

- **訓練 (Training)**：
  1.  訓練問題和 `Datastore` 經 `Retriever` 取得 `Retrieved context`。
  2.  `Data labeler LLM` 負責生成 `Target`（包括二元掩碼和重排序分數）。
  3.  這些目標數據用於訓練 `Provence` 模型。

---

## 5. 實驗結果

### **主要發現**

- **性能領先**：Provence 在相似的壓縮率下，於所有測試數據集上均實現了**最高的問答性能**。
- **高效與經濟**：Provence 的效率優於需要更多計算資源的方法（如LLMLingua），並且在與重排序器統一後，**幾乎沒有額外的計算成本**。
- **廣泛適用性**：Provence 能夠在多個問答領域以可忽略或零性能下降的情況下修剪上下文。在某些數據集上，修剪甚至因為**過濾了噪音而帶來性能提升**。
- **動態適應性**：Provence 能靈活地偵測相關資訊的數量及其在上下文中的任何位置。
- **對相關資訊位置的穩健性**：在「大海撈針」（needle-in-the-haystack）實驗中，Provence 能正確選擇「針」句子。
- **對上下文粒度的穩健性**：對於不同粒度的上下文（2、6、10個句子或100個詞），Provence 都能保持高性能。
- **保留重排序能力**：共同訓練能夠有效地**保留模型初始的重排序能力**，性能與基準重排序器相當。
- **超參數的通用性**：Provence 的修剪閾值 `T`（例如0.1和0.5）在不同數據集之間普遍適用，無需大量調整。

### **性能表現**

- **LLM-Eval 性能**：Provence 始終位於帕累托前沿（Pareto front），即在實現高壓縮率的同時保持高質量。
- **壓縮率**：依數據集和閾值 `T` 的不同，自動在 50% 到 80% 之間變動。
- **生成速度提升**：對於大型批次（batch sizes），觀察到高達 **2倍** 的生成速度提升。
- **計算效率**：相比LongLLMLingua和LLMLingua2，Provence 需要的MFLOPS顯著較低。

---

## 6. 個人心得

### **閱讀感想**

> RAG 在處理長上下文和不相關資訊時面臨的挑戰是現實且重要的問題，上下文修剪作為解決方案顯得合乎邏輯且必要。Provence 將修剪與重排序統一的創新方法非常巧妙，不僅有效解決了問題，還將修剪的計算成本降至幾乎為零，顯著提升了RAG管線的整體效率。該研究透過將上下文修剪建模為序列標記任務，並使用輕量級的DeBERTa模型進行訓練，有效克服了先前研究中效率不足和通用性差的限制。使用Llama-3-8B-Instruct生成「銀標籤」的策略，為獲取高質量訓練數據提供了實用的方法。

### **啟發與思考**

- 這項工作強調了在設計AI模型時，不僅要關注其效果（*effectiveness*），更要兼顧其**效率（efficiency）和穩健性（robustness）**，以便在實際應用中實現真正的「隨插即用」。
- 統一相關任務（如修剪和重排序）以優化整個機器學習管線的思路，為未來其他複雜系統的設計提供了寶貴的啟示。
- 消融實驗（*ablation study*）的結果，特別是關於訓練數據多樣性和提示工程對標籤生成的影響，為未來開發穩健的上下文修剪器提供了實用的指導。
- 「大海撈針」測試和對不同上下文長度的穩健性評估，為衡量實際修剪器性能提供了有力的基準。

### **與其他研究的關聯**

- Provence 直接改進了**檢索增強生成（RAG）**範式，使其更高效和可靠。
- 它與現有的多種上下文修剪方法（如RECOMP、FilCo、COMPACT、DSLR、LLMLingua系列等）進行了詳細比較，旨在解決它們的局限性。
- 雖然存在透過改變LLM架構或增加LLM訓練數據多樣性來處理長上下文的替代方案，Provence 則強調了上下文修剪作為一種**更靈活且易於部署的即插即用解決方案**。
- Provence 還與標準的**跨編碼器重排序器**（如DeBERTa-v3）進行了整合，進一步優化了整個檢索管線。

---

## 7. 相關資源

- **論文連結**：雖然本文為預印本或會議投稿（"*Paper under double-blind review*"），但其模型已在Hugging Face上發布。
  - Hugging Face模型頁面：[https://huggingface.co/naver/provence-reranker-debertav3-v1](https://huggingface.co/naver/provence-reranker-debertav3-v1)
- **程式碼**：通常可在模型發布頁面找到相關程式碼或使用範例。
- **相關論文**：
  - *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (Lewis et al., 2020)
  - *LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models* (Jiang et al., 2023)
  - *RECOMP: Improving retrieval-augmented LMs with context compression and selective augmentation* (Xu et al., 2024)