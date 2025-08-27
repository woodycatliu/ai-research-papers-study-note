# Multi-view Content-aware Indexing for Long Document Retrieval

Kuicai Dong†, Derrick Goh Xin Deik†, Yi Quan Lee, Hao Zhang, Xiangyang Li, Cong Zhang, Yong Liu  Huawei Noah's Ark Lab  {dong.kuicai; goh.xin.deik; liu.yong6}@huawei.com

Abstract

Long document question answering (DocQA) aims to answer questions from long documents over 10k words. They usually contain content structures such as sections, sub- sections, and paragraph demarcations. However, the indexing methods of long documents remain under- explored, while existing systems generally employ fixed- length chunking. As they do not consider content structures, the resultant chunks can exclude vital information or include irrelevant content. Motivated by this, we propose the Multi- view Content- aware indexing (MC- indexing) for more effective long DocQA via (i) segment structured document into content chunks, and (ii) represent each content chunk in raw- text, keywords, and summary views. We highlight that MC- indexing requires neither training nor fine- tuning. Having plug- and- play capability, it can be seamlessly integrated with any retrievers to boost their performance. Besides, we propose a long DocQA dataset that includes not only question- answer pair, but also document structure and answer scope. When compared to state- of- art chunking schemes, MC- indexing has significantly increased the recall by  \(42.8\%\) \(30.0\%\) \(23.9\%\)  ,and  \(16.3\%\)  via top  \(k = 1.5\)  3,5,and 10 respectively. These improved scores are the average of 8 widely used retrievers (2 sparse and 6 dense) via extensive experiments.

## 1 Introduction

Document question answering (DocQA) is a pivotal task in natural language processing (NLP) that involves responding to questions using textual documents as the reference answer scope. Conventional DocQA systems comprise three key components: (i) an indexer that segments the document into manageable text chunks indexed with embeddings, (ii) a retriever that identifies and fetches the most relevant chunks to the corresponding

Question (a): HOW TO BAKE A CHOCOLATE CAKE? Desired Reference Text: You can bake a chocolate cake by following procedures: 1. Preparation: ... 2. Gather Ingredients: ... 3. Dry Ingredients Mixture: ... 4. Wet Ingredients Mixture: ... 5. Combine Mixtures: ... 6. Bake the Cake: ... (500 words) Actual Chunks Retrieved: ... You can bake a chocolate cake by following procedures: 1. Preparation: ... (100 words)

(a) The whole section (approx. 500 words) is required to answer the question. The retrieved chunk only has 100 words.

Question (b): WHAT IS THE HARDWARE SPECIFICATIONS (CPU, DISPLAY, BATTERY, ETC) OF DELL XPS 13?

Desired Reference Text: ... 14th Gen Intel Core i7 processor ... a 13.4- inch FHD Infinite- Edge display ... battery life ... backlit keyboard ... with Thunderbolt 4 ports ... (250 words)

Actual Chunks Retrieved:

1. ... an 11th Gen Intel Core i7 processor ... 13.4-inch FHD Infinity Edge display in (Content: Dell XPS 13, 100 words) 
2. ... new M1 Pro chip ... 14-inch Liquid Retina XDR display showcases ... (Content: MacBook Pro, 100 words) 
3. ... a powerful Intel Core M processor ... 13.3-inch 4K UHD touch display ... (Content: Dell XPS 12, 100 words)

(b) The whole section (approx. 250 words) is required to answer the given question related to Dell XPS 13. Missing information (e.g, model name) leads to conflicting information.

Figure 1: Bad cases from fixed- length chunking due to relevant text missing and inclusion of irrelevant text.

question, and (iii) a reader that digests the retrieved answer scope and generates an accurate answer. Unlike the retriever: Robertson and Zaragoza, 2009; Karpukhin et al., 2020; Khattab and Zaharia, 2020a) and reader (Nie et al., 2019; Lewis et al., 2020; Izacard and Grave, 2021) that are vastly studied, the indexer received relatively less attention.

Existing indexing schemes overlook the importance of content structures when dealing with long documents, as they are usually organized into chapters, sections, subsections, and paragraphs (Yang et al., 2020; Buchmann et al., 2024), i.e., structured. The widely used fixed- length chunking strategy can easily break the contextual relevance between text chunks for long documents. Such chunking errors can be further aggravated by the retriever and the reader. Moreover, determining the boundary

![](images/2a039ee99138109726e0d5343e8c76576dfc2392da4beb5ca091c51e862ff25d.jpg)  
Figure 2: Commercial LLM on Span-QA retrieval using full document vs dedicated section.

between chunks can be tricky, requiring delicate design to prevent contextual coherence disruption. Ideally, each chunk should represent a coherent and content- relevant textual span. Otherwise, it can lead to the exclusion of relevant information or the inclusion of irrelevant text, as exemplified in Figure 1. Our empirical study on fixed- length chunking reveals that setting the chunk length to 100 results in over  \(70\%\)  of long answers/supporting evidence being truncated, i.e., incomplete. Such incompleteness still exists at  \(45\%\)  despite an increase of chunk length to 200.  \(^{1}\)

Recently, there is a growing interest in utilizing Large Language Models (LLMs) for QA tasks (Chen et al., 2023; Sarhi et al., 2024). However, the notorious token limit constraint puts a barricade for applying LLM to long documents. For instance, LLaMA (Touvron et al., 2023a), LLaMA 2 (Touvron et al., 2023b), and Mistral (Jiang et al., 2023) are limited to 2k, 4k, and 8k tokens, respectively, which is insufficient for long documents with more than 10k words. Even advanced commercial LLMs struggle to effectively process and understand such long documents. As depicted in Figure 2, our research indicates that the performance of advanced commercial LLM deteriorates substantially when applied to span- based QA on long documents (12k- 30k tokens) as compared to a dedicated section (370 tokens in average).  \(^{2}\)  Furthermore, Liu et al. (2023) indicate that LLMs face difficulties in retaining and referencing information from earlier portions of long documents.

To mitigate the aforementioned challenges, we present a new approach Multi- view Content- aware Indexing, termed MC- indexing, which allows for a more effective retrieval of long documents. Our method involves content- aware chunking of structured long documents, whereby, instead of em ploying naive fixed- length chunking, the document is segmented into section chunks. Each of these section chunks is then indexed in three different views, representing each chunk with raw- text, a list of keywords, and a summary. This multi- view representation significantly enhances the semantic richness of each chunk. For retrieval, we aggregate the top relevant chunks based on multi- views. Note that the entire process of MC- indexing is unsupervised. We leverage on the strength of existing retrievers for the embedding generation of raw- text, keyword, and summary views. We leverage on the answer composition capability of LLM for answer generation, based on retrieved chunks. To our best knowledge, existing DocQA datasets do not provide content structure. Hence, we transform an existing long documents dataset, namely WikiWeb2M (Burns et al., 2023), into a QA dataset, by adding annotations to the documents. In addition, we complement Natural Questions dataset (Kwiatkowski et al., 2019) with content structure, and filter only long documents for our experiment. Distinct from other QA datasets, our documents are longer (averaging at 15k tokens) and contain detailed content structure. Our contributions are in fourfold:

- We propose a long document QA dataset annotated with question-answer pair, document content structure, and scope of answer.- We propose Multi-view Content-aware indexing (MC-indexing), that can (i) segment the long documents according to their content structures, and (ii) represent each chunk in three views, i.e., raw-text, keywords, and summary.- MC-indexing requires neither training nor fine-tuning, and can seamlessly act as a plug-and-play indexer to enhance any existing retrievers.- Through extensive experiments and analysis, we demonstrate that MC-indexing can significantly improve retrieval performance of eight commonly-used retrievers (2 sparse and 6 dense) on two long DocQA datasets.

## 2 Related Work

Retrieval Methods. Current approaches to content retrieval are primarily classified into sparse and dense retrieval. There are two widely- used sparse retrieval methods, namely TF- IDF (Salton et al., 1983) and BM25 (Robertson et al., 1995). TF- IDF calculates the relevance of a word to a document in the corpus by multiplying the word frequency

with the inverse document frequency. BM25 is an advancement of TF- IDF that introduces nonlinear word frequency saturation and length normalization to improve retrieval accuracy.

Recently, dense retrieval methods have shown promising results, by encoding content into highdimensional representations.  DPR (Karpukhin et al., 2020) is the pioneering work of dense vector representations for QA tasks. Similarly, Colbert (Khattab and Zaharia, 2020b) introduces an efficient question- document interaction model, enhancing retrieval accuracy by allowing fine- grained term matching. Contriever (Izacard et al., 2022) further leverages contrastive learning to improve content dense encoding. E5 (Wang et al., 2022) and BGE (Xiao et al., 2023) propose novel training and data preparation techniques to enhance retrieval performance, e.g., consistency- filtering of noisy web data in E5 and the usage of RetroMAE (Xiao et al., 2022) pre- training paradigm in BGE. Moreover, GTE (Li et al., 2023) integrates graph- based techniques to enhance dense embedding.

In summary, these systems focus on how to retrieve relevant chunks, but neglecting how text content is chunked. In contrast, MC- indexing can utilize the strengths of existing retrievers, and further improve their retrieval performance.

Chunking Methods. Chunking is a crucial step in either QA or Retrieval- Augmented Generation (RAG). When dealing with ultra- long text documents, chunk optimization involves breaking the document into smaller chunks. In practice, fixedlength chunking is a commonly used method that is easy to be implemented. It chunks text at a fixed length, e.g., 200 words. Sentence chunking involves dividing textual content based on sentences. Recursive chunking employs various delimiters, such as paragraph separators, newline characters, or spaces, to recursively segment the text. However, these methods often fail to preserve semantic integrity of critical content. In contrast, contentaware chunking (Section 3.1) chunk the text by the smallest subdivision according to the document's content structure. This ensures each chunk to be semantically coherent, thus reducing chunking error.

Long Document Retrieval. Traditional retrieval methods such as BM25 and DPR only retrieve short consecutive chunks from the retrieval corpus, limiting the overall understanding of the context of long documents. To overcome this drawback, several methods focusing on long document retrieval have been proposed. Nie et al. (2022) propose a compressive graph selector network to select questionrelated chunks from the long document and then use the selected short chunks for answer generation. AttenWalker (Nie et al., 2023) addresses the task of incorporating long- range information by employing a meticulously crafted answer generator. Chen et al. (2023) convert the long document into a tree of summary nodes. Upon receiving a question, LLM navigates this tree to find relevant summaries until sufficient information is gathered. Sarthi et al. (2024) utilize recursive embedding, clustering, and summarizing chunks of text to build a tree with different levels of summarization. However, existing methods only consider the retrieval of long documents from one view, limiting the semantic completeness and coherence.

## 3 Methodology

### 3.1 Content-aware Chunking

We elaborate how Content- Aware chunking is performed in order to obtain section chunks. Given a piece of structured document (e.g., Markdown, Latex, Wikipedia Articles), we first extract the table of contents of the document (or header information, in the event where the table of content is not readily available). Upon acquiring this information, we identify the smallest division in the document, such as a section, subsection, or sub- subsection, depending on the structure of the content. It is reasonable to assume that these smallest divisions function as atomic, coherent semantic units within the document. The text present in each smallest division is the desired section chunk.

Chunking text based on the smallest division, as opposed to fixed length chunking, ensures that information in each chunk cannot contain information across two different sections. Most importantly, we preserve the semantic integrity during the chunking process, leading to each section chunk to be an atomic and coherent semantic unit. Note that different sections may have a hierarchical relationship between them. We ignore them for now and assume a flat structure between different chunks.

### 3.2 Multi-View Indexing and Retrieval

Most dense retrieval methods primarily uses raw text from each chunk to determine the relevancy of each chunk with respect to a given question. It works well when using fixed- length chunk to ensure segments do not exceed the dense retrieval

![](images/50383d65733774cc84bbb168d246e4d829f602f0bfd2d66f5d11eb7b09eeedae.jpg)  
Figure 3: Comparison between conventional fixed length chunking and content-aware LLM QA systems.

models' token limits. However, our content- aware chunking approach, which divides documents into their smallest meaningful divisions, often results in chunks larger than the models' maximum token capacity. As indicated in Table 1, Wiki- NQ dataset has averaging 500 tokens per section. This can lead to truncation and information loss during retrieval, undermining the effectiveness of the method.

To mitigate the aforementioned issue, it is necessary to introduce more concise representations or views of section chunks to improve retrieval efficiency. Beyond the raw text view, which consists solely of the section's text, we also introduce the summary view and the keyword view.

The summary view, as the name suggests, represents each section chunk with a succinct summary. It captures the key information of each section. The summary is generated by generative LLM using the prompt given in Figure 8. This ensures that the summary fits within the dense retrieval model's maximum input limit during retrieval.

To compensate for the potential omission of critical details in the generated summaries, we introduce a keyword view. This view characterizes each section chunk by a list of essential keywords, including significant concepts, entities, and terms from the section. We employ a prompt outlined in Figure 9, enabling generative LLM to identify and list the keywords associated with each section.

Finally, we describe the procedure for utilizing multi- view indexing to retrieve top-  \(k\)  relevant sections with respect to a given question. For each of the views, e.g., raw- text, summary, keywords, we simply rank the sections using each view to first retrieve the top-  \(k'\)  results. Setting  \(k' \approx 2k / 3\)  works since empirically we expect on average a total of  \(3k' / 2\)  unique results after deduplication. Thereafter we feed the set of raw text to LLM for answer generation using prompt in Figure 10. Note that MC- indexing is independent of retriever selection. It can be used to augment any retriever by supplying top- ranked chunks across different views. As a plug- and- play boost for all existing retrievers, MC- indexing requires no additional training or fine- tuning to integrate effectively.

## 4 Dataset Construction

In our work, we focus on long and structured document, thus we collect dataset corpus based on the following two factors. (1) Presence of structured information: The content of long documents is usually divided into multiple sections. For example, a research paper is organized into various sections such as Abstract, Introduction, Methodology and Conclusion. Structured documents have explicitly labelled sections along their corresponding text. Most of the existing QA datasets (e.g., SQuAD (Rajpurkar et al., 2016), TriviaQA (Joshi et al., 2017), MS MARCO (Bajaj et al., 2018)) focus on retrieval of relevant passages. Thus the original organizational structure of the source documents are usually omitted. However, for the purpose of MC- indexing, it is necessary to preserve structure information. (2) Sufficiently Long Document: Our method lever

ages using the smallest division as a section chunk. In particular, if a document is too short, a document would be divided into only a few section chunks under Content- aware Chunking. To avoid these trivial cases, we only select documents with at least 10k words, as the number of sections correlates positively with document length.

According to these criteria, we select Wikipedia Webpage 2M (WikiWeb2M) (Burns et al., 2023) and Natural Questions (NQ) (Kwiatkowski et al., 2019) datasets. We discuss dataset processing and annotations on these datasets in finer detail.

### 4.1 Wikipedia Webpage 2M (WikiWeb2M)

WikiWeb2M is designed for multimodal webpage understanding rather than QA. The dataset stores individual sections within each Wikipedia article. Thus, on top of the structured information, we annotate additional question- answer pairs and their answer scope. We utilize advanced commercial LLM to construct questions for selected articles (over 10k tokens) in WikiWeb2M. To ensure that the questions rely on long answer scope span, we define the 8 types of questions. For each section given, we request LLM (using prompt shown in Figure 6) to generate (i) three questions, (ii) the corresponding answers to the each question, and (iii) the answer scope for each answer. We then evaluate the retrieval efficiency and answer quality of MC- indexing by utilizing the constructed data.

Using this approach we have generated questions for 83,625 sections from 3,365 documents. For evaluation, in order to demonstrate the effectiveness of our method in long DocQA, we only use questions generated from documents with 28k to 30k tokens, resulting in 30 documents for evaluation. The remaining questions not used in evaluation are intended for training / fine- tuning, if needed. Note that our proposed method (in Section 3) does not require any form of fine- tuning.

### 4.2 Natural Questions (NQ)

The NQ dataset provides rendered HTML of Wikipedia articles alongside the questions and answer scope. By parsing the rendered HTML, we are able to extract the section name and the corresponding texts in each section of the document. We augment the NQ dataset with our extracted structured information. We omit sections such as 'See Also', 'Notes', and 'References', which refer as references for the main content, to reduce noise during retrieval. We follow NQ's train/dev split setting in our work. However, we only retain the question where its corresponding document has more than 10k tokens. In addition, we filter out queries where the annotated answer scope spans across multiple sections. For dev set, there exists multiple annotations. We only retain questions where all annotations reside within the same section. After filtering, we obtain 36,829 and 586 question- article pairs for train/dev respectively. Again, we emphasise that our approach does not require fine- tuning and solely utilises the dev- set.

![[table_1_document_statistics.png]]

## 5 Experiment

### 5.1 Baseline Systems

When conducting our experiments, we primarily employ the Fixed- Length Chunking (FLC) scheme as our baseline. We firstly segment the document into individual sentences to avoid the first and last sentence in each chunk being truncated. We merge the sentences accordingly into chunks, with approximately 100, 200 or 300 tokens.

We also evaluate on the capability of content awareness in boosting FLC. We first segment the document into section chunks, and further apply FLC on each section. Hence, a section may have multiple chunks but each chunk can only be associated with a section. This is denoted as FLC- content. We also evaluate the variations of our MC- indexing on multi- view as mentioned in Section 3.2. In comparison to raw text, these views offer a different perspective on how to approach and analyze data.

We apply MC- indexing and all baselines on 2 sparse (TF- IDF and BM25) and 6 dense (DPR, Colbert, Contriever, E5, BGE, and GTE) retrievers in Section 2. Implementation details of these retrievers can be found in Appendix A.1.

### 5.2 Evaluation Metrics

We evaluate the performance of MC- indexing and other baselines based on (i) recall of retrieval and (ii) quality of answer generation.

![[table_2_main_results_recall.png]]

Table 2: Main results: recall of ground truth span. Last column is the average scores of all columns.

Recall of Retrieval. The retriever scores each chunk in the document based on its relevance to the question, and returns the top  \(k\)  chunks with the highest scores. We define recall as the proportion of the ground truth answer scope that is successfully retrieved by retriever. For instance, if each of three retrieved chunks overlaps with  \(10\%\) \(50\%\)  and  \(0\%\)  of the ground truth answer scope, the recall is the sum of all individual scores to be 0.6. The recall gives us a clear indication of how effective our chunking strategy has boosted the retriever.

Due to the fact MC- indexing combines the results from three views, we reduce the number of chunks retrieved from each view to have a fair comparison with single- view baselines. Specifically, when comparing with top  \(k = 3\)  single- view base

lines, MC- indexing will only retrieve top  \(k = 1\)  or 2 from each view. By combining the chunks from each view and remove overlapping ones, MC- indexing manages to retrieve an approximate of 3 chunks in total. Similarly for top  \(k = 5\)  our method retrieves only 3 chunks form each view. For top  \(k = 10\)  our method retrieves 6 or 7 chunks from each view. To evaluate the performance of our method in greedy ranking, our method retrieves exactly 1 chunk from each view, while other baselines retrieves 1.5 chunks in average. This is achieved by retrieving 1 chunk for half of the questions and 2 chunks for the other half.

Answer Generation. As the final goal of DocQA is to generate accurate answer, it is essential for us to evaluate the quality of final answer based

![[answer_generaion_chunk_scheme.png]]

on retrieved chunks. We evaluate the answers via pairwise evaluation using advanced commercial LLM as evaluator. Specifically, we provide prompt for LLM (see Figure 11) to score each answer. To avoid any positional bias, which may cause the LLM model to favor the initial displayed answer, we switch answer positions in two evaluation rounds. The winning answer is determined based on scores in two rounds.

For Score- based evaluation, each answer's scores from the two rounds are combined. The answer with higher overall score is the winner. The result is a tie if both answers have same score. For Round- based evaluation, the scores from each round are compared, and the winner of each round is determined by the higher score. The overall winner is the one that wins both rounds. In cases where each answer wins a round, or answers tie in both rounds, the result is marked as a tie.

### 5.3 Main Results

We display our main result in Table 2 and summarise the our analysis with several key observations as follows: (1) The size of chunk significantly impacts the recall. When comparing between different chunk size of FLC, we find out that larger chunks led to an increased recall. We believe that larger chunks are able to retain more information of the answer scope in a single chunk, which lead to better prediction from the retrieval. As shown in Table 2, the improvement from FLC- 100 to FLC- 300 is around  \(10 - 15\%\) . (2) Content- aware chunking further reduces possibility of the ground truth answer scope being split. We define such error as the chunking error (see Section 5.6). As shown in Table 2, given that the chunk size of FLC and FLC- content approximately the same, we observe an improvement of  \(3 - 5\%\)  in FLC- content. (3) Each view of multi- view strategy tends to help retrieval achieves a higher recall than FLC. Among each individual view, utilizing summary view generate the best results, while raw- text view generate the second best results. Despite keywords view down- performs overall due to text having poor semantic structure, we observe that keyword is able to solve some tasks which the other two view unable. This contributes to a positive impact (see Section 5.4). (4) The multi- view strategy, which consolidates top- ranked results of raw- text, keywords, and summary views, can substantially improve the recall of all FLC and single- view baselines. We believe the improvement is mainly contributed by content- aware chunking and multi- view indexing. On the other hand, different views are able to rank the relevance of sections to question from different perspectives, thus providing complimentary information.

### 5.4 Ablation Study

We conducted an in- depth study by ablating each view from our multi- view indexing strategy and measuring the performance by recall. From the results presented in Table 3, we observe that: (1) Removing the summary view leads to the most significant decrease in performance, ranging between 2 and  \(8\%\) . (2) Eliminating the raw- text view results in the second- most considerable performance drop, varying between 2 and  \(5\%\) . (3) Disregarding the keywords view contributes to a decrease of performance ranging from 1 to  \(4\%\) .

Thus, we infer that the impact of each view on the recall performance of retrieval, from the most to the least significant, is as follows: summary view, raw- text view, and keywords view. In conclusion, each view plays a crucial role in improving recall performance. More ablation results on NQ dataset are shown in Appendix A.2.

![](images/4359e9b66ca0c89176ada56d55b15997104c71e0f2a84a8a7379518430df5d8d.jpg)  
Figure 4: The evaluation results of answer generation.

![[table_4_chunking_error_for_each.png]]

### 5.5 Evaluation of Answer Generation

We compare the performance of MC- indexing against FLC- 300 via the relevance of generated answers. For our experiments, we employ various retrieval methods, including BM25, DPR, Colbert, and BGE. For each of MC- indexing and FLC- 300, we first use these retrievers to sample the sections related to the question. Given the retrieved sections, we proceed to generate answers using the prompt provided in Figure 10. The generated answers are then compared using pairwise comparison (see Section 5.2).

The results of this comparative assessment are displayed in Figure 4. We find that MC- indexing consistently demonstrates higher win rates than loss rates against FLC- 300 across all retrievers and both evaluation metrics.

Positional bias in LLM may cause it to assign higher scores to the first answer in the prompt. Unlike score- based evaluation, which takes into account the magnitude of score differences, round- based evaluation is purely predicated on the number of rounds won by each answer. Consequently, we anticipate that the round- based evaluation will yield more ties than the score- based evaluation.

### 5.6Chunking Error

As previously discussed in Section 1, FLC tends to cause significant chunking errors. In this section, we elaborate the chunking errors from two fixedlength chunking strategies.

Firstly, the existing FLC method is contentagnostic. This is due to the fact the method divides the entire document into fixed- length chunks, which may inadvertently break a coherent section into separate parts. Alternatively, we recommend a different FLC approach that segments each section of the document into fixed- length chunks. This would ensure that a chunk doesn't span across two different sections, thereby more robust to chunking errors. In summary, our proposed content- aware chunking strategy ensures that no chunk extends over two sections, effectively reducing chunking errors. Results shown in Table 4 highlight the impact of content- aware chunking on chunking error.

## 6Conclusion

In this paper, we propose a new approach: Multiview Content- aware indexing (MC- indexing) for more effective retrieval of long document. Specially, we propose a long document QA dataset which annotates not only the question- answer pair, but also the document structure and the document scope to answer this question. We propose a content- aware chunking method to segment the document into content chunks according to its organizational content structure. We design a multiview indexing method to represent each content chunk in raw- text, keywords, and summary views. Through extensive experiments, we demonstrate that content- aware chunking can eliminate chunking errors, and multi- view indexing can significantly benefit long DocQA. For future work, we would like to explore how to use the hierarchical document structure for more effective retrieval. Moreover, we would like to train or finetune a re- triever that can generate more fine- grained or nuanced embeddings across multiple views.

### Limitations

The limitations of our method MC- indexing, can be evaluated from two primary perspectives.

Firstly, our method heavily depends on the structured format of a document. When the document lacks clear indications of content structure, applying our content- aware chunking technique becomes challenging. However, we would like to emphasize that our work focuses on structured indexing and retrieval of long documents, and long documents usually have structured content to be utilised. It is unusual to encounter lengthy and poorly structured documents in which the authors have written tens of thousands of words without providing clear document section or chapter demarcations.

To expand the usability of our method to unstructured documents, we propose the development of a section chunker that can segment documents without predefined content structures into atomic and coherent semantic units as a future research direction.

Secondly, short documents, being within the Large Language Model's (LLM) capacity, which means structured layout might not be required for the model to perform Question Answering (QA) tasks efficiently. Hence, we clarify that our method does not aim to enhance retrieval performance on unstructured short document. In contrast, our method can significantly benefit the retrieval of structured long documents.

## References

Payal Bajaj, Daniel Campos, Nick Craswell, Li Deng, Jianfeng Gao, Xiaodong Liu, Rangan Majumder, Andrew McNamara, Bhaskar Mitra, Tri Nguyen, Mir Rosenberg, Xia Song, Alina Stoica, Saurabh Tiwary, and Tong Wang. 2018. Ms phrase: A human generated machine reading comprehension dataset. Jan Buchmann, Max Eichler, Jan- Micha Bodensohn, Ilia Kuznetsov, and Iryna Gurevych. 2024. Document structure in long document transformers. In Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1056- 1073, St. Julian's, Malta. Association for Computational Linguistics. Andrea Burns, Krishna Srinivasan, Joshua Ainslie, Geoff Brown, Bryan A. Plummer, Kate Saenko, Jianmo Ni, and Mandy Guo. 2023. Wikiweb2m: A page- level multimodal wikipedia dataset. Howard Chen, Ramakanth Pasumuru, Jason Weston, and Asli Celikyilmaz. 2023. Walking down the mem

ory maze: Beyond context limit through interactive reading.

Gautier Izacard, Mathilde Caron, Lucas Hosseini, Sebastian Riedel, Piotr Bojanowski, Armand Joulin, and Edouard Grave. 2022. Unsupervised dense information retrieval with contrastive learning. Trans. Mach. Learn. Res., 2022.

Gautier Izacard and Edouard Grave. 2021. Leveraging passage retrieval with generative models for open domain question answering. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 874- 880, Online. Association for Computational Linguistics.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lelio Renard Lavaud, Marie- Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b.

Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. 2017. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601- 1611, Vancouver, Canada. Association for Computational Linguistics.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen- tau Yih. 2020. Dense passage retrieval for open- domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6769- 6781, Online. Association for Computational Linguistics.

Omar Khattab and Matei Zaharia. 2020a. Colbert: Efficient and effective passage search via contextualized late interaction over bert. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR '20, page 39- 48, New York, NY, USA. Association for Computing Machinery.

Omar Khattab and Matei Zaharia. 2020b. Colbert: Efficient and effective passage search via contextualized late interaction over bert. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, SIGIR '20, page 39- 48, New York, NY, USA. Association for Computing Machinery.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Ilia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming- Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural questions: A benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:452- 466.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen- tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval- augmented generation for knowledge- intensive nlp tasks. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS'20, Red Hook, NY, USA. Curran Associates Inc.

Zehan Li, Xin Zhang, Yanzhao Zhang, Dingkun Long, Pengjun Xie, and Meishan Zhang. 2023. Towards general text embeddings with multi- stage contrastive learning.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2023. Lost in the middle: How language models use long contexts.

Yixin Nie, Songhe Wang, and Mohit Bansal. 2019. Revealing the importance of semantic retrieval for machine reading at scale. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP- IJCNLP), pages 2553- 2566, Hong Kong, China. Association for Computational Linguistics.

Yuxiang Nie, Heyan Huang, Wei Wei, and Xian- Ling Mao. 2022. Capturing global structural information in long document question answering with compressive graph selector network. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 5036- 5047, Abu Dhabi, United Arab Emirates. Association for Computational Linguistics.

Yuxiang Nie, Heyan Huang, Wei Wei, and Xian- Ling Mao. 2023. AttenWalker: Unsupervised long- document question answering via attention- based graph walking. In Findings of the Association for Computational Linguistics: ACL 2023, pages 13650- 13663, Toronto, Canada. Association for Computational Linguistics.

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100.000+ questions for machine comprehension of text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2383- 2392, Austin, Texas. Association for Computational Linguistics.

Stephen E Robertson, Steve Walker, Susan Jones, Micheline M Hancock- Beaulieu, Mike Gatford, et al. 1995. Okapi at trec- 3. Nist Special Publication Sp, 109:109.

Stephen E. Robertson and Hugo Zaragoza. 2009. The probabilistic relevance framework: BM25 and beyond. Found. Trends Inf. Retr., 3(4):333- 389.

Gerard Salton, Edward A Fox, and Harry Wu. 1983. Extended boolean information retrieval. Communications of the ACM, 26(11):1022- 1036.

Parth Sarthi, Salman Abdullah, Aditi Tuli, Shubh Khanna, Anna Goldie, and Christopher D. Manning. 2024. Raptor: Recursive abstractive processing for tree- organized retrieval.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie- Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023a. Llama: Open and efficient foundation language models.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghur Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie- Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kembadur, Sharan Narang, Aurelien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023b. Llama 2: Open foundation and fine- tuned chat models.

Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. 2022. Text embeddings by weakly- supervised contrastive pre- training.

Shitao Xiao, Zheng Liu, Yingxia Shao, and Zhao Cao. 2022. Retromac: Pre- training retrieval- oriented language models via masked auto- encoder.

Shitao Xiao, Zheng Liu, Peitian Zhang, and Niklas Muennighoff. 2023. C- pack: Packaged resources to advance general chinese embedding.

Liu Yang, Mingyang Zhang, Cheng Li, Michael Bendersky, and Marc Najork. 2020. Beyond 512 tokens: Siamese multi- depth transformer- based hierarchical encoder for long- form document matching. In Proceedings of the 29th ACM International Conference on Information & Knowledge Management, CIKM '20, page 1725- 1734, New York, NY, USA. Association for Computing Machinery.

## A Appendix

### A.1 Implementation Details of Retrievers

We used 6 dense retrieval models and 2 sparse retrieval models in section 5. The dense retrieval

![[table_5_implementation_details_for_dense.png]]

models deployed are namely DPR (Dense Passage Retriever), ColBERT, Contriever, E5, BGE and GTE. These models use the WordPiece tokenizer from Bert and also inherit the maximum input length of 512 tokens from Bert. We use pretrained checkpoints available on HuggingFace 4; the specific checkpoint information can be found in Table 5 alongside other configuration details. Additionally, we make use of the sentence- transformer library when deploying E5, BGE and GTE.

The 2 sparse retrieval models implemented are BM25 and TF- IDF (Term Frequency - Inverse Document Frequency). Note that when calculating scores for BM25 and TF- IDF for each question, we restrict the set of corpus to chunks appearing in the sole relevant Wikipedia article. For BM25, we use the code from github repository: https://github.com/dorianbrown/rank_bm25. For TF- IDF we use the TF- IDF Vectorizer from scikit- learn library 6. We briefly describe how we rank document using the TF- IDF vectorizer here. First, given the corpus (i.e. the chunks appearing in the sole relevant Wikipedia article) we convert each chunk into a sparse vector with each entry indicating the TF- IDF score of each word appearing in the chunk. Next, we convert the question into a sparse vector. Finally to rank each chunk, we calculate the cosine similarity between the question sparse vector and sparse vectors of each individual chunk.

### A.2 Extended Ablation Study on NQ

In this section, we reported the ablation results of MC- indexing on NQ dataset, serving as the extension of Section 5.4. From the data in Table 6, it's evident that: (1) Removing the raw- text view leads to the most significant performance drop, ranging between 3.2 and  \(7.6\%\) . (2) Eliminating the summary view results in the second- most considerable performance drop, varying between 3.2 and  \(6.6\%\) . (3) Disregarding the keywords view contributes to a performance drop between 2.7 and  \(5\%\) .

![[table_6_ablation_study_of_recall.png]]

### A.3 Question Generation for WikiWeb2M

We aim to generate question that tends to rely on a long answer scope. Typically, the length of answer scope ranges from 50 to 500 tokens. We define questions of the following 8 types:

- Narrative and Plot Details: inquire specific details or sequence of events in a narrative (e.g., a story, movie, or historical account).- Summarization: require the summarization of a

![](images/35e3c507cd4cb4698783e890cd09813549d00f5b50cd88f64f5805492d1d89c1.jpg)  
Figure 5: Pie chart of question type distribution.

long passage, argument, or complicated process. Inferential and Implied: depend on understanding subtleties and reading across a long passage. Information Synthesis: inquire the synthesis of information dispersed across a long passage. Cause and Effect: understand the causal relationship between events in a long passage. Comparative: ask for comparisons between different ideas, characters, or events within a text. Explanatory: ask for explanations of complex concepts or processes that are described in detail. Themes and Motifs: consider entire text to identify patterns and conclude on central messages.

The distribution of generated question types is shown in Figure 5.

### A.4 Question Answer Annotation for WikiWeb2M

For each given section, we request the advanced commercial LLM to generate 3 questions, the corresponding answers and identify the raw text that maps to the answer. In our prompt from Figure 6, we provide LLM the raw text of the given section, the description of the 8 question types from Appendix A.3 and our designed prompt instruction. Our prompt instruction ensures LLM to generate the continuous context sentences to sufficiently answer the question. The answer scope is then used to evaluate the retrieval efficiency of MC- indexing.

### A.5 Prompt Design

In this paper, we utilize the following prompts on the advanced commercial LLM to facilitate the respective process:

- The generation of WikiWeb2M question, question type, answer, and answer contextual sentences. The prompt is shown in Figure 6.
- The contextual sentences retrieval when provided with a long document or a section of the document. This is used to evaluate if the advanced commercial LLMs can directly cope with long document. The prompt is shown in Figure 7. 
- The generation of summary for the sections consisting of more than 200 tokens. 
- The generated summary is used as additional view for document indexing. The prompt is shown in Figure 8. 
- The generation of the list of keywords for each section. The generated keywords list is used as additional view for document indexing. The prompt is shown in Figure 9. 
- The answer generation when provided with retrieved top  \(k\)  chunks or sections. The prompt is shown in Figure 10. 
- The automatic answer evaluation of two answers, given the ground truth answer. This is used to evaluate the answer quality. This prompt is shown in Figure 11.

### A.6 Evaluating LLM on Long Document QA.

We assess if state- of- the- art LLMs can effectively handle long document Question- Answering (QA). We have opted for the Span- QA setting to simplify the process. Here, the gold answer is a span of raw text extracted directly from the input document. We then measure the precision, recall, and  \(\mathrm{F_1}\)  score of the retrieved span against this gold answer.

Since the most advanced commercial LLM is capable of taking fairly long input, it is tasked to analyze longer documents, approximately 30k tokens, and given 2,000 questions to answer. These questions are all sourced from our Wiki- 2M dataset (in Section 4.1). To further determine if LLMs can perform more proficiently given a shorter answer scope, we adapt our input approach. Instead of feeding the model the entire document, only the corresponding section is inputted, which contains an average of 370 tokens.

Finally, we compare the performances when using the entire document versus using only a section. This allows for a comprehensive evaluation of the models' capabilities in handling different lengths of text. The result is elaborated in Figure 2.

You are a sophisticated question generator. You need to use the reference text to generate a question, with its question type, and the supporting context sentences, and the short answer.

The generation should strictly follow the following guidelines:

![[last_text.png]]