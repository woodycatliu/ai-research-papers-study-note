# Recursively Summarizing Enables Long-Term Dialogue Memory in Large Language Models

![[qingyuetitle1.png]]

Abstract

Recently, large language models (LLMs), such as GPT- 4, stand out remarkable conversational abilities, enabling them to engage in dynamic and contextually relevant dialogues across a wide range of topics. However, in a long- term conversation, these chatbots fail to recall appropriate information from the past, resulting in inconsistent responses. To address this, we propose to recursively generate summaries/ memory using large language models to enhance their long- term dialog ability. Specifically, our method first stimulates the LLM to memorize small dialogue contexts. After that, the LLM recursively produces new memory using previous old memory and subsequent contexts. Finally, the chatbot is prompted to generate a response based on the latest memory. The experiments on widely used LLMs show that our method generates more consistent responses in long- term

conversations, and it can be significantly enhanced with just two/ three dialog illustrations. Also, we find that our strategy could nicely complement both large context windows (e.g., 8K and 16K) and retrieval- enhanced LLMs, bringing further long- term dialogue performance. Notably, our method is a potential solution to enable the LLM to model the extremely long dialog context. The code and scripts will be released later.

Keywords: recursive summary, long- term memory, large language models, dialog generation.

## 1. Introduction

Recently, large language models (LLMs), such as ChatGPT<sup>1</sup> and GPT- 4 (Achiam et al., 2023), demonstrate promising performances in various natural language applications (Brown et al., 2020; Zeng et al., 2022; Zhong et al., 2023; Lu et al., 2023b; Peng et al., 2023; Wu et al., 2023). One notable capability lies in their remarkable conversational prowess, comprehending input, and generating human- like responses.

Large context windows allow many LLMs<sup>2</sup> to process entire dialog histories, yet they often struggle to effectively comprehend past interactions and integrate key information into responses (Zhou et al., 2023). Applications such as personal AI companions, which need to recall past conversations for rapport building, and health assistants, which must consider a complete record of patient inquiries to provide diagnostic results, demonstrate the importance of maintaining consistency

![](images/645493a8eddeefd5f5b773f36cae49e0eee60ecd7de81a9d05dd2e5512563a9b.jpg)  
Figure 1: A long-term conversation example from the Multi-Session Chat Dataset (Xu et al., 2022a). When the user refers back to previous subjects (i.e., composing music), even the ChatGPT (gpt-turbo-3.5-0301 version) generates an inconsistent response.

and coherence in long- term dialogues. Figure 1 illustrates a dialog spanning over 20 turns, centered around a discussion of the speakers' personas (e.g., the bot composes music, and the user enjoys country music). However, even the powerful ChatGPT forgets past information and produces a poor response, showing the necessity to explicitly model long- term memory during conversations.

To address this, there are two mainstream methods to enhance the long- term dialog ability of LLMs. The first one is the retrieval- based method, which directly stores past conversational utterances in the storage and adapts an advanced retriever to identify the most relevant history (Guu et al., 2020; Lewis et al., 2020). However, it is difficult to obtain a well- performing (ideal) retriever, ensuring that the retrieved utterances capture the complete semantics about current conversations. The second way is to employ a memory module to summarize important conversation information to assist the LLM, which is also called memory- based

approaches (Mazaré et al., 2018; Xu et al., 2022b; Chen et al., 2024a). They usually apply a separately trained model or a powerful large language model to generate memory for past dialogues. Nevertheless, these methods lack the necessary iteration mechanism on generated memory, resulting in the reserved outdated information directly hurting the quality of responses.

In this paper, we propose a simple and effective plug- in method that enables LLM itself to generate summaries, which store the real- time information of speakers through continuous updating and reviewing past context to aid long- term interactions. In practice, a generative LLM is first prompted to produce a summary given a short dialog context. After that, we ask the LLM to continue updating and generate a new summary/ memory by combining the previous memory and subsequent dialogues. Finally, we encourage the LLM to respond using the latest memory as the primary reference to engage the ongoing dialogue. Given that the generated summaries are much shorter than the full dialogues, our proposed schema not only models long- term conversation memory but also serves as a potential solution to enable current LLMs to handle extremely long contexts (across multiple dialogue sessions) without expensively expanding the maximum length setting.

Experimentally, we implement our method using a variety of state- of- the- art open (Llama (Touvron et al., 2023) and ChatGLM (GLM et al., 2024)) and closed (OpenAI's GPT- 3.5- Turbo) LLMs, and the performance on long- term dialog surpasses that of popular approaches both in automatic and human evaluations. Moreover, we verify the effectiveness of using explicit memory for long- term dialogs and using our generated memory is easier for LLMs to digest. These findings underscore the importance of developing advanced memory generation

strategies. Our method can further enhance response quality by incorporating the in- context learning (ICL) technique, where multiple samples in the format of (dialogue, memory, and golden response) are presented to LLMs. This allows them to utilize the generated memory more flexibly. Additionally, we demonstrate the generalizability of our approach across different LLMs, with our method achieving approximately a  \(+3\%\)  improvement in BLEU score on text- davinci- 003. Finally, we observe that our schema complements existing window- extended LLMs (e.g., GPT- 3.5- Turbo- 16k and LongLoRA- 8k) and retrieval- enhanced LLMs (e.g., LLM- BM25 and LLM- DPR), producing more coherent and consistent responses in long- term conversations.

In summary, our contributions are as follows:

- We propose a novel method by recursively summarizing past dialogues to enhance the LLM's memory, enabling the generation of highly consistent responses in long-term conversations.- The solid experiments on the public datasets show the superiority of the proposed method, with multiple open-source and closed-source LLMs verifying its universality and robustness.- The simplicity of our method makes it nicely complements existing works, including retrieval-based and long-context techniques, having the great potential to be an orthogonal plug-in for the LLM community.

## 2. Related Work

### 2.1. Large Language Models

Language language models (LLMs) have shown outstanding performance in a variety of user- facing language technologies, including conversation, summarization, and creative writing (Achiam et al., 2023; Shuster et al., 2022; Rubin and Berant, 2024). While these LLMs achieve notable success in many popular tasks, their ability to model long text remains a challenge (An et al., 2023). To address the problem, some works are to adapt transformers to accommodate longer inputs, such as position interpolation (Chen et al., 2023) and efficient self- attention (Beltagy et al., 2020; Chen et al., 2024b). However, these context window- extended LLMs not only require continual training on high- quality long texts but still struggle to use and retrieve the core information from the entire input (Liu et al., 2023). Recently, in question- answer task, some researchers found that the performances of LLMs degrade significantly when people change the position of relevant information, indicating that current language models do not robustly make use of information in long input contexts (Liu et al., 2023; Li et al., 2023). Many works suggest that the lack of explicit memory mechanisms in current LLMs hinders their performance on tasks requiring sustained context awareness and understanding (Chen et al., 2024a). Nowadays, the performance of LLMs has not yet been explored deeply in long- range dialogue scenarios. This work focuses on developing the long- term modeling ability of the LLM, where we prompt it to self- memory, self- update, and self- use in conversations, aiding consistent response generation.

### 2.2. Long-term Open-Domain Dialogue

2.2. Long- term Open- Domain DialogueOpen- domain dialogue systems (Liu et al., 2016; Zhang et al., 2018; Kann et al., 2022), also known as chatbots or conversational agents, have gained immense popularity and a lot of studies in recent years. Among them, the long- term conversation setting is pretty a hard problem, because it needs the capability of understanding and memorizing key dialogue history information (Wu et al., 2022; Zhang et al., 2022) about current query. The most popular and potential solution is to directly store the partial information for tracking the history of conversation (Lee et al., 2023), usually in the form of dialogue utterances or summaries. The current conversation and relevant information are then inputted into the response generator. One intuitive idea is to apply a retriever to find the most relevant utterances according to the current dialog, which is called as the retrieval- based method. Another popular method is a memory- based method, which tries to generate and manage the summary to obtain key information from history. For example, MemoChat (Lu et al., 2023a) allows chatbots to reorganize the past dialogue histories according to different topics of speakers and prompt the LLM to retrieve from the structured memory during generation. Going further, MemoryBank (Zhong et al., 2024) proposes a new memory mechanism by generating summaries for each dialog session first and then compressing them into a global one. However, their memory is completely fixed once stored, failing to guarantee its consistency with ongoing dialog. The important comparison between these existing methods and ours is shown in Figure 2. As seen, our approach and these methods mainly diverge in the way of memory generation, where we continuously integrate historical information and old memory to obtain real- time memory, enabling the gain of accurate memory and modeling of long- distance dependencies.

![](images/77502265896ff2ca7c1c7d855b972107eccde1d97f8bab380c896a71ed6f93b6.jpg)  
Figure 2: Comparison among baselines and ours. The "U", "S", "T", and "M" are abbreviations for the Utterance, Session, dialog Topic, and Memory. The red dashed box refers to the memory used to generate the response.

## 3. Approach Overview

Following previous works (Xu et al., 2022a; Bae et al., 2022), we denote that a long- context dialogue consists of multiple sessions with a specific user, which is also called Multi- Session Dialogue. The goal of the task is to generate context- relevant and highly consistent responses to the user based on past sessions and current context. Formally, each dialogue can be written as  \(D = \{S, C_t, r_t\}\) . Here,  \(S = \{S_1, S_2, \ldots , S_N\}\)  represents  \(N\)  past sessions and each of the sessions consists of multiple utterances between two speakers.  \(r_t\)  is the ground truth response to  \(C_t\)  with background sessions  \(S\) .  \(C_t = \{u_1, r_1, \ldots , u_t\}\)  denotes the dialogue context of the current session at  \(t\)  step, where  \(u\)  and  \(r\)  represent the utterances from the user and the chatbot, respectively.

In this paper, we propose a new memory mechanism to aid a large language model for multi- session dialog tasks. The memory contains multiple natural language sentences, storing the key information of speakers extracted from previous sessions. Our goal is to obtain a reliable memory given past sessions and predict a consistent and meaningful response using the current dialogue context and the

![](images/881c048bb72ffbac49cd6963c06ef13eb7f73872e4b61dd9340ace8709ac69e9.jpg)  
Figure 3: The schematic overview of our method. The model uses the first session to generate initial memory (green arrows), then updates the memory when the second session ends (yellow arrows), and generates a response using the latest memory at the third session (blue arrows).

memory. Specifically, we decompose the goal into two stages with the following probability distribution:

![[wdjaiwdjiajdawj.png]]

where  \(M_{i}\)  represents the available memory when the  \(i\) - th session is finished. And  
![[截圖 2025-08-17 下午5.42.49.png]]
 is a sequential or Markov process where each memory  \(M_{i}\)  of session  \(i\)  depends only on the current session and the previous memory  \(M_{i - 1}\) .

## 4. Approach

To achieve long- term dialog, we prompt an arbitrary large language model to finish two tasks, i.e., memory iteration and memory- based response generation. The former is responsible for recursively summarizing the key information along with long- term dialogue, and the latter is to incorporate the latest mem-

ory and current dialog to generate an appropriate and consistent response. The workflow of our proposed method is shown in Figure 3.

### 4.1. Memory Iteration

The goal of memory iteration is to obtain a coherent and up- to- date summary for the chatbot. Early works (Bae et al., 2022; Choi et al., 2023) update memory by carrying multiple "hard operations" on summaries, such as replace, append, and delete, which rely on high- quality dialogue with operation labels. However, this laborious design disrupts the semantic coherence of the summary and is not suitable for management over a long period. Differently, we guide the LLMs to recursively self- generate memory (summaries) using dialogue context and previous memory. By utilizing old summaries, the model can fully digest the current dialog context and thus gain a high- quality memory. Formally, the updated memory is computed by:

![[截圖 2025-08-17 下午5.43.47.png]]

where  \(M_{i} = \{m_{1},m_{2},\dots,m_{J}\}\)  denotes multiple sentences, containing summarized key information from the session  \(S_{i}\) , and  \(\mathrm{P}_{\mathrm{m}}\)  is the prompt of LLM for generating new memory. The memory iteration will be repeated  \(N\)  times until all previous sessions end, where we can obtain the latest memory  \(M_{N}\) . Take the dialog in Figure 3 as an example, two memory iterations happen at the end of the first and second sessions. In the second iteration, the LLM incorporates the new personality (i.e., the bot recently joined a new gym) from Session 2 into the old memory (i.e., the bot enjoys walking and running).

Prompt Construction. To enable LLM to efficiently carry the memory iteration task, we design a specific prompt for it, which is shown in Table 1. It mainly

Table 1: The prompt design of memory iteration, including task definition, task description and task inputs  

<table><tr><td>Prompt</td><td>You are an advanced AI language model with the ability to store and update a memory to keep track of key personality information for both the user and the bot. You will receive a previous memory and dialogue context. Your goal is to update the memory by incorporating the new personality information. To successfully update the memory, follow these steps:
1. Carefully analyze the existing memory and extract the key personality of the user and bot from it.
2. Consider the dialogue context provided to identify any new or changed personality that needs to be incorporated into the memory.
3. Combine the old and new personality information to create an updated representation of the user and bot&#x27;s traits.
4. Structure the updated memory in a clear and concise manner, ensuring it does not exceed 20 sentences. Remember, the memory should serve as a reference point to maintain continuity in the dialogue and help you respond accurately to the user based on their personality. [Previous Memory] [Session Context]</td></tr><tr><td>Output</td><td>[Updated Memory]</td></tr></table>

consists of three parts: (1) Task definition is responsible for defining the role of the current LLM, as well as the memory iterator's (LLM) input and output. (2) Task description gives detailed steps to finish the above task. To make sure the memory update is timely, we remind the LLM to create a new representation of speakers by considering old summaries and current sessions.(3) Task input contains two placeholders, where we take previous memory and a whole session as the inputs. Through experimental verification, we found that using step- by- step instructions helps the LLM better understand and execute the memory iteration<sup>3</sup>.

Table 2: The prompt of memory-based response generation, including task definition, task description and task inputs  

<table><tr><td>Prompt</td><td>You will be provided with a memory containing personality information for both yourself and the user.
Your goal is to respond accurately to the user based on the personality traits and dialogue context.
Follow these steps to successfully complete the task:
1. Analyze the provided memory to extract the key personality traits for both yourself and the user.
2. Review the dialogue history to understand the context and flow of the conversation.
3. Utilize the extracted personality traits and dialogue context to formulate an appropriate response.
4. If no specific personality trait is applicable, respond naturally as a human would.
5. Pay attention to the relevance and importance of the personality information, focusing on capturing the most significant aspects while maintaining the overall coherence of the memory.
[Previous Memory] [Current Context]</td></tr><tr><td>Output</td><td>[Response]</td></tr></table>

### 4.2. Memory-based Response Generation

The final goal is to produce consistent and natural responses given dialogue memory and the context of the current session. Formally, the response of the current session can be obtained by stimulating the LLM as a response generator:

![[截圖 2025-08-17 下午5.44.31.png]]

where  \(P_{r}\)  is a prompt of the generator. Especially, taking the generated memory  \(M_{N}\)  and current session  \(C_t\)  , we ask the LLM again to generate a response. In Figure 3, the LLM considers the newest memory from Session 2 and provides a high consistent response to the user, i.e., " more flexibility with a 24- hour gym membership".

Prompt Construction. The the prompt for memory- based response generation is shown in Table 2. The prompt is similar to that of memory iteration, including task definition, description, and inputs, where we remind the LLM to utilize the

![[截圖 2025-08-17 下午5.45.00.png]]

extracted information and maintain the consistency of memory when responding. Also, the step- by- step instructions method is also effective for memory- based response generation.

### 4.3. Algorithm

The process of response generation using recursive memory is illustrated in Algorithm 1. In the beginning, the initial memory is set as an empty, i.e., "none" string. After that, we recursively update the memory using each session context (line 3). Finally, the LLM generates a response with the help of the latest memory (line 5). The generative pre- trained models used for memory iteration and response generation can be different. For instance, the developers can train a private memory iterator through customized models or data, to enhance the target open/

closed LLMs for long- term or long- context tasks.

## 5. Experimental Settings

### 5.1. Datasets

We validate the effectiveness of the proposed method on two widely- used long- term dialogue datasets: Multi- Session Chat (MSC) dataset (Xu et al., 2022a) and Carecall dataset (Bae et al., 2022).

MSC dataset. is the largest human- human long conversations dataset so far. The early sessions are a short conversation where two speakers get to know each other for the first time and then they either continue to talk about the previous subject or spark up conversation on a new topic.

Carecall. is a Korean open- domain multi- session dataset, which is used for monitoring patient health. For a fair comparison, we use the public machine- translated English version<sup>4</sup> in the experiments.

The CareCall's setting is similar procedure presented in the MSC dataset. The main difference is that the Carecall additionally contains more persona updates that are likely to change in a short period, such as the user's health and diet, while the persona information in the MSC dataset remains fixed once it is stored. Both of the two datasets have five sessions and each one consists of multiple utterances between two speakers (the user and chatbot), where the conversationalists reengage after several hours or days and continue chatting. As early sessions only have a very short history of conversations, we mainly evaluate the proposed method in

session 4 and 5 for the ability of long- term modeling. The statistics of two datasets are given in Appendix A.

### 5.2. Evaluation Metrics

we conduct diverse evaluations during experiments, including automatic metrics, human evaluation, and LLM judgments, focusing on the quality of generated memory and response.

Automatic Metrics. We employ BLEU- 1/2 (Papineni et al., 2002), F1 with reference to the human annotations. Besides, we compute BertScore (Li et al., 2016) to measure the semantics similarity between the references and the generated responses.

Human Evaluation. Many works point out that automatic evaluation metrics are insufficient for capturing the nuances of conversations (Deriu et al., 2019). Following the previous works (Bae et al., 2022), we ask three crowd- sourcing workers to assign a score from 0 to 2 (0:bad, 1:OK, 2:good) to the generated responses based on the aspects of engagingness, coherence, and consistency. These criteria are discussed as follows: (1) Engagingness: It evaluates whether the chatbot captures the user's interest and makes them want to continue the conversation. A high score on engagingness means the responses are interesting and contextually appropriate, encouraging users to keep chatting. (2) Coherence: It measures whether the response maintains a logical and clear flow based on the conversation's context. A coherent response ensures the conversation makes sense and stays relevant, enhancing user engagement. (3) Consistency: It assesses whether the response aligns with the information provided in previous interactions. Consistent responses build trust and reliability by demonstrating that the chatbot remembers and integrates past exchanges accurately.

LLM Evaluation. Recently, LLM- as- a- Judge strategy (Pan et al., 2023) has been widely used in evaluating generation tasks. Some works reveal minimal deviation of GPT- 4's evaluation from humans ( \(>0.85\)  agreements) in dialog quality (Zhang et al., 2023a). Inspired by this, we employ the GPT- 4 as an advanced evaluator, using two common methods to assess the quality of generated responses. (1) Single model evaluation (Lu et al., 2023a): we prompt GPT- 4 to rate the responses individually from the three aspects, i.e., engagingness, coherence and consistency with an integer scale from 1 (very bad) to 100 (very good). (2) Pairwise model evaluation (Dubois et al., 2024): we ask the GPT- 4 to directly compare two anonymous generations and determine which response is better. While single model evaluation provides detailed insights into specific aspects of each response, pairwise comparison is essential for understanding relative performance, particularly when distinguishing subtle differences between outputs.

## 5.3. Baselines

We mainly employ the following methods for long- text dialogues in LLMs: context- only approaches (without using any memory), retrieval- based approaches (with different retrievers), and memory- based approaches (with different memory mechanisms)

Context- only Approach. It is the most naive approach to directly employ the LLM as a chatbot, where it concatenates past sessions and current dialogue context as

the input. We use “Llama2- 7B” (Touvron et al., 2023), “ChatGLM2- 6B”<sup>5</sup>, and OpenAI ChatGPT “gpt- 3.5- turbo- 0301” as the backbone LLMs for the context- only approach<sup>6</sup>.

Retrieval- based Approach. Many previous works (Xu et al., 2022a) employ re- . trievers to filter key information and then include top-  \(k\)  documents into inputs to assist long- context dialogs. For the long- term dialog, the top-  \(k\)  documents refer to the relevant utterances from history. Here, we choose two widely used retrieval algorithms, i.e., BM25 (Robertson et al., 2009) and pre- trained dense passage retrieval (DPR) (Karpukhin et al., 2020), to look up the relevant utterances from past sessions. For convenience, we name the above retrieval- based baselines as ChatGPT- BM25 and ChatGPT- DPR, respectively.

Memory- based Approach. Recent works employ a summarizer to abstract important information from the past to aid long- term conversation. Simply, we just choose two representative methods from various memory- based techniques, MemoryBank (Zhong et al., 2024) and MemoChat (Lu et al., 2023a). MemoryBank proposes a human- like long- term memory mechanism, which creates ordered summaries of past dialogs with timestamps, and then reorganizes them to obtain the global memory. The memory will be forgotten and updated by Ebbinghaus's forgetting curve. Here, we plug MemoryBank with ChatGPT as a strong baseline, named ChatGPT- MemoryBank. Differently, MemoChat maintains the structured conversational memory to aid long- term dialogue, i.e., generating the summaries for each dialogue topic. We plug the MemoChat into ChatGPT, named

ChatGPT- MemoChat, for a fair comparison with others.

Note that our approach focuses on the zero- shot setting for LLMs to engage in a long- term dialog, making the comparisons with other fine- tuned models unfair.

## 5.4. Implementation

We implement our method by letting the LLM response using recursively generated memory in a long- term dialogue, thus it is called as "LLM- Rsum".

Backbone LLMs. We employ OpenAI ChatGPT "gpt- 3.5- turbo- 0301", "Llama2- 7B" and "ChatGLM2- 6B" in the main experiments, "text- davinci- 003" and "Llama2- 7B" (Touvron et al., 2023) in the analysis to show the universality, "longlora- 8k" (Chen et al., 2024b) and ChatGPT- 16k "gpt- 3.5- turbo- 16k" as the backbones of complementary discussion. Unless otherwise specified, we employ the same LLM to finish the memory iteration and memory- based response generation. During generation, we set the temperatures of all LLMs as 0 for fair comparisons. The max length of input tokens for Carecall and MSC datasets is no more than 4k, thus all backbone LLMs in experiments can process the entire dialog context.

Retrievers. Considering the scale of past utterances is not large enough to use the FAISS (Research, 2019), we choose the top- k most relevant utterances that will be combined with ongoing dialogues, prompting the LLM to respond. Following previous works (Xu et al., 2022a) for long- term dialogs, the k is set into 3 and 5.

Memory- based Approaches. The implementation and prompt design of the MemoryBank and Memochat methods are based on the code publicly released in their original papers. For details, please refer to the original papers.

Table 3: Comparison of automatic and human evaluations among different methods on MSC and Carecall datasets, reporting the quality of generated response. The BScore", Enga,", Cohe" and "Cons" are the abbreviations of BertScore, Engagingness, Coherence, and Consistency. The best value is bolded.  

<table><tr><td rowspan="2">Method</td><td colspan="6">MSC Dataset</td><td colspan="6">Carecall Dataset</td></tr><tr><td>F1</td><td>BLEU-1/2</td><td>BScore</td><td>Enga.</td><td>Cohe.</td><td>Cons.</td><td>F1</td><td>BLEU-1/2</td><td>BScore</td><td>Enga.</td><td>Cohe.</td><td>Cons.</td></tr><tr><td>Context-only LLM</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Llama2-7B</td><td>16.43</td><td>20.96/12.09</td><td>84.04</td><td>1.32</td><td>1.20</td><td>1.13</td><td>13.71</td><td>20.89/12.28</td><td>84.49</td><td>0.75</td><td>0.75</td><td>1.00</td></tr><tr><td>ChatGLM2-6B</td><td>15.38</td><td>21.69/12.51</td><td>84.48</td><td>1.10</td><td>1.15</td><td>1.07</td><td>13.09</td><td>20.59/12.03</td><td>84.91</td><td>0.66</td><td>0.63</td><td>0.86</td></tr><tr><td>ChatGPT</td><td>19.41</td><td>21.23/12.24</td><td>86.13</td><td>1.83</td><td>1.37</td><td>1.32</td><td>13.69</td><td>21.15/12.20</td><td>85.53</td><td>1.50</td><td>1.52</td><td>1.43</td></tr><tr><td>Retrieval-based Approach</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ChatGPT-BM25 (k=3)</td><td>19.56</td><td>21.60/12.46</td><td>85.82</td><td>1.72</td><td>1.48</td><td>1.32</td><td>12.64</td><td>21.57/12.44</td><td>85.24</td><td>1.40</td><td>1.31</td><td>1.31</td></tr><tr><td>ChatGPT-DPR (k=3)</td><td>20.23</td><td>21.75/12.55</td><td>86.04</td><td>1.76</td><td>1.51</td><td>1.34</td><td>12.21</td><td>21.39/12.35</td><td>85.25</td><td>1.55</td><td>1.35</td><td>1.45</td></tr><tr><td>Memory-based Approach</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>ChatGPT-MemoChat</td><td>18.93</td><td>21.82/12.59</td><td>85.99</td><td>1.70</td><td>1.55</td><td>1.35</td><td>11.19</td><td>21.07/12.18</td><td>85.22</td><td>1.45</td><td>1.20</td><td>1.30</td></tr><tr><td>ChatGPT-MemoBank</td><td>20.28</td><td>21.82/12.58</td><td>86.12</td><td>1.78</td><td>1.57</td><td>1.40</td><td>13.15</td><td>21.29/12.39</td><td>85.34</td><td>1.57</td><td>1.52</td><td>1.68</td></tr><tr><td>ChatGPT-Rstm (Ours)</td><td>20.48</td><td>21.83/12.59</td><td>86.89</td><td>1.85</td><td>1.60</td><td>1.45</td><td>14.02</td><td>21.64/12.48</td><td>86.05</td><td>1.62</td><td>1.60</td><td>1.70</td></tr></table>

LLM evaluations. We use the GPT- 4 model (version "gpt- 4- 0314") as the evaluator, setting the temperature to 0 when making judgments. The prompts for evaluating the single model and pairwise models are referred to (Lu et al., 2023a) and (Dubois et al., 2024), respectively. All prompts used in the experiments can be seen in Appendix B.

# 6. Experimental Results

## 6.1. Main Results

Automatic Metrics Results. In Table 3, we compare different methods over session 5 in the MSC and Carecall datasets using popular LLMs. Firstly, among the vanilla models ("Llama2- 7B", "ChatGLM2- 6B", and "ChatGPT"), ChatGPT consistently performs well across two datasets, with competitive scores in BScore, F1 and BLEU- 1/2. These results illustrate that ChatGPT is robust enough to handle a

long- term dialog, thus, we leave ChatGPT as the backbone model for our method. Secondly, as expected, the proposed method ("ChatGPT- Rsum") achieves the best performance on both datasets, showing the benefits of using automatically recursive memory. Specifically, our method achieves about  \(+0.2\%\)  on the F1 score, which is acceptable compared to that in previous works (Xu et al., 2022a). The MSC datasets are harder than other open- domain datasets due to the 3x context, so the slight improvements are normal. Thirdly, retrieval- based methods may not be always helpful in enhancing the quality of generation. From the results, the performances of ChatGPT- BM25 and ChatGPT- DPR are much worse than vanilla ChatGPT in the Carecall dataset, which is completely contrary to that in MSC. The reason is that the chatbot needs to actively guide the dialog topics in the Carecall, thus it is hard to retrieve appropriate and relevant context from the user's query. Therefore, the performance of generated responses will be damaged due to unrelated information.

Human Evaluation Results. We also present the results of human evaluations on different methods in Table 3. From the results, we find that: 1) Most memory- augmented methods gain higher scores on consistency and coherence than vanilla ChatGPT, proving that maintaining a memory is more effective than using the whole history directly when engaging in a long- term conversation for LLMs; 2) Our method can generate more engaging response than other memory- based baselines (ChatGPT- MemoryBank and ChatGPT- MemoChat). The reason is that continually updating memory actively establishes global dependencies inside past histories, which helps LLMs to better understand the dialog and generate high- quality responses.

Table 4: Comparison of evaluation metrics results by GPT-4 across different methods on session 5 in MSC dataset.  

<table><tr><td>Method</td><td>Engagingness</td><td>Coherence</td><td>Consistency</td><td>Average</td></tr><tr><td>ChatGPT</td><td>75.48</td><td>75.00</td><td>75.48</td><td>75.32</td></tr><tr><td>ChatGPT-MemoryBank</td><td>74.68</td><td>80.92</td><td>84.56</td><td>80.05</td></tr><tr><td>ChatGPT-MemoChat</td><td>72.32</td><td>77.36</td><td>78.96</td><td>76.21</td></tr><tr><td>ChatGPT-Rsum (Ours)</td><td>78.92</td><td>83.56</td><td>84.76</td><td>82.41</td></tr></table>

## 6.2. LLM Evaluation

Single Model Evaluation. Table 4 reports the GPT- 4's evaluation metrics results on session 5 among various methods in the MSC dataset. Also, the results prove the high agreements between humans (in Table 3) and GPT- 4 judgment on the overall quality of generated responses, i.e., ChatGPT- Rsum >ChatGPT- MemoryBank >ChatGPT- MemoChat >ChatGPT. Given this, we mainly present the LLM evaluations in the following experiments to reduce labor costs. Lastly, it is worth noting that compared to human evaluation results, the GPT- 4 tends to give higher scores on both sentence coherence and consistency. We suppose that the values of humans and LLMs might not be fully aligned at a fine- grained level, which could be a new direction for developing LLM evaluation.

Pairwise Models Evaluation. Furthermore, we randomly sample 1000 generated responses from pairwise models, i.e., ours vs. baselines, and then ask the GPT- 4 to decide which response is better based on engagingness, consistency, and coherency. The results are shown in Figure 4. Compared to the most competitive baseline (MemoryBank), our proposed method obtains a  \(36.3\%\)  improvement (winning  \(48.2\%\)  and only losing  \(11.9\%\) ), which illustrates the advancement of the proposed iteration mechanism.

![](images/3f2906ce33f5a4887b07525261b470c652c3964710969ef80695a309906874e0.jpg)  
Figure 4: Comparative win rate of our method and competitive baselines, including ChatGPT, ChatGPT-MemoChat, and ChatGPT-MemoryBank.

## 6.3. Ablation Study

To better understand the effectiveness of proposed memory mechanisms for LLMs, we employ ChatGPT as the LLM and conduct ablation studies in session 5. The results are shown in Table 5. First, we only use the dialogue context of the current session as the input of LLMs, named "W/O Memory". As expected, the performance of the model has decreased significantly, proving the necessity of available memories from the past in a long- term conversation. Second, we replace the generated memory with ground truth memory, i.e., prompt ChatGPT to generate responses using golden memory and dialogue context, named "Gt. Memory". Interestingly, the model gains lower BLEU and F1 scores than using predicted memory (Ours). The potential reason is that the golden memories, e.g., "I am trying to lose weight" and "I want to start running", are fragmented and lack cohesiveness, which is sub- optimal as the LLM's prompt, whereas recursively summarizing memory generation method could wisely model the long dependencies, and generate the easy- to- digest prompt for LLMs. More analysis

Table 5: The ablation study on memory in MSC dataset.  

<table><tr><td>Method</td><td>BScore</td><td>F1</td><td>BLEU-1/2</td></tr><tr><td>ChatGPT-Rsum (Ours)</td><td>86.89</td><td>20.48</td><td>21.83/12.59</td></tr><tr><td>W/O Memory</td><td>85.40</td><td>18.94</td><td>21.10/12.17</td></tr><tr><td>Gt. Memory</td><td>85.93</td><td>20.46</td><td>21.50/12.40</td></tr></table>

can be seen in  \(\S 6.4\)  and  \(\S 6.5\)

## 6.4. Analysis

Beyond the main results and ablation study, we also aim to delve deeper into our method. In the following part, we would like to discuss several research questions (RQs): RQ1: What is the quality of the generated memory? RQ2: What errors may occur in memory generation? RQ3: Is the proposed method robust to other LLMs? RQ4: Can our zero- shot method be effectively applied in few- shot scenarios?

Our method can produce accurate and available memory (Q1). Central to this framework is to generate the dialog memory by continuously summarizing. To verify the quality of summarization, we compute automatic metrics between predicted memory and golden memory in the MSC dataset on ChatGPTMemoryBank and ours, respectively, which is shown in Figure 5. As seen, the generated memories of both models gain considerable F1 scores  \((+25\%)\)  , explaining the reliability of using dialogue summaries as memory. Besides, ChatGPTRsum (Ours) achieves higher overall performances on memory, suggesting that recursively summarizing can obtain more complete and long- term information than ChatGPT- MemoryBank. Finally, given the response performances of session 5 in Table 3 (Ours  \(>\)  ChatGPT- MemoryBank), we suppose that the accuracy of memory prediction is positively correlated with the quality of response. We also

![](images/1d964f993c1c099d09aaa5d7424b045721c22bbd5daeb4b16c27a6242a460c16.jpg)  
Figure 5: The evaluation on generated memory on ChatGPT-MemoryBank and ours. The "P" and "R" refer the precision and recall, respectively.

Table 6: Three error types in generated memory, including corresponding examples and error proportion of content. The error context is marked in red.  

<table><tr><td>Error Type</td><td>Past Dialogs</td><td>Generated Memory</td><td>Golden Memory</td><td>Prop.</td></tr><tr><td>Fabricated Facts</td><td>Bot: I like to walk to work instead of driving so I see animals.</td><td>The bot enjoys walking to work to see animals.</td><td>bot: I walk to work.</td><td>2.7%</td></tr><tr><td>Incorrect Relationships</td><td>User: I ended up giving in and getting my daughters the cat....Well when you have daughters you sort of give in to them. She named it Angie.</td><td>The user&#x27;s daughters now have a cat named Angie, which the user gave in to.</td><td>User: I got my daughters a cat. My cat is named Angie. My daughters named the cat.</td><td>3.2%</td></tr><tr><td>Missing Details</td><td>User: I just saw the best movie on netflix. Bot: What movie did you see? User: It&#x27;s a documentary called The Social Dilemma.</td><td>User: Enjoys watching TV, reading, and listening to music</td><td>User: I think the documentary The Social Dilemma is the best movie on Netflix.</td><td>3.9%</td></tr></table>

believe advanced memory mechanisms can achieve further improvement, which might be investigated in future works.

Our memory suffers from a few fact errors within an acceptable range (Q2).

One may doubt that the generated summaries might have serious factual inconsistency and error propagation issues. We argue that summarization is not a difficult task and some works find that LLM summaries exhibit better factual consistency and fewer hallucinations(Pu et al., 2023). To further address this con-

cern, we randomly choose 100 dialog samples and manually evaluate the quality of memory in the last session. Table 6 reports three error types found in our generated memory. 1) Fabricated facts refers to the memory containing some information that the dialog history cannot verify. In the first case, the bot walks to work not to see animals. 2) Incorrect relationships refers that the generated memory concludes error causal or references from history. In the second dialogue, the user gave in to her daughter, rather than that cat. 3) Missing details refers that the memory drops partial details of events. In the third scenario, the model ignores the user's favorite movie name and only roughly summarizes it as enjoying watching TV. Although there exist some mistakes in our generated memory, the incorrect/inaccurate information does not exceed  \(10\%\)  of the summaries content, indicating that most recursively generated summaries are trustworthy and usable to aid long- term dialog. Besides, extensive experiments (in Table 3 and Figure 5) validate the efficacy of our approach in utilizing generated summaries for long- term dialogs, which constitutes the primary contribution of this paper. Lastly, the above analysis also illustrates that our method needs to be strengthened in memorizing accurate dialog details. More advanced agent- based techniques, such as retrieval- enhanced methods (Zhang et al., 2023b), can be applied in the future.

Our plug- and- play method is effective for other small- scale and large- scale LLMs (Q3). To check whether the proposed recursively summarizing method is robust to other large language models to tackle long- term sessions, we evaluate the framework by employing "Llama2- 7B" and "text- davinci- 003" as the backbone models. The performances in long- context dialogue (session 5 of the MSC dataset) are shown in Figure 6, where the significant improvements confirm the robustness of our method upon different LLMs. We also believe that more powerful

![](images/ecdf24f6a65e0ab93d4aa228a67f76116bb425d56f86a59d532b0dfda2fec35e.jpg)  
Figure 6: The F1 score on responses when using other LLMs.

![](images/a8f093abddf047b7af411d5ce8f0bbd51887f77bca260ed6cf5f45df46e2bb14.jpg)  
Figure 7: The average number of tokens in generated responses.

language models, i.e., "text- davinci- 003" >"Llama2- 7b", indeed understand the context better, and generate more accurate memory and responses, thus leading to better improvements from the proposed mechanism.

Our method can be further enhanced by several labeled dialogs (Q4). We evaluate the few- shot performance of the proposed mechanism by the in- context technique. In precise, we utilize several dialogues with generated memory and labeled responses (ground truth), randomly sampled from the valid set, to prompt the response generator before the test inputs. Table 7 shows that even two labeled samples can bring obvious advantages under our framework, on both F1 and BLEU scores, indicating the potential of our framework. We analyze that the generated memory may contain a significant amount of speaker preference information, which undoubtedly increases the difficulty of generating replies. Therefore, labeled data is highly valuable for the proposed method, as it naturally guides the LLM in utilizing the memory effectively.

Table 7: The comparative results  \((\%)\)  on zero-shot and few-shot when using generated memory in MSC dataset.  

<table><tr><td rowspan="2">N-shot</td><td colspan="2">Session 4</td><td colspan="2">Session 5</td></tr><tr><td>F1</td><td>BLEU-1/2</td><td>F1</td><td>BLEU-1/2</td></tr><tr><td>Zero-shot</td><td>20.19</td><td>21.80/12.57</td><td>20.48</td><td>21.76/12.59</td></tr><tr><td>Two-shot</td><td>20.37</td><td>22.11/12.65</td><td>20.63</td><td>22.04/12.71</td></tr><tr><td>Three-shot</td><td>20.98</td><td>22.43/12.76</td><td>21.08</td><td>22.23/12.82</td></tr></table>

![](images/b3222ddd782a8eec114e6a15aeb669a312a4504513035ca93029345a08515534.jpg)  
Figure 8: Generated responses when using different methods in MSC dataset. Among them, our framework can obtain the up-to-date memory and incorporate it into generated response. For clarity, we omit other utterances and predicted memory unrelated to the current query.

## 6.5. Case Study

To check whether the memory incorporates long- range dialogue information into the response, we first compare the length of the response when using different methods. As shown in Figure 7, the average response length of using generated memory is about 15 tokens longer than the vanilla LLM (without memory) across all sessions. Furthermore, we take a sample from the generated responses to analyze the impact of memory on different methods. In the case of Figure 8, the user mentions "shopping addiction" at the current turn, referring to the bot's

habit of buying too many shoes. From the result, we can draw the following conclusions: (1) The retriever- based method (ChatGPT- DPR) and context- only LLM (ChatGPT) tend to focus on local (or closest) dialog information with long- context inputs. (2) Compared to using the golden memory, the generated memory is more fluent and coherent. It also explains why our method outperforms better than using golden memory directly, which has been observed in Table 5. (3) Compared to the competitive memory mechanism (MemoryBank), our method can iterate and update the memory promptly, keeping consistent with the ongoing conversation. (4) The proposed recursively summarized memory method indeed integrates long- term dialogue information into generated responses. In Figure 8, the latest memory (i.e., the bot's preference for basketball shoes) is understood and mentioned in the response.

## 6.6. Complementary to Existing Works

Our proposed method is a new memory mechanism for improving the long- range dialogue ability of LLMs, which is expected to complement existing works, including retrieval- based and input- expended methods. Here, we list two representative approaches and show the orthogonality.

Retrieval- based LLMs.. The effectiveness of retrieval- based methods on the MSC dataset can be observed in Table 3, illustrating their potential in long- range conversations. Here, we further explore the complementary to the proposed method. As shown in Table 8, retrieval- enhanced methods (ChatGPT- BM25 and ChatGPT- DPR) achieve further improvements than vanilla ChatGPT, showing the importance of recalling relevant information in long- range dialogs. Besides, using our framework could push these retrieval- based methods toward better performance,

Table 8: Complementarity between ours and retrieval-based methods, in terms of automatic and LLM's evaluation on session 5 of the MSC dataset.  

<table><tr><td>Method</td><td>F1</td><td>Coherence</td><td>Consistency</td></tr><tr><td>ChatGPT</td><td>19.41</td><td>75.00</td><td>75.48</td></tr><tr><td>ChatGPT-BM25 (k=5)</td><td>20.91</td><td>75.44</td><td>76.88</td></tr><tr><td>+ Our framework</td><td>21.81</td><td>84.44</td><td>90.68</td></tr><tr><td>ChatGPT-DPR (k=5)</td><td>20.97</td><td>78.60</td><td>79.20</td></tr><tr><td>+ Our framework</td><td>21.69</td><td>83.40</td><td>86.48</td></tr></table>

Table 9: Complementarity between ours and long-context LLMs, in terms of automatic and LLM's evaluation on session 5 of the MSC dataset.  

<table><tr><td>Method</td><td>F1</td><td>Coherence</td><td>Consistency</td></tr><tr><td>longlora-8k</td><td>14.02</td><td>42.44</td><td>62.04</td></tr><tr><td>longlora-8k + Our framework</td><td>15.77</td><td>53.41</td><td>68.60</td></tr><tr><td>ChatGPT</td><td>19.41</td><td>75.00</td><td>75.48</td></tr><tr><td>ChatGPT-16k</td><td>19.92</td><td>78.60</td><td>79.20</td></tr><tr><td>ChatGPT-16k+ Our framework</td><td>19.29</td><td>90.04</td><td>92.44</td></tr><tr><td>GPT-4o</td><td>20.35</td><td>87.70</td><td>82.00</td></tr><tr><td>GPT-4o + Our framework</td><td>21.02</td><td>91.12</td><td>93.29</td></tr></table>

gaining about  \(+0.8\%\)  of F1 scores. We explain that these retrieved utterances can be viewed as evidence of event details, which together with our generated memory enhances LLM's long- term dialog ability.

Long- context LLMs.. To process the entire context and reduce the information loss, many researchers try to extend the context length of LLMs via training from scratch, fine- tuning, or other specific algorithms (e.g., FlashAttention) (Dao et al., 2022). For example, LongLoRA (Chen et al., 2024b) extends Llama2 (Touvron et al., 2023) 7B from 4k context to 100k. Although the maximum length of the datasets used is no more than 4k, which accommodates most popular LLMs, we

still want to explore whether the proposed method is complementary to these length- extended models. Here, we utilize three popular LLMs with large context windows, i.e., "longlora- 8k"<sup>7</sup>, ChatGPT- 16k (the version "gpt- 3.5- turbo- 16k- 0613") and GPT- 4o<sup>8</sup> to verify the effectiveness of our proposed framework. To ensure the quality of generated memory, we use the ChatGPT "gpt- 3.5- turbo- 0301" as the memory iterator, and only apply the above long- context LLMs to finish the memory- based response generation. From the results in Table 9, we conclude that: 1) Increasing the maximum context length of LLMs indeed enhances the long- term ability of dialog, with obvious improvements in coherency and consistency, even when the input length is much smaller than the context window. 2) Even if the input length does not exceed the window size, applying our method to LLMs remains an effective way to recall information and maintain long- term connections. 3) With a stronger LLM as the backbone (e.g., GPT- 4o<sub>7</sub>, ChatGPT<sub>i</sub>, longloRA), the model achieves better performance after our enhancements. We explain that our recursive summaries help reorganize and digest past information efficiently, thereby enhancing the understanding of semantics in long- range conversations.

# 7. Conclusion

In this paper, we propose a simple and effective strategy by recursively summarizing to improve the long- term dialogue ability in LLMs. The experimental results indicate our generated memory can model long- term dependencies and prompt LLMs to generate highly consistent responses. Additional analysis indicates that the method is robust across different LLMs and can be further boosted in few- shot scenarios. Importantly, our method also shows strong complementary to both popular retrieval- based and long- context models.

Limitations. One of the primary limitations of our method is that it does not account for the cost associated with calling large models. This is a significant factor that cannot be overlooked in real- life applications, where computational resources and associated costs are often a constraint. Besides, while our generated memory is effective, it occasionally suffers from minor factual errors. These inaccuracies, though few, highlight an area for improvement that can be addressed in future research.

Future work. One promising direction for future work is to explore the effectiveness of our method in modeling long- context tasks beyond dialogue, such as story generation. Investigating how well our approach can handle different types of long- context tasks will provide deeper insights into its versatility and potential applications. Another avenue for future research is to optimize the performance of our summarization method by using a locally supervised, fine- tuned LLM. This approach could potentially reduce the reliance on expensive online APIs, making the method more accessible and cost- effective for broader use.

# 8. Acknowledgements

This work is supported by the National Natural Science Foundation of China (No.U2336202).

## Appendix A. Dataset Statistics

Appendix A. Dataset StatisticsThe statics of MSC and Carecall are shown in Table A.10 and Table A.11. The session ID  \(i\)  indicates there are  \(i - 1\)  history conversation sessions that happened before the last session. The "#Ave. Tokens" and "#Max. Tokens" refer to the average and maximum number of tokens of dialogs, respectively.

Table A.10:The statistics of MSC Dataset.  

<table><tr><td>Session ID</td><td>#Dialog</td><td>#Response</td><td>#Ave. Tokens</td><td>#Max. Tokens</td></tr><tr><td>Session 2</td><td>501</td><td>5,939</td><td>444.84</td><td>951</td></tr><tr><td>Session 3</td><td>501</td><td>5,924</td><td>810.19</td><td>1733</td></tr><tr><td>Session 4</td><td>501</td><td>5,940</td><td>1195.08</td><td>2234</td></tr><tr><td>Session 5</td><td>501</td><td>5,945</td><td>1598.78</td><td>2613</td></tr></table>

Table A.11:The statistics of Carecall Dataset.  

<table><tr><td>Session ID</td><td>#Dialog</td><td>#Response</td><td>#Ave. Tokens</td><td>#Max. Tokens</td></tr><tr><td>Session 2</td><td>2798</td><td>7826</td><td>285.03</td><td>692</td></tr><tr><td>Session 3</td><td>743</td><td>7693</td><td>459.63</td><td>1093</td></tr><tr><td>Session 4</td><td>674</td><td>7065</td><td>636.55</td><td>1418</td></tr><tr><td>Session 5</td><td>638</td><td>6553</td><td>809.45</td><td>1744</td></tr></table>

## Appendix B. Prompt Designs

The following are all prompts utilized in our experiments:

Context- only LLMs (Llama- 7B, ChatGLM- 6B, and ChatGPT): Table B.12 Retrieval- based LLMs (ChatGPT- BM25 & ChatGPT- DPR): Table B.13

- Retrieval-based LLMs enhanced by our framework (ChatGPT-BM25 + Our framework & ChatGPT-DPR + Our framework): Table B.14- Our method enhanced by in-context learning (Take the one-shot as an example): Table B.15

- LLM evaluations: Table B.16 and Table B.17

Table B.12: The prompt for the context-only LLM.  

<table><tr><td>Prompt</td><td>You are an advanced AI language model capable of engaging in personality-based conversations. Respond to the user based on the provided dialogue context. Craft a response that is natural and conversational.
Dialog context: [dialog]
The response to user is:</td></tr><tr><td>Output</td><td>[response]</td></tr></table>

Table B.13: The prompt for the retrieval-based LLM.  

<table><tr><td>Prompt</td><td>You are an advanced AI designed for engaging in natural, personality-based conversations.
You will be provided with dialogue memory, relevant historical context, and dialogue context.
The dialogue memory contains the personality, preferences, and experiences of the speakers
(the user and the assistant). When responding, consider maintaining a conversational and fluent tone.
Responses should be contextually relevant and aim to keep the conversation flowing.
Relevant context: [retrieved utterances]. Dialogue context: [dialog].
So the response to the user is:</td></tr><tr><td>Output</td><td>[response]</td></tr></table>

Table B.14: The prompt for the retrieval-based LLM enhanced by our framework.  

<table><tr><td>Prompt</td><td>You are an advanced AI designed for engaging in natural, personality-based conversations.
You will be provided with dialogue memory, relevant historical context, and dialogue context. The dialog memory contains the personality, preferences and experiences of speakers (the assistant and the user). When responding, consider maintaining a conversational and fluent tone. Responses should be contextually relevant and aim to keep the conversation flowing.
Relevant context: [retrieval utterances]
Memory: [latest memory]
Dialogue: [current context]</td></tr><tr><td>Output</td><td>[response]</td></tr></table>

Output [response]

Table B.15: The prompt for our method with in-context learning.  

<table><tr><td>Prompt</td><td>You are an advanced AI designed for engaging in natural, personality-based conversations. You will be provided with a memory, containing the personal preferences and experiences of speakers (the assistant and the user), as well as a dialogue context. When responding, consider maintaining a conversational and fluent tone. Responses should be contextually relevant, consistent with given memory, aiming to keep the conversation flowing. Your goal is to provide engaging and coherent responses based on the dialogue context provided. To help you understand this task, we provide 1 example below. EXAMPLE 1: The example memory is:
User: - Divorced - Raising one child - Immigrated from Britain last year - Metal worker Assistant: - Not married - Girlfriend has 2 kids - Works on mTurk, landscaping, sales, envelope stuffing, painting - Used to love winter, but has become intolerant of it
Works with a friend who owns &quot;John of all trades&quot;
The example dialogue context is:
User: Today was the hottest day I&#x27;ve ever experienced here in Florida!
So the response to the user is: do you enjoy the heat more than the cold in Britain?
The following is the case you need to test: The test memory is: [previous memory]
The test dialogue context is: [dialog] So the response to the user is:</td></tr></table>

<table><tr><td>Output</td><td>[response]</td></tr></table>

Table B.16: The prompt for single model evaluation.  

<table><tr><td colspan="2">Table B.16: The prompt for single model evaluation.</td></tr><tr><td colspan="2">You are an impartial judge. You will be shown a Conversation Context, Personality of Speakers and Assistant Response.
#Fluency: Please evaluate whether the Assistant&#x27;s response is natural, fluent, and similar to human communication, avoiding repetition and ensuring a diverse range of output.
#Consistency: Please evaluate whether the Assistant&#x27;s response is consistent with the information of persona list. Any deviation from the expected personality may indicate a lack of coherence.</td></tr><tr><td>Prompt</td><td>#Coherency: Please evaluate whether the Assistant&#x27;s response maintains a coherent and logical flow of conversation based on the evolving context. A response with good context coherence can understand and respond appropriately to changes in conversation topics, providing smooth and sensible interactions.
Conversation Context: [dialog] Personality: [persona] Assistant Response: [response]
Begin your evaluation by providing a short explanation, then you must rate the Assistant Response on an integer score of 1 (very bad) to 100 (very good) by strictly following this format: [[score]].</td></tr><tr><td>Output</td><td>[output]</td></tr></table>

### Table B.17: The prompt of pairwise model evaluation.

Hi! We are a group of researchers working on Artificial Intelligence. In this task, we will ask you to help us rate the assistant's responses. In the area below, you will first read:

Hi! We are a group of researchers working on Artificial Intelligence. In this task, we will ask you to help us rate the assistant's responses. In the area below, you will first read:1. A conversation context comes from two speakers (the user and bot)2. The personality of two speakers (the user and bot) extracted from past dialogs.3. Two responses from AI systems. Your task is to decide which response is better. There are several dimensions that you can think along. Consider the following questions:1. Is the response coherent? A response with good context coherence can understand and respond appropriately to changes in conversation topics, providing smooth and sensible interactions.2. Is the response consistent? Evaluate whether the response is consistent with the information of persona list. Any deviation from the expected personality may indicate a lack of consistency.3. Is the response natural and fluent? Please evaluate whether the response is natural, fluent, and similar to human communication, avoiding excessive repetition and ensuring a diverse range of output.

Prompt

Based on your aesthetics, which one do you prefer? For example, you might prefer one poem over another poem. Ultimately, you should decide which response is better based on your judgment and your own preference. There are four options for you to choose from:

1. Response 1 is better : If you think response 1 has an advantage, then choose this option. 
2. Response 1 is slightly better : Response 1 is very marginally better than response 2 and the difference is small.

3. Response 2 is slightly better : Response 2 is very marginally better than response 1 and the difference is small.

4. Response 2 is better : If you think response 2 has an advantage, then choose this option. There are cases where the difference between the two responses is not clear. In this case, you can choose the second or the third option. However, in general, we ask you to choose those options as few as possible.

response 1: [response1] response 2: [response2]

Output [response]

# References

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F.L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al., 2023. Gpt- 4 technical report. arXiv preprint arXiv:2303.08774.

An, C., Gong, S., Zhong, M., Li, M., Zhang, J., Kong, L., Qiu, X., 2023. L- eval: Instituting standardized evaluation for long context language models. arXiv preprint arXiv:2307.11088 abs/2307.11088.

Bae, S., Kwak, D., Kang, S., Lee, M.Y., Kim, S., Jeong, Y., Kim, H., Lee, S.W., Park, W., Sung, N., 2022. Keep me updated! memory management in long- term conversations, in: Findings of the Conference on Empirical Methods in Natural Language Processing.

Beltagy, I., Peters, M.E., Cohan, A., 2020. Longformer: the long- document transformer. arXiv preprint arXiv:2004.05150.

Brown, T.B., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., Agarwal, S., Herbert- Voss, A., Krueger, G., Henighan, T.J., Child, R., Ramesh, A., Ziegler, D.M., Wu, J., Winter, C., Hesse, C., Chen, M., Sigler, E., Litwin, M., Gray, S., Chess, B., Clark, J., Berner, C., McCandlish, S., Radford, A., Sutskever, I., Amodei, D., 2020. Language models are few- shot learners. arXiv preprint ArXiv:2005.14165.

Chen, N., Li, H., Huang, J., Wang, B., Li, J., 2024a. Compress to impress: Unleashing the potential of compressive memory in real- world long- term conversations. arXiv preprint arXiv:2402.11975.

Chen, S., Wong, S., Chen, L., Tian, Y., 2023. Extending context window of large language models via positional interpolation. arXiv preprint arXiv:2306.15595

Chen, Y., Qian, S., Tang, H., Lai, X., Liu, Z., Han, S., Jia, J., 2024b. Longora: Efficient fine- tuning of long- context large language models, in: the International Conference on Learning Representations.

Choi, E., On, K.W., Han, G., Kim, S., Nam, D.W., Jo, D., Rho, S.E., Kwon, T., Seo, M., 2023. Effortless integration of memory management into open- domain conversation systems. arXiv preprint arXiv:2305.13973.

Dao, T., Fu, D., Ermon, S., Rudra, A., Re, C., 2022. Flashattention: Fast and memory- efficient exact attention with io- awareness. Advances in Neural Information Processing Systems 35, 16344- 16359.

Deriu, J., Rodrigo, Á., Otegi, A., Echegoyen, G., Rosset, S., Agirre, E., Cieliebak, M., 2019. Survey on evaluation methods for dialogue systems. Artificial Intelligence Review 54, 755 - 810.

Dubois, Y., Li, C.X., Taori, R., Zhang, T., Gulrajani, I., Ba, J., Guestrin, C., Liang, P.S., Hashimoto, T.B., 2024. Alpacafarm: A simulation framework for methods that learn from human feedback. Advances in Neural Information Processing Systems 36.

GLM, T., Zeng, A., Xu, B., Wang, B., Zhang, C., Yin, D., Rojas, D., Feng, G., Zhao, H., Lai, H., et al., 2024. Chatglm: A family of large language models from glm- 130b to glm- 4 all tools. arXiv preprint arXiv:2406.12793.

Guu, K., Lee, K., Tung, Z., Pasupat, P., Chang, M., 2020. Retrieval augmented language model pre- training, in: the International Conference on Learning Representations.

Kann, K., Ebrahimi, A., Koh, J.J., Dudy, S., Roncone, A., 2022. Open- domain dialogue generation: What we can do, cannot do, and should do next, in: NLP4CONVAI.

Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., Yih, W.t., 2020. Dense passage retrieval for open- domain question answering, in: the Conference on Empirical Methods in Natural Language Processing.

Lee, G., Hartmann, V., Park, J., Papailiopoulos, D., Lee, K., 2023. Prompted llms as chatbot modules for long open- domain conversation, in: Findings of the Annual Meeting of the Association for Computational Linguistics.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Kuttler, H., Lewis, M., tau Yih, W., Rocktischel, T., Riedel, S., Kiela, D., 2020. Retrieval- augmented generation for knowledge- intensive nlp tasks, in: the Annual Conference on Neural Information Processing Systems.

Li, J., Galley, M., Brockett, C., Gao, J., Dolan, B., 2016. A diversity- promoting objective function for neural conversation models, in: Annual Conference of the North American Chapter of the Association for Computational Linguistics.

Li, J., Wang, M., Zheng, Z., Zhang, M., 2023. Loogle: Can long- context language models understand long contexts? arXiv preprint arXiv:2311.04939.

Liu, C.W., Lowe, R., Serban, I., Noseworthy, M., Charlin, L., Pineau, J., 2016.

How NOT to evaluate your dialogue system: An empirical study of unsupervised evaluation metrics for dialogue response generation, in: the Conference on Empirical Methods in Natural Language Processing.

Liu, N.F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., Liang, P., 2023. Lost in the middle: How language models use long contexts, in: Transactions of the Association for Computational Linguistics.

Lu, J., An, S., Lin, M., Pergola, G., He, Y., Yin, D., Sun, X., Wu, Y., 2023a. Memochat: Tuning llms to use memos for consistent long- range open- domain conversation. arXiv preprint arXiv:2308.08239.

Lu, Q., Qiu, B., Ding, L., Xie, L., Tao, D., 2023b. Error analysis prompting enables human- like translation evaluation in large language models: A case study on chatgpt. arXiv preprint.

Mazare, P.E., Humeau, S., Raison, M., Bordes, A., 2018. Training millions of personalized dialogue agents, in: the Conference on Empirical Methods in Natural Language Processing.

MetaAI, 2024. Llama 3. URL: https://llama.meta.com/llama3/. accessed: 2024- 08- 26.

Pan, A., Shern, C.J., Zou, A., Li, N., Basart, S., Woodside, T., Ng, J., Zhang, H., Emmons, S., Hendrycks, D., 2023. Do the rewards justify the means? measuring trade- offs between rewards and ethical behavior in the machiavelli benchmark, in: the International Conference on Machine Learning.

Papineni, K., Roukos, S., Ward, T., Zhu, W.J., 2002. Bleu: A method for automatic evaluation of machine translation, in: the Annual Meeting of the Associa-

tion for Computational Linguistics. URL: https://doi.org/10.3115/1073083.1073135.

Peng, K., Ding, L., Zhong, Q., Shen, L., Liu, X., Zhang, M., Ouyang, Y., Tao, D., 2023. Towards making the most of chatgpt for machine translation. arXiv preprint arXiv:2303.13780.

Pu, X., Gao, M., Wan, X., 2023. Summarization is (almost) dead. arXiv preprint arXiv:2309.09558 abs/2309.09558.

Research, F.A., 2019. Faiss. URL: https://ai.meta.com/tools/faiss/. accessed: 2024- 08- 26.

Robertson, S., Zaragoza, H., et al., 2009. the probabilistic relevance framework: Bm25 and beyond. Foundations and Trends® in Information Retrieval.

Rubin, O., Berant, J., 2024. Long- range language modeling with self- retrieval, in: Transactions of the Association for Computational Linguistics.

Shuster, K., Xu, J., Komeili, M., Ju, D., Smith, E.M., Roller, S., Ung, M., Chen, M., Arora, K., Lane, J., Behrooz, M., Ngan, W., Poff, S., Goyal, N., Szlam, A., Boureau, Y.L., Kambadur, M., Weston, J., 2022. Blenderbot 3: a deployed conversational agent that continually learns to responsibly engage. arXiv preprint arXiv:2208.03188.

Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., et al., 2023. Llama 2: Open foundation and fine- tuned chat models. arXiv preprint arXiv:2307.09288.

Wu, H., Wang, W., Wan, Y., Jiao, W., Lyu, M., 2023. Chatgpt or grammarly? evaluating chatgpt on grammatical error correction benchmark. arXiv preprint arXiv:2303.13648.

Wu, Q., Lan, Z., Qian, K., Gu, J., Geramifard, A., Yu, Z., 2022. Memformer: A memory- augmented transformer for sequence modeling, in: Findings of the Annual Meeting of the Association for Computational Linguistics.

Xu, J., Szlam, A., Weston, J., 2022a. Beyond goldfish memory: Long- term open- domain conversation, in: the Annual Meeting of the Association for Computational Linguistics.

Xu, X., Gou, Z., Wu, W., Niu, Z.Y., Wu, H., Wang, H., Wang, S., 2022b. Long time no see! open- domain conversation with long- term persona memory, in: Findings of the Annual Meeting of the Association for Computational Linguistics.

Zeng, A., Liu, X., Du, Z., Wang, Z., Lai, H., Ding, M., Yang, Z., Xu, Y., Zheng, W., Xia, X., Tam, W.L., Ma, Z., Xue, Y., Zhai, J., Chen, W., Zhang, P., Dong, Y., Tang, J., 2022. Glm- 130b: An open bilingual pre- trained model. arXiv preprint arXiv:2210.02414.

Zhang, C., D'Haro, L.F., Chen, Y., Zhang, M., Li, H., 2023a. A comprehensive analysis of the effectiveness of large language models as automatic dialogue evaluators, in: AAAI Conference on Artificial Intelligence.

Zhang, P., Xiao, S., Liu, Z., Dou, Z., Nie, J.Y., 2023b. Retrieve anything to augment large language models. arXiv preprint arXiv:2310.07554.

Zhang, S., Dinan, E., Urbanek, J., Szlam, A., Kiela, D., Weston, J., 2018. Personalizing dialogue agents: I have a dog, do you have pets too?, in: the Annual Meeting of the Association for Computational Linguistics.

Zhang, T., Liu, Y., Li, B., Zeng, Z., Wang, P., You, Y., Miao, C., Cui, L., 2022. History- aware hierarchical transformer for multi- session open- domain dialogue system, in: Findings of the Conference on Empirical Methods in Natural Language Processing.

Zhong, Q., Ding, L., Liu, J., Du, B., Tao, D., 2023. Can chatgpt understand too? a comparative study on chatgpt and fine- tuned bert. arXiv preprint arXiv:2302.10198.

Zhong, W., Guo, L., Gao, Q., Ye, H., Wang, Y., 2024. Memorybank: Enhancing large language models with long- term memory, in: Proceedings of the AAAI Conference on Artificial Intelligence, pp. 19724- 19731.

Zhou, J., Chen, Z., Wang, B., Huang, M., 2023. Facilitating multi- turn emotional support conversation with positive emotion elicitation: A reinforcement learning approach, in: Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 1714- 1729.