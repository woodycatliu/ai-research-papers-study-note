# AgentGym-RL: Training LLM Agents for Long-Horizon Decision Making through Multi-Turn Reinforcement Learning

Zhiheng Xi \(^{1*}\) , Jixuan Huang \(^{1*}\) , Chenyang Liao \(^{1*}\) , Baodai Huang \(^{1}\) , Honglin Guo \(^{1}\) , Jiaqi Liu \(^{1}\) , Rui Zheng \(^{1}\) , Junjie Ye \(^{1}\) , Jiazheng Zhang \(^{1}\) , Wenxiang Chen \(^{1}\) , Wei He \(^{1}\) , Yiwen Ding \(^{1}\) , Guanyu Li \(^{1}\) , Zehui Chen \(^{2}\) , Zhengyin Du \(^{2}\) , Xuesong Yao \(^{2}\) , Yufei Xu \(^{2}\) , Jiecao Chen \(^{2}\) , Tao Gui \(^{1,3}\) , Zuxuan Wu \(^{1,3}\) , Qi Zhang \(^{1}\) , Xuanjing Huang \(^{1}\) , Yu- Gong Jiang \(^{1}\)

\(^{1}\) Fudan University,  \(^{2}\) ByteDance Seed,  \(^{3}\) Shanghai Innovation Institute

Abstract

Developing autonomous LLM agents capable of making a series of intelligent decisions to solve complex, real- world tasks is a fast- evolving frontier. Like human cognitive development, agents are expected to acquire knowledge and skills through exploration and interaction with the environment. Despite advances, the community still lacks a unified, interactive reinforcement learning (RL) framework that can effectively train such agents from scratch—without relying on supervised fine- tuning (SFT)—across diverse and realistic environments. To bridge this gap, we introduce AgentGym- RL, a new framework to train LLM agents for multi- turn interactive decision- making through RL. The framework features a modular and decoupled architecture, ensuring high flexibility and extensibility. It encompasses a wide variety of real- world scenarios, and supports mainstream RL algorithms. Furthermore, we propose ScalingInter- RL, a training approach designed for exploration- exploitation balance and stable RL optimization. In early stages, it emphasizes exploitation by restricting the number of interaction, and gradually shifts towards exploration with larger horizons to encourage diverse problem- solving strategies. In this way, the agent develops more diverse behaviors and is less prone to collapse under long horizons. We perform extensive experiments to validate the stability and effectiveness of both the AgentGym- RL framework and ScalingInter- RL approach. Our agents match or surpass commercial models on 27 tasks across diverse environments. We offer key insights and will open- source complete AgentGym- RL framework—including code and datasets—to empower the research community in developing the next generation of intelligent agents.

![](images/090d2862dc7956881d4e6a114737aadd49c5693d4b3e540a2b9104388ac1228d.jpg)  
Figure 1 Left: Performance of proprietary models, open-source models, and our RL models across different agentic tasks. Right: Performance w.r.t model scale. Working in concert, our framework and method substantially enhances the open-sourced 7B-scale models' capabilities to a level that rivals or even surpasses top-tier proprietary large models.

# 1 Introduction

As Large Language Models (LLMs) have largely developed [2, 11, 41, 65, 84], their applications have extended from chatbots to autonomous agents that can handle long- horizon real- world tasks [39, 79]. Given a complex task, these agents interact with the environment, making a series of intelligent decisions to achieve the goal [95]. Analogous to human cognitive development, LLM agents are expected to acquire new knowledge and skills by actively exploring and interacting with the environment [42, 77]. Therefore, a natural approach is to train these agents using Reinforcement Learning (RL) [62].

Despite the progress of RL in areas like LLM reasoning [12, 18, 23, 66, 68, 76], most existing studies are restricted to single- turn tasks, where models are not required to engage in multi- turn interaction with complex environments [71]. While some recent efforts have attempted to extend RL to train LLM agents with multi- turn capabilities [3, 26, 30, 47, 71, 95], these works are limited in task complexity and environment diversity. Furthermore, they struggle with optimization stability and efficiency, resulting in suboptimal performance. Critically, the community currently lacks a unified, end- to- end, interactive multi- turn RL framework that is proven to be effective across a wide range of real- world scenarios and environments for training LLM agents without SFT as a preliminary step [12].

To bridge this gap, we introduce AgentGym- RL, a new framework for training LLM agents for multi- turn interactive decision- making through RL (Figure 2). Designed with a modular and decoupled architecture, AgentGym- RL enables clean separation of agents, environments, and learning algorithms—offering high extensibility and flexibility for diverse research needs. The framework supports mainstream RL algorithms, including PPO [53], GRPO [54], and REINFORCE+—[21], and is equipped with a wide range of real- world scenarios, e.g., web navigation [13, 87, 94], deep search [26, 72], digital games [15, 45], embodied tasks [7, 58], and scientific tasks [60, 69].

Furthermore, to tackle the exploration- exploitation trade- off and improve optimization stability in agent RL training, we propose ScalingInter- RL, a method that progressively extends the agent- environment interaction horizon during training. The core insight of this approach is to let the agent adapt to the environment in stages: beginning with exploitation to achieve reliable mastery of basic skills and simple tasks; subsequently increasing interaction horizon to promote exploration, refine behaviors, overcome shortcuts, and address more complex challenges. This progressive interaction- scaling strategy enables the agent to uncover richer interaction patterns (e.g., planning and reflection) and cultivate a broader set of skills and behaviors over time.

![](images/1118028b76bbcd3e96aa55e595fc54067b9482a82bca9e1dff9697ee5d6cabe2.jpg)  
Figure 2 Overview of the AgentGym-RL framework. It features a decoupled, flexible, and extensible architecture, comprising three primary modules—the environment, the agent, and the training module. It supports diverse scenarios, environments, and algorithms.

Our extensive experiments prove that AgentGym- RL delivers consistent and significant performance gains for agents across five tasks spanning 5 scenarios (Figure 1a). Open- source models (e.g., Qwen- 2.5- 7B [83]) trained with our framework and method achieved an average improvement of 33.65 points, matching—or even outperforming—larger commercial, closed- source models such as OpenAI- o3 [42] and Gemini- 2.5- Pro [10]. We also conducted numerous analytical experiments to provide key findings and insights, showing that scaling post- training and test- time compute have significant potential for developing agentic intelligence (Figure 1b). We hope our work will be a valuable contribution to the community's progress.

In summary, our main contributions are:

1. We propose and open-source AgentGym-RL, a new unified, modular, and flexible end-to-end RL framework designed for agent multi-turn interactive decision-making that includes a diversity of scenarios and environments.  
2. We propose ScalingInter-RL, a progressive interaction-scaling framework that incrementally adapts agents to their environment, facilitating the refinement of interaction patterns and skill acquisition. It enhances optimization stability in RL and achieves a balance between exploration and exploitation.  
3. Our extensive experiments demonstrate that AgentGym-RL and ScalingInter-RL deliver significant and consistent performance gains, matching or exceeding commercial models. In addition, we conduct empirical analyses that yield critical insights into agent design and operational paradigms, offering valuable guidance and resources for future research.

# 2 Preliminaries

## 2.1 Formulation

In this work, we study the multi- turn interactive decision- making tasks, i.e., agentic tasks, and we model them as a Partially Observable Markov Decision Process (POMDP)  \((\mathcal{U}, \mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{T}, r)\)  like Xi et al. [77], Zhou et al.

[95], where  \(\mathcal{U}, \mathcal{S}, \mathcal{O}, \mathcal{T}: \mathcal{S} \times \mathcal{A} \to \mathcal{S}, r: \mathcal{U} \times \mathcal{S} \to \mathbb{R}\)  represents the instruction space, the state space, the action space, the observation space, the deterministic state transition function, and the reward function, respectively.

Given a task instruction  \(u\in \mathcal{U}\)  , the agentic task requires the LLM agent to generate a sequence of actions  \(a_{k}^{T}\sim \pi_{\theta}(\cdot |s_{k})\)  based on its policy  \(\pi_{\theta}\)  parameterized by  \(\theta\)  to complete the given task, where  \(a_{k}\in \mathcal{A}\)  , and  \(s_k\in S\)  and  \(T\)  is the thinking path [88]. The agent then receives an observation  \(o_k\in \mathcal{O}\)  from the environment, and the state is then transitioned to  \(\mathcal{T}(s_k,a_k) = s_{k + 1}\)  . Finally after  \(N\)  turns of interactions, the environment  \(e\)  provides an outcome reward  \(r(\tau)\in [0,1]\)  to describe the completion of the multi- turn interactive decision- making tasks.

## 2.2 Policy Gradient

We utilize policy gradient methods [63] that optimizes our policy to maximize the expected cumulative reward. Unlike value- based methods that estimate the value function to derive a policy, policy gradient methods directly search the policy parameter space to find the optimal policy.

The core idea of policy gradient methods is to perform gradient ascent according to the objective  \(J(\theta)\)  , which is a function of the policy parameters  \(\theta\)  . Specifically,  \(J(\theta)\)  represents the expected cumulative reward the agent anticipates receiving when following policy  \(\pi_{\theta}\)  and interacting with the environment. Mathematically, this is expressed as the expectation of the total reward  \(r(\tau)\)  over trajectories  \(\mathcal{T}\)  generated by the policy:

![](images/2de22a45a07b9201ca42c72552916c8d516db1d3cbec302a9ae613493efd02b2.jpg)

To perform gradient ascent on  \(J(\theta)\)  , we require the policy gradient  \(\nabla_{\theta}J(\theta)\)  . In the vanilla policy gradient methods, the policy gradient can be estimated by:

![](images/dad371b2577c1c6547a4d69e547f5a13540e168f35a3243212205163f8b7f268.jpg)

where  \(\pi_{\theta}\)  is the policy parameterized by  \(\theta\) \(\mathcal{T}\)  represents a trajectory consisting of a sequence of states and actions,  \(a_{k}\)  and  \(s_k\)  are the action and state at time step  \(k\)  , and  \(r(\tau)\)  is the reward of the trajectory  \(\mathcal{T}\)

With the policy gradient estimated, we can optimize the parameters  \(\theta\)  of the policy  \(\pi_{\theta}\)  towards a direction of maximizing the expected cumulative reward with the gradient descent method by:

![](images/ca1de1dbeb5d6b7fd8a26e0a201ccb834a5412f27e9d220b7856c34ed9db80df.jpg)

where  \(\alpha \in [0,1]\)  is the learning rate. Mainstream RL algorithms for training LLMs include PPO [53], GRPO [54], REINFORCE++ [21], and RLOO [30] all of which are integrated into our framework.

# 3 The AgentGym-RL Framework

## 3.1 Architecture Overview

The AgentGym- RL framework is built on AgentGym [77], which provides several basic interactive environments for LLM agents. Our main extensions focus on three aspects:

1. Introducing more realistic environments and tasks (e.g., Deep Search tasks) to facilitate the development of more general agents. 
2. Incorporating a diverse set of online reinforcement learning algorithms covering both classical and state-of-the-art methods, to ensure consistency with current research frontiers, and offer an extensible foundation for the community to build upon. 
3. Implementing extensive engineering optimizations and agent-environment co-design, such as improved rollout parallelization and memory-leak mitigation.

Stage 1: Generate responses task_ids  \(=\)  expand(task_ids, sample_num) envs  \(=\)  create_env_clients(task_ids, "webarena", base_url)

Do in parallel: for (env, task_id) in zip(envs, task_ids): env.reset(task_id) handlers  \(= [\)  RolloutHandler().add_user_message(env.observe) for env in envs] for i in range(novsounds) prompts  \(=\)  [h.get_prompt) for h in handlers] responses  \(=\)  actor.generate(prompts) results  \(=\)  thread_safe_list Do in parallel: for (env, response) in zip (envs, responses): results.append(env.step(response)) for (h, r, res) in zip(handlers, responses, results): h.add_assistant_message(r) h.add_user_message(res.state) h.score  \(=\)  res.score if all_done(handlers): break

Stage 2: Prepare experience batch  \(=\)  gen_batch_from_rollout_handler(handlers) batch  \(=\)  actor.compute_log_prob(batch) batch  \(=\)  reference.compute_ref_log_prob(batch) batch  \(=\)  compute_advantages(batch, method  \(\equiv\)  "grp0")

Stage 3: Actor training actor.update_actor(batch)

![](images/07ae63c4e7d6ac0f2f323f2b1c3aafea4ce1c42ff48ed3b63b17de070ca603d9.jpg)  
Figure 3 Pseudocode demonstrating the example usage of our proposed framework (provided APIs marked orange), alongside a simplified theoretical diagram illustrating the agent - environment interaction and training pipeline.

As shown in Figure 2, the framework is organized into three main components:

The Environment module provides diverse scenarios via a standardized server- client architecture with unified HTTP protocols. The Agent module encapsulates the reasoning and decision- making process of agents in multi- turn interactions, with support for advanced mechanisms such as long- horizon planning and self- reflection. The Training module implements reinforcement learning pipelines and other training methods to optimize agent policies.

A more detailed description of our architecture is shown in Appendix A.

Given a batch of user queries and initial environment states, our framework initializes multiple independent environment clients in parallel. Each client interacts exclusively with a single agent, ensuring that executions are isolated and non- interfering. In every client, the agent generates an action that is executed in the environment, which then returns the updated state and reward for the next decision. A batch of such trajectories is collected concurrently across clients and subsequently fed into the training module to update the agent policy. The overall workflow and corresponding pseudocode of our framework are shown in Figure 3.

## 3.2 Features and Characteristics

In this section, we highlight the key features of the AgentGym- RL framework, covering four aspects: environment coverage, algorithm support, architectural advantages, and open- source contributions.

### 3.2.1 Diverse scenarios and environments.

To build LLM agents capable of multi- turn sequential decision- making for complex tasks in real- world environments, AgentGym- RL covers a broad spectrum of scenarios to comprehensively evaluate and foster the agent's ability to perceive its environment, long- term planning towards a goal, in- depth reasoning for making intelligent decisions, aptitude for reflection and correction when facing setbacks or making mistakes.

It includes:

It includes:- Web Navigation: Interacting with dynamic websites for tasks such as booking flights or extracting structured information, which requires agents to follow instructions, interpret textual and visual content, manipulate dynamic interfaces, and plan multi- step actions.- Deep Search: Performing multi- step, goal- directed queries with tools like browsers or Python interpreters, demanding strong information- seeking, multi- hop reasoning, long- term memory, and knowledge synthesis across sources.- Digital Games: Exploring and solving problems in interactive game- like environments, emphasizing real- time decision- making, strategy development, and adaptability to complex, dynamic settings.- Embodied Tasks: Controlling virtual or physical bodies for navigation, manipulation, and task execution, which calls for goal- directed planning, spatial reasoning, and robust perception- action grounding.- Scientific Tasks: Conducting experiments and solving problems in physically grounded, knowledge- intensive settings, requiring precise execution, dynamic interpretation of feedback, evidence- based reasoning, and iterative hypothesis refinement.

### 3.2.2 Comprehensive RL algorithm support.

While the original AgentGym supports only a limited set of training methods based on supervised fine- tuning, AgentGym- RL places online reinforcement learning at its core, empowering agents to dynamically explore and adapt through continuous interactions with the environment.

AgentGym- RL implements a suite of mainstream online RL algorithms: (1) PPO[53], a policy gradient method that improves training stability by clipping policy updates to prevent overly large steps, simplifying the trust- region concept from TRPO[52] while maintaining strong empirical performance; (2) GRPO[54], a PPO- derived method that normalizes rewards within groups of sampled actions per state and applies PPO- style clipping, reinforcing higher- performing actions relative to others; (3) RLOO[30], a REINFORCE variant that uses the average reward of other samples in the same batch as a per- sample baseline, reducing variance in policy gradient estimates; (4) REINFORCE++[21], an enhanced REINFORCE[73] algorithm that integrates PPO- style clipping and KL penalties, enabling more stable, simpler, and computationally efficient training without the need for a critic network.

Beyond online RL, the framework also supports a broad range of complementary training paradigms: SFT (Supervised Fine- Tuning)[44] is a standard training method where the agent learns to imitate expert demonstrations or golden trajectories step by step. DPO (Direct Preference Optimization)[50] is a variant of reinforcement learning that does not involve online interaction with the environment; instead, it learns from pre- collected preference pairs. For rejection sampling[35], we support methods like AgentEvol[77], which iteratively fine- tunes agents on trajectories generated by themselves and filtered based on task success.

### 3.2.3 Extensibility, Scalability, and Reliability.

Since AgentGym- RL is primarily designed to support large- scale reinforcement learning research and development for the community, we have carried out extensive engineering design, practice, and optimization to ensure the framework's extensibility, scalability, and reliability.

3.2.3 Extensibility, Scalability, and Reliability.Since AgentGym- RL is primarily designed to support large- scale reinforcement learning research and development for the community, we have carried out extensive engineering design, practice, and optimization to ensure the framework's extensibility, scalability, and reliability.- Extensibility. Extensibility is critical for supporting evolving research needs, allowing a framework to accommodate new environments, agent architectures, and training strategies without disrupting existing components. In our system, we adopt a modular and decoupled design, where the core components—Environment, Agent, and Training—are fully plug- and- play. As a result, researchers can easily incorporate new reinforcement learning objectives, reward functions, or sampling techniques, facilitating reproducible experiments and enabling exploration across a wide spectrum of algorithmic directions. For example, a new environment can be introduced by simply inheriting from BaseEnvClient and implementing the required methods such as reset(), step(), and observe(). Once implemented,

![](images/91a8193aa9958d7200f06db2d4b162dad6456a79582cd3d7431aa0d83f7e691a.jpg)  
Figure 4 An overview of the visualized user interface of our framework.

the new environment can be seamlessly used with existing agent architectures and training routines, enabling rapid experimentation without modifying any of the core framework components.

- Scalability. Recent advances in reinforcement learning increasingly rely on large-scale training, involving massive amounts of data and extended interaction sequences, which poses significant challenges for system scalability. To meet these challenges, a framework must be able to scale both in parallelism and interaction duration. We implemented a series of optimizations to achieve this. For example, we replaced WebArena's default single-browser-per-process design with a subprocess-based architecture, enabling a single server to manage multiple Chromium instances concurrently and thereby enhancing parallelism. Similarly, in SciWorld environment, we redesigned the environment's initialization and reset routines to support robust parallel creation and resetting of multiple instances, resolving previous failures in concurrent instantiation. In addition, we support longer training horizons through a full-reset interface in WebArena, which restores each web server to its initial state after every episode and mitigates state inconsistencies over time. Together, these optimizations allow our framework to scale effectively, facilitating large-scale training and enabling the research community to conduct a broad range of experiments.

- Reliability. Large scale multi-turn agent RL training poses a significant challenge to system reliability, that is, the ability to maintain consistent and reliable operation over long training periods. To achieve reliability, a framework must prevent failures that could disrupt training and ensure that critical resources are managed correctly. For instance, we optimized the memory management implementation in TextCraft. The original environment suffered from a memory leak in its recursive crafting_tree implementation, where redundant self-replication of a list structure caused exponential memory growth and eventual crashes during training. We resolved this issue by refactoring the recursion to eliminate redundant copies. Likewise, in SciWorld, a memory leak in its internal clock mechanism caused progressive memory accumulation and instability during extended rollouts. We addressed this issue by re-actoring the clock implementation to eliminate the leakage. Through our optimization, the framework provides a reliable environment for long-horizon training, ensuring consistent and uninterrupted operation over extended interaction sequences.

Collectively, these design and optimizations remove major engineering bottlenecks and make reproducible, large- scale RL experiments feasible across heterogeneous environments.

### 3.2.4 Open-source availability and community extensibility.

We design AgentGym- RL to foster a collaborative ecosystem where community contributions directly accelerate methodological progress while upholding verifiable research standards. AgentGym- RL is released as an open- source framework under permissive licensing, built upon established open- source frameworks such as veRL[56] and AgentGym[77] while maintaining full open- source availability. The framework provides comprehensive documentation, reproducible training pipelines, and standardized APIs to ensure research transparency and

![](images/e555c91b6986ce5cdbc3ad7c31b584ce3e5d0e04134cb799259b439588d9eca3.jpg)  
Figure 5 Illustration of the ScalingInter-RL approach. It allows the agent to adapt in stages: initially, by limiting interaction turns to prioritize exploitation, master basic skills, and solve easy tasks; later, by gradually increasing interactions to explore, avoid shortcuts, refine behavior, and tackle harder problems. Ultimately, this process trains a stronger agent.

practical adoption. Its modular architecture- which includes clearly defined extension points- enables the seamless integration of new environments and training methods, allowing the research community to extend functionality without disrupting the core workflows. To facilitate probing of data and model behaviors, we provide an interactive user interface, which streamlines empirical analysis for iterative development.

Usability, reproducibility and standardized evaluation. AgentGym- RL is designed to be user- friendly for the community. To systematically address reproducibility challenges in LLM- based reinforcement learning, AgentGym- RL institutes a standardized evaluation process and reproducible training pipelines. This design enforces uniform metrics and consistent experimental procedures to ensure fair comparisons. We provide easy- to- setup reproduction scripts that automate the entire workflow, from environment configuration to final evaluation. This design enables researchers to replicate prior findings with high fidelity and significantly lowers the barrier for building upon existing work, thereby promoting verifiable research standards.

Visualized user interface for observability and analysis. As shown in Figure 4, AgentGym- RL includes an interactive user interface designed to facilitate the probing of data and model behaviors. This tool streamlines empirical analysis by enabling researchers to perform a fine- grained, step- by- step inspection of an agent's decision- making process. It allows for the replay and examination of full interaction trajectories, visualizing the interplay between environmental observations, the agent's internal reasoning, and its resulting actions. This capability provides direct insights into model performance and failure modes, thereby accelerating the iterative development and debugging cycle.

## 3.3 Scalinglnter-RL: Progressive Scaling Interaction for Agent RL

Motivation and core insight. When assigned a task, an agent engages in iterative interactions with the environment- observing changes, reasoning about them, and executing subsequent actions. Through this cycle, the agent explores and experiments thoroughly, ultimately reaching the target state. This process is analogous to inference- compute scaling in LLM reasoning (as exemplified by OpenAI o1 and DeepSeek- R1), where additional computational resources are allocated at test time or during RL rollouts, allowing the model to reason more deeply before producing a final answer.

In comparison, we argue that beyond relying on internal reasoning to select the next action, agents should also expand their external interactions with the environment to ensure sufficient exploration and accumulate richer context toward the final goal- capturing a form of practice- driven insight. Yet, our preliminary experiments indicate that beginning with a large number of interaction turns often leads the model into redundant reasoning and unproductive actions, ultimately causing training collapse and degraded performance. Conversely, constraining the number of interactions to remain consistently small tends to narrow exploration and limits the agent's ability to master diverse patterns. This motivates us to propose our method.

Method. We draw inspiration from reinforcement learning for LLM reasoning [12, 18, 23, 66, 68, 76] and propose ScalingInter- RL, a training approach designed to balance exploration and exploitation while ensuring stable optimization. At its core is a progressive horizon- scaling strategy that adaptively adjusts the number of interaction turns during RL. The objective is to maximize the expected terminal reward under a constrained interaction budget:

![](images/ac95ab5aee2785ee7f4af77b1067bb1d6b14d8c1eedd6d122de6e1f9585a3f50.jpg)

where each trajectory  \(\tau = \left(a_0^T, o_1, a_1^T, \ldots , a_{K - 1}^T, o_K\right)\)  is sampled from the current policy  \(\pi_{\theta}\) , with  \(K\)  representing the total number of interaction turns. To enable the agent to rapidly learn effective behaviors under limited interaction resources, we begin training with a small horizon. By initially constraining the horizon, the agent learns to exploit its policy with maximum efficiency, achieving early proficiency on simple tasks, and laying the groundwork for deeper, long- horizon reasoning. As training progresses, we introduce a monotonic schedule  \(\{h_1 < h_2 < \dots < h_n\}\) , where  \(h_t\)  defines the maximum number of interaction turns allowed during phase  \(t\) :

![](images/a6ea13dcdebbe3db876dbe91f0b5ed5cdb70fd9e49ec1b6c70a2b7b0c54a7c59.jpg)

The horizon  \(h_t\)  is updated every  \(\Delta\)  training steps according to a curriculum schedule:

![](images/9408cd84d553e3ab0d10e6a730c0062dbddf22067d34a74eec15fb1722a214d9.jpg)

where  \(\delta_h\)  is an adaptive increment. As the horizon increases, the agent is incentivized to explore longer decision paths, facilitating the emergence of higher- order cognitive behaviors such as planning, reflection, and strategic backtracking, which is similar to the length- scaling phenomenon in RLVR for large reasoning models [3, 12, 37]. This phased scaling allows ScalingInter- RL to align the depth of interaction with the agent's evolving policy capabilities, bridging efficient early- stage exploitation and long- horizon generalization.

# 4 Experiments

4 ExperimentsTo verify the stability and effectiveness of the AgentGym- RL framework, we conduct extensive experiments across a diverse set of scenarios and environments. Our results demonstrate that LLM agents are capable of exploring and learning from scratch based solely on environment feedback, without the need for prior supervised fine- tuning, ultimately achieving performance that is comparable to, or even surpasses, that of commercial closed- source models such as OpenAI o3.

## 4.1 Experimental Settings

4.1 Experimental SettingsScenarios, Environments and Tasks. As mentioned before, we include five scenarios in AgentGym- RL. Specifically, for web navigation, we include WebArena [94] which is a realistic and reproducible web environment containing four distinct domains prevalent on the internet: online shopping, discussion forums, collaborative development, and business content management; for deep search, we include a RAG- based environment [19, 26, 28, 32, 38, 46, 67, 85] which enables LLMs to interact with search engines and solve multi- turn retrieval and reasoning tasks; for digital games, we include TextCraft [45], a text- based crafting game environment in which agents complete tasks via natural language interactions and task- based planning; for embodied tasks, we include BabyAI [7] which provides a controllable grid world with text instructions for embodied reasoning in simulated environments; for scientific tasks, we include SciWorld [69] which offers a scientific exploration simulator where agents conduct scientific experiments through text- driven reasoning cycles.

Baselines and backbone models. We leverage Qwen- 2.5- 3B and Qwen- 2.5- 7B [83] as our primary backbone models. We introduce the closed- source Gemini 2.5 Pro [10], OpenAI o3 [42], and GPT- 4o [22] as baselines. Additionally, we include the open- source DeepSeek- R1 [12], Qwen- 2.5- 72B [83], Llama- 3.1- 8B [14], and Llama- 3.1- 70B [14] models for comparison.

Detailed settings of each environment. We provide detailed descriptions of the tools, APIs, and experimental settings for each environment in Appendix B.

![](images/8c69f22297f06d6bd2210e4d96a6f2ff17016a22799a2144a2194188564d3d20.jpg)  
Figure 6 Training rewards in different environments.

## 4.2 Overall Results, Findings, and Insights

The main results are shown in Figure 1, Table 1, Table 2, Table 3, Table 4, and Table 5. In this section, we discuss the overall findings and insights.

Reinforcement learning generally improves agentic intelligence of open- source LLMs to the level of proprietary models. As illustrated in Figure 1, the AgentGym- RL- 7B model not only outperforms

other open- source models by a large margin but also demonstrates a clear lead in average success rate over leading closed- source models like GPT- 4o and Gemini- 2.5- Pro across five different scenarios. This achievement highlights our framework's effectiveness in enabling models to learn and make decisions in complex interactive tasks, successfully bridging the performance gap between open- source and proprietary models on advanced intelligent assignments.

ScalingInter- RL boosts performance significantly and consistently. Instead of relying on extensive hyperparameter tuning, we set the transition points between phases according to the total optimization steps of the original RL process. As shown in our results, ScalingInter- RL consistently outperforms the baseline across diverse environments and tasks. Notably, it delivers more than a  \(10\%\)  improvement on WebArena, bringing performance close to that of closed- source commercial models. On the TextCraft benchmark, it surpasses the base model by 30 points, achieving state- of- the- art results. These findings highlight the effectiveness of our approach in striking a balance between exploration and exploitation in reinforcement learning. As illustrated in Figure 6, experiments across different environments show that leveraging our AgentGym- RL framework with the ScalingInter- RL algorithm yields stable, sustained, and substantial reward improvements.

Large interaction budget accelerates early gains but ultimately leads to unstable training. As shown in Figure 7, we observe that using a larger maximum interaction turn (e.g., 10) achieves higher performance in the early stage compared to a shorter- turn setting (e.g., 5), but rapidly collapses as training progresses. This indicates that excessive exploration in early stages of training is not necessarily a good choice. Before establishing a solid foundation, the agent may perform unproductive and inefficient exploration, leading to the risk of training instability. By contrast, shorter rounds restrict early exploration but provide more stable learning signals, leading to more reliable long- term performance. Taken together, these contrasting dynamics between longer and shorter turns motivate our ScalingInter- RL method, which progressively extends the interaction horizon during training.

ScalingInter- RL demonstrates more stable and efficient training dynamics during RL optimization. As shown in Figure 7, our method is initially constrained by the number of interaction turns. Although it struggles to fully master difficult tasks at first, by exploiting foundational skills and knowledge it achieves a noticeable increase in rewards; later, as it engages in more turns of interaction and exploration with the environment, it shapes its reasoning paradigm and interaction behaviors, ultimately reaching a high level of performance. In contrast, RL with fewer turns yields diminishing returns in later stages and hits a performance ceiling, while RL with large interaction budget quickly collapses. Furthermore, just as is observed with RL for reasoning models [3], our gradual scaling of interactions dramatically reduces the computational resources and time required in RL phase, enabling more efficient optimization.

Post- training and test- time compute show higher scaling potential than model size. A key insight from our experiments is that strategic investment in post- training and test- time computation is more impactful than merely increasing a model's parameter count. Figure 1 (right) clearly illustrates this point: our ScalingInter- RL model, with only 7B parameters, achieves an average success rate of approximately  \(58.6\%\)  after being trained with our reinforcement learning framework. This performance not only surpasses other open- source models of similar size but also significantly outperforms much larger models like Llama3.1- 70B ( \(47\%\) ) and Qwen2.5- 72B ( \(43\%\) ), which have nearly ten times the parameters. This demonstrates that the performance improvement gained from simply scaling model size is limited and less steep compared to the gains from targeted post- training and inference- time computation using frameworks like AgentGym- RL.

Environmental structure is a key determinant for the efficiency of reinforcement learning. The effectiveness of AgentGym- RL varies depending on the nature of the environment and feedback. In simulated worlds with clear rules and explicit cause- and- effect, like TextCraft, BabyAI, and SciWorld, RL delivers the most significant performance leaps. On SciWorld's complex scientific reasoning tasks, our method boosted the model's score from 1.50 to 50.50, an astounding increase of nearly 50 points. In contrast, for more open- ended environments like WebArena and Deep Search, the performance gains from RL were rather moderate, though still positive. In these tasks, agents must navigate the complexities of real websites, handle multi- step crafting

![](images/b83e61a3a8b9f5eb2bb543d60151722bb4162515297109b6ee41fc6d564f6064.jpg)  
Figure 7 Training dynamics under different maximum interaction turns in Deep Search environment. Longer-turn settings (e.g., 10) initially achieve higher rewards by enabling richer exploration, but rapidly collapse due to high variance, credit assignment difficulties, and overfitting to spurious behaviors. Shorter turns (e.g., 5) yield more stable but less exploratory learning, leading to a performance ceiling. Our ScalingInter-RL method progressively increases the interaction horizon, and ultimately achieves higher and more efficient long-term performance.

Table 1 Evaluation results on WebArena benchmark. For each group, the best result is in bold, and the second-best is underlined. In the first row, G & R means GitLab and Reddit.  
![](images/0d3fac28d119c0984818ab80b38d7c4586efd943b94762821f83cb3faca19104.jpg)

plans, or process noisy information from search engines, making it more challenging to learn optimal strategies through trial and error. This suggests that while RL has broad applicability, it excels most in environments where clear feedback and successful pathways can be readily discovered through exploration.

## 4.3 Detailed Task Performance across Environments

Web navigation. As shown in Table 1, our models demonstrate highly competitive performance on the WebArena benchmark. In particular, the ScalingInter- 7B model achieves an overall accuracy of  \(26.00\%\)  significantly surpassing top- tier proprietary models like GPT- 4o (16.00%) and performing on par with larger

Table 2 Evaluation results on Deep Search benchmark. For each group, the best result is in bold, and the second-best is underlined. SearchR1-it-v0.3 baseline uses Search-R1-v0.3 models[25]  
![](images/f43dc241aa06e6b9b4555a434c601bcfbbc0e3ef92e5ff98199af2f98f642837.jpg)

models like DeepSeek- R1- 0528 (28.00%) and Gemini- 2.5- Pro (28.00%). Furthermore, another 7B model of ours, AgentGym- RL- 7B, also achieved an overall score of 16.00%, matching the performance of GPT- 4o. This strong overall performance is underpinned by ScalingInter- 7B's state- of- the- art proficiency in structured web navigation, where it achieved scores of 33.33% in Shopping and 26.67% in CMS, matching the best performance among all models in these categories. However, a significant performance gap remains when compared to the top- performing OpenAI o3 (34.00%) and o4- mini (36.00%), a disparity almost entirely concentrated in the "GitLab & Reddit" sub- task.

Deep search. The evaluation results in Table 2 show the importance of sophisticated reasoning abilities, where proprietary models—particularly the OpenAI 'o' series—currently set the performance benchmark, with OpenAI o3 achieving the highest overall score of 49.50. Against this competitive landscape, our models demonstrate exceptional performance. Specifically, our ScalingInter- 7B model achieved an excellent overall score of 38.25, not only surpassing top- tier proprietary models like GPT- 4o (26.75) and Gemini- 2.5- Pro (36.50) but also performing comparably to the strongest open- source model, DeepSeek- R1- 0528 (40.25). Its strengths are particularly salient in key domains: it achieved the highest score overall on the NQ task (52.00) and tied for first place on TriviaQA (70.00) with GPT- 4o. Furthermore, our AgentGym- RL- 7B (34.00) and AgentGym- RL- 3B (25.75) models also delivered strong results, each significantly outperforming open- source counterparts of similar or even larger scales. These results provide strong evidence that our reinforcement learning approach effectively unlocks the model's inherent reasoning capabilities, enabling it to reach or even exceed the performance of elite reasoning models in key scenarios—crucially, without the need for explicit additional long- reasoning.

Digital game. The TextCraft benchmark effectively assesses model capabilities across a wide spectrum of difficulty, as detailed in Table 3. At shallow depths (Depth 1), tasks are largely solved by top models. Conversely, the challenge becomes nearly insurmountable at maximum complexity (Depth 4), creating a

Table 3 Evaluation results on TextCraft benchmark. For each group, the best result is in bold, and the second-best is underlined.  
![](images/c4c01ae48f4db43868e8e86f9ee96b6cd78099fffb76169c951c804548e2b22c.jpg)

performance cliff for most agents. It is at these intermediate and highest difficulties that the efficacy of our models becomes particularly evident. Our ScalingInter- 7B model achieves an outstanding overall score of 91.00, placing it firmly among the top- tier proprietary and large open- source models (93.00- 94.00). Critically, it is one of only a few models to achieve a non- zero score at Depth 4, scoring 33.33 and demonstrating a unique robustness at maximum complexity. Our AgentGym- RL- 7B also excels with a score of 89.00, surpassing prominent models like GPT- 4o (83.00). The benefit of our RL training is especially dramatic for smaller models, where AgentGym- RL- 3B obtains a score of 75.00, vastly outperforming similarly- sized models like Qwen2.5- 3B- Instruct (14.00). These results showcase that our RL approach elevates our models to achieve competitive performance on complex, sequential decision- making tasks.

Embodied tasks. As demonstrated in Table 4, our RL model achieves state- of- the- art (SOTA) performance on the BabyAI benchmark, with an overall score of 96.67, which is competitive with the leading proprietary models such as o3 and o4- mini. Notably, our ScalingInter- 7B model attains the highest overall accuracy of 96.67%, outperforming top- tier models such as OpenAI o3 (94.44%) and GPT- 4o (86.67%). This exceptional performance is driven by ScalingInter- 7B's consistent mastery of diverse sub- tasks, achieving perfect scores of 100% in GoTo, ActionObjDoor (AOD), and SynthLoc, and strong results of 80% in both FindObjS7 (Find) and OneRoomS20 (Room). Similarly, our AgentGym- RL- 7B and AgentGym- RL- 3B models demonstrate robust capabilities, reaching overall accuracies of 92.22% and 93.33%, respectively, and securing perfect scores in GoTo and AOD tasks. Compared to other open- sourced models, such as Qwen3- 235B- A22B (87.78%) and DeepSeek- R1- 0528 (93.33%), our RL- based models maintain consistently high performance while effectively handling more challenging sub- tasks like Room and Find, where many LLMs exhibit notable variability. Overall, these results highlight the strength of our RL- based approaches, particularly ScalingInter- 7B, in achieving state- of- the- art performance on both structured navigation and object- interaction tasks in the BabyAI benchmark.

Table 4 Evaluation results on BabyAI benchmark. For each group, the best result is in bold, and the second-best is underlined. In the first row, AOD means ActionObjDoor, Find means FindObjS7, Room means OneRoomS20, SLoc means SynthLoc.  
![](images/8d0602cc8fc0752c301d41feed17d65e0cac774a8cb677ab93f1172e0ea7b4ad.jpg)

Scientific Scenario. Our experiments on the SciWorld benchmark, summarized in Table 5, demonstrate the advanced performance of our RL- trained models. Our ScalingInter- 7B model establishes a new state- of- the- art with an overall score of 57.00, which significantly surpasses all open- source and proprietary models, including the next- best proprietary model, OpenAI o3 (41.50). This superior performance is primarily attributed to high scores in the "Find" (88.64) and "Test- Cond" (55.42) sub- tasks. Furthermore, our AgentGym- RL- 7B model also shows strong capabilities, securing the second- highest overall score (50.50) and achieving the top score in "Test- Cond" (59.04). These results highlight the effectiveness of our RL method for training agents in exploration and procedural execution tasks. However, our findings also identify a critical limitation shared across all evaluated models. The "Chem- Mix" sub- task proved to be intractable, with every model, including our top performers, scoring zero. This uniform result indicates a systemic challenge for current language models in tasks requiring complex scientific reasoning and multi- step chemical simulation, marking this as a crucial area for future research.

# 5 Discussion

## 5.1 Test-Time Scaling for Agents

In this subsection, we investigate how agent performance changes as inference compute increases.

Scaling sequential inter action. First, we study how performance changes when the maximum number of interaction turns available to the model is raised. As shown in Figure 8, all models exhibit clear gains as the number of turns increases, which validates the insight behind our ScalingInter- RL approach—namely, agents must thoroughly explore the environment to shape their interaction and behavioral patterns. Furthermore, our trained agent consistently outperforms the baseline by a significant margin, further demonstrating the

Table 5 Evaluation results on SciWorld benchmark. For each group, the best result is in bold, and the second-best is underlined. In the first row, Test-Cond. means test-conductivity, Chem-Mix means chemistry-mix.  
![](images/c363e887fb6000d25c24d9ea21352d5567ace59cb1ca446b98341e0e8101eb22.jpg)

effectiveness of our method.

effectiveness of our method.Scaling parallel sampling. As shown in Figure 9, increasing the number of samples yields a marked improvement in Pass@K performance, signaling the downstream optimization potential of each model. Our model surpasses the baselines even with a small sampling budget, and as sampling increases, it continues to outperform the baseline in a stable and significant manner: for example, with 64 sampling attempts, our RL model achieves a  \(5.5\%\)  improvement in Deep Search environment and a  \(7.05\%\)  improvement in SciWorld environment over the untrained base model, showcasing its superior optimization capability.

## 5.2 Performance of Different RL Algorithm

We compare two mainstream RL algorithms for LLM post- training, i.e., GRPO and REINFORCE++. As shown in Table 6, our experiments reveal that GRPO consistently and substantially outperforms REINFORCE++ on the TextCraft, BabyAI, and Deep Search benchmarks. While model scaling from 3B to 7B parameters improves results for both algorithms, the superiority of GRPO is particularly stark: its 3B variant achieves higher scores than the 7B REINFORCE++ model. This finding points to a fundamental algorithmic advantage that is more impactful than sheer model scale.

The performance difference can be explained by how each algorithm handles the core difficulties of these tasks: vast exploration spaces and sparse rewards. The learning signal for REINFORCE++ is derived from full- episode Monte Carlo returns, which often results in high- variance gradients that are sensitive to stochastic successes and failures over long trajectories. In contrast, GRPO mitigates this instability by evaluating the relative merit of actions against a learned baseline. The focus on action advantage provides a more stable and robust gradient, facilitating more efficient exploration and credit assignment in complex, low- signal environments.

![](images/537983799fe2148cf0e9d26929b6c2a98a9a3e14770de0aa909e78be629d69cc.jpg)  
Figure 8 Scaling test interaction turns.

![](images/d5d697f7e5b79dc9157f0daace5f888725596dffb6c2b381dab82d5151b995bf.jpg)  
Figure 9 Pass@K performance.

## 5.3 Case Study

In this section, we present a series of case studies that highlight both the shortcomings of the base agent and the improvements achieved by our reinforcement learning models. Across navigation, compositional problem solving, and web interaction scenarios, the RL- trained agents consistently overcame unproductive behavioral loops, exhibited adaptive recovery strategies, and demonstrated more systematic task execution. To provide a balanced perspective, we also include two representative failure cases—in scientific reasoning and in efficient web interaction—that underscore areas where further refinement is needed. In the main text, we showcase the WebArena trajectory illustration and its corresponding visualization, while additional trajectory illustrations for other environments are provided in Appendix C.

Enhanced navigation. Figure 12 demonstrates a notable improvement in navigation capabilities within BabyAI environment. While the base agent exhibited suboptimal behavior characterized by repetitive movement patterns- going through previously explored locations without developing a strong search strategy for completion- the model trained through reinforcement learning manifested more effective exploration strategy. The RL agent demonstrated strategic backtracking capabilities, systematically exiting through doorways before selecting alternative pathways, ultimately accessing a green door that provided direct access to the target blue box. This highlights the RL model's superior ability in spatial reasoning and its ability to circumvent

Table 6 Evaluation results of different RL algorithms.  
![](images/b152562e4962dcb4f3213b80d367f7eaf9c65270a708a38917f5cf0c84c012c0.jpg)

unproductive behavioral loops.

Compositional Task Mastery. Figure 14 exemplifies the successful application of reinforcement learning to complex scientific task execution. The base agent exhibited fundamental deficiencies in task interpretation, misusing non- interactive objects and generating invalid actions. In contrast, the RL- optimized agent demonstrated comprehensive task understanding through its systematic approach: correctly identifying and manipulating a living thing (the banana tree), executing appropriate inventory management operations, navigating multi- room environments with obstacle resolution capabilities and successfully completing the objective by depositing the tree in the designated purple box. This highlights the RL agent's enhanced capabilities in reasoning, planning, and sequential task execution within compositional problem spaces.

Adaptive Web Navigation Strategies. Figure 10 and figurell illustrates the emergence of web navigation capabilities through reinforcement learning optimization. The base agent persistently interacted with nonresponsive interface elements, specifically engaging in repetitive clicking behaviors on ineffective targets without recognizing the futility of these actions. Our RL- trained agent exhibited markedly superior adaptive behavior: it successfully implemented error recovery mechanisms when encountering a "Page not found" error, subsequently utilizing the search box to locate the "pittsburgh" forum, identifying contextually relevant content within trending posts, and completing the subscription task successfully- - demonstrating enhanced robustness in error handling, purposeful navigation strategies, and the ability to maintain task focus while avoiding unproductive behavioral patterns.

Limitations in Scientific Scenario. Figure 15 reveals fundamental procedural execution failures that persist in SciWorld task completion despite the RL agent's ability to reach task- relevant game states. These instances exemplify two distinct failure modalities: first, when confronted with interaction failures requiring systematic debugging, the agent inappropriately substitutes direct factual recall for the intended experimental procedure; second, the agent demonstrates insufficient systematic exploration, as evidenced by its premature task termination after navigating to the outdoor environment and focusing only on the chameleon egg rather than analyzing all available animals that the task demands. These failures collectively indicate that the model lacks the deep procedural understanding necessary for executing rigorous scientific comparative analyses.

Over- Interaction Patterns in Web Navigation. Figure 16 demonstrates a prevalent failure mode of excessive and inefficient interaction sequences during web navigation tasks. Despite successfully navigating to the correct target websites in both illustrated cases, the RL agent engages in superfluous interaction patterns- - including redundant clicking, unnecessary hovering, and excessive scrolling- - that impede successful information extraction from the target pages. These behavioral patterns suggest that the reinforcement learning process failed to instill the precision and efficiency required for optimal task completion, indicating a gap between state- reaching capabilities and effective action selection within those states.

![](images/3a9cd188a4e30cac64e93b8c0e543f412c64e6b30e997a269c14baed8a646261.jpg)  
Figure 10 Smart case of our RL agent compared to the Base Model on WebArena task.

# 6 Related Work

Developing agents with large language models. With the advancement of large language models [1, 2, 65], researchers have begun using them to build agents capable of multi- turn intelligent decision- making [79, 86]. The predominant approaches rely on prompting to guide the model to invoke tools [48, 90], augmented with mechanisms such as self- reflection [51, 57, 78, 82], long- horizon planning [36, 40, 45, 61], and self- correction [29, 31]. Some work constructs workflows that assign different roles to multiple LLMs [17, 20, 34, 64, 74], each playing a specialized part in task completion. However, these methods typically depend on powerful proprietary models (e.g., OpenAI o3) and do not train the underlying models to evolve into agents with intrinsic agentic capabilities. Another line of work gathers large- scale expert trajectories and trains agents to mimic experts step by step [4, 6, 91, 92], thereby acquiring abilities such as API invocation, planning, and

![](images/43f8e49cd097c36ee6e3b6db5494f360e3b27c9680cb6d72d92d9b45e5346714.jpg)  
Figure 11 Trajectory visualization of our RL agent on Webarena task.

self- reflection. However, this approach is expensive, difficult to scale, and the model struggles to self- improve through interactions with the environment.

Reinforcement learning for large language model agents. Reinforcement learning has become a crucial post- training technique for large language models, enabling alignment with human preferences [5, 24, 43, 80, 93], enhancing reasoning capabilities [12, 18, 23, 49, 68, 76], and serving as a new scaling dimension [12]. Representative algorithms include PPO [53], GRPO [54], REINFORCE++ [21], RLOO [30], and others. However, most existing works—such as DeepSeek- RL—are confined to single- turn, static tasks in which models do not engage in multi- turn interactions with complex environments. Recent work has used RL to train agents for self- reflection [82], tool use [89], and even long- horizon interactions [3, 26, 30, 47, 71, 95]. However, these methods often struggle with scalable deployment due to limited task complexity and environment diversity, and they frequently encounter optimization instability that hinders performance. To overcome these challenges, we propose a unified, end- to- end RL framework spanning diverse real- world environments for training models in multi- turn decision- making without requiring SFT as a preliminary step. We further introduce ScalingInter- RL, an interaction- scaling technique that stabilizes optimization and boosts the agent's final performance.

Scaling Inference Compute for language models. OpenAI o1 and DeepSeek- R1 have shown that increasing compute during inference (both at test time and during RL rollouts) can yield strong scaling effects [12, 23, 59, 75]. Researchers have also explored various approaches to achieve similar gains—such as long- chain- of- thought reasoning [59, 78], majority voting [33, 70], best- of- N sampling [9, 27], beam search [81, 97], and Monte Carlo tree search [8, 16]. However, in the field of LLM agents, few works discuss how to scale inference compute. Zhu et al. [96] explore various test- time scaling strategies in agents and achieve significant gains, yet they do not investigate inference scaling in RL. The closest work may be TTI [55], which uses rejection sampling to teach agents to allocate more compute in interactions instead of thinking on web navigation tasks. In contrast, our approach employs mainstream on- policy RL algorithms—such as GRPO and REINFORCE++—and does not constrain the agent to use compute only for thinking or acting. Instead, we simply scale interactions and let the model decide how best to allocate its additional compute. Our method progressively grants the agent more exploration capacity, enabling it over time to better adapt to its environment, acquire more complex skills, and master more challenging tasks.

# 7 Conclusion and Future Work

In this work, we introduced AgentGym- RL, a novel and unified reinforcement learning framework designed to train LLM- based agents with long- horizon, multi- turn decision- making capabilities. The framework features diverse environments and tasks, supports mainstream RL algorithms, and is highly extensible—offering the community a practical and powerful toolkit. Additionally, we proposed the ScalingInter- RL method to progressively enhance agents' interactive intelligence in a staged manner. Extensive experiments demonstrate the effectiveness of both the framework and the method. However, several important directions remain for future exploration:

Developing agents with generalization and transfer capabilities. Currently, our trained agents perform well within in- domain settings. A key challenge going forward is to enable agents to adapt seamlessly to novel environments and unfamiliar tools while maintaining high performance.

Scaling RL training to longer- horizon and more realistic, physically grounded tasks. Most existing studies—including ours—focus on relatively simple digital tasks. However, real- world tasks are often longer- horizon, more complex, and grounded in the physical world. These tasks demand that agents process richer sensory inputs and reason over significantly larger action spaces, introducing new challenges for both reinforcement learning training and test- time interaction.

Advancing multi- agent reinforcement learning. Our current framework primarily targets single- agent training. However, multi- agent architectures open up new possibilities and may lead to stronger performance. At the same time, they introduce additional uncertainty and pose greater demands on both infrastructure and algorithm design.

# Acknowledgement

This work was supported by Huawei Ascend AI processors. We sincerely thank Huawei for providing the computing resources that made this research possible.

# References

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. Gpt- 4 technical report. arXiv preprint arXiv:2303.08774, 2023. [2] AI Anthropic. The claude 3 model family: Opus, somnet, haiku. Claude- 3 Model Card, 1(1):4, 2024. [3] Shiyi Cao, Sumanth Hegde, Dacheng Li, Tyler Griggs, Shu Liu, Eric Tang, Jiayi Pan, Xingyao Wang, Akshay Malik, Graham Neubig, Kourosh Hakhamaneshi, Richard Liaw, Philipp Moritz, Matei Zaharia, Joseph E. Gonzalez, and Ion Stoica. Skyrl- v0: Train real- world long- horizon agents via reinforcement learning, 2025.

[4] Baian Chen, Chang Shu, Ehsan Shareghi, Nigel Collier, Karthik Narasimhan, and Shunyu Yao. Fireact: Toward language agent fine- tuning. CoRR, abs/2310.05915, 2023. doi: 10.48550/ARXIV.2310.05915. URL https://doi.org/10.48550/arXiv.2310.05915.

[5] Lu Chen, Rui Zheng, Binghai Wang, Senjie Jin, Caishuang Huang, Junjie Ye, Zhihao Zhang, Yuhao Zhou, Zhiheng Xi, Tao Gui, Qi Zhang, and Xuanjing Huang. Improving discriminative capability of reward models in RLHF using contrastive learning. In Yaser Al- Onaizan, Mohit Bansal, and Yun- Nung Chen, editors, Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP 2024. Miami, FL, USA, November 12- 15, 2024, pages 15270- 15283. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024. EMNLP- MAIN.852. URL https://doi.org/10.18653/v1/2024. emnlp- main.852.

[6] Zebui Chen, Koikun Liu, Qiuher Wang, Wenwei Zhang, Jianqing Liu, Dahua Lin, Kai Chen, and Feng Zhao. Agent- flat: Designing data and methods of effective agent tuning for large language models. In Lun- Wei Ku, Andre Martins, and Vivek Srikumar, editors, Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11- 16, 2024, pages 9354- 9366. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024. FINDINGS- ACL.557. URL https://doi.org/10.18653/v1/2024. findings- acl.557.

[7] Maxime Chevalier- Boisvert, Dzmitry Bahdanau, Salem Lahlou, Lucas Willems, Chitwan Saharia, Thien Huu Nguyen, and Yoshua Bengio. Babyai: A platform to study the sample efficiency of grounded language learning. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6- 9, 2019. OpenReview.net, 2019. URL https://openreview.net/forum?id=rJeXCoOYX.

[8] Yizhou Chi, Yizhang Lin, Sirui Hong, Duyi Pan, Yayong Fei, Guanghao Mei, Bangbang Liu, Tianqi Pang, Jacky Kwok, Ceyao Zhang, Bang Liu, and Chenglin Wu. SELA: tree- search enhanced LLM agents for automated machine learning. CoRR, abs/2410.17238, 2024. doi: 10.48550/ARXIV.2410.17238. URL https://doi.org/10.48550/arXiv.2410.17238.

[9] Yinlam Chow, Guy Tennenholtz, Izzeddin Gur, Vincent Zhuang, Bo Dai, Aviral Kumar, Rishabh Agarwal, Sridhar Thiagarajan, Craig Boutilier, and Aleksandra Faust. Inference- aware fine- tuning for best- of- n sampling in large language models. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24- 28, 2025. OpenReview.net, 2025. URL https://openreview.net/forum?id=77gQdQdhE7.

[10] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

[11] DeepSeek- AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J. L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jiawei Wang, Jin Chen, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, Junxiao Song, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Litong Wang, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng Zhang, Qiancheng Wang, Qihao Zhu, Qinyu Chen, Qiushi Du, R. J. Chen, R. L. Jin, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runxin Xu, Ruoyu Zhang, Ruyi Chen, S. S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu, Shengfeng Ye, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shufeneng Zhou, Shuting Pan, T. Wang, Tao Yun, Tian Pei, Tianyu Sun, W. L. Xiao, and Wangding Zeng. Deepseek- v3 technical report. CoRR, abs/2412.19437, 2024. doi: 10.48550/ARXIV.2412.19437. URL https://doi.org/10.48550/arXiv.2412.19437.

[12] DeepSeek- AI, Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, Bing Xue, Bingxuan Wang, Bochao Wu, Bei Feng, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Hui Li, Jianzhong Guo, Jiashi Li, Jiawei Wang, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, J. L. Cai, Jiaqi Ni, Jian Liang, Jin Chen, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Liang Zhao, Litong Wang, Liyue Zhang, Lei Xu, Leyi Xia, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Meng Li, Miaojun Wang, Mingming Li, Ning Tian, Panpan Huang, Peng Zhang, Qiancheng Wang, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, R. J. Chen, R. L. Jin, Ruyi Chen, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng

Ye, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, and S. S. Li. Deepseek- r1: Incentivizing reasoning capability in lhs via reinforcement learning. CoRR, abs/2501.12948, 2025. doi: 10.48550/ARXIV.2501.12948. URL https://doi.org/10.48550/arXiv.2501.12948.

[13] Xiang Deng, Yu Gu, Boyuan Zheng, Shijie Chen, Samual Stevens, Boshi Wang, Huan Sun, and Yu Su. Mind2web: Towards a generalist agent for the web. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/5950bf290a1570ea401bf98882128160- A bstract- Datasets_and_Benchmarks.html.

[14] Abhimanya Dubey, Abhinav Joubri, Abhinav Pandey, Abhishak Kadian, Ahmad Al Dahla, Aisha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikoladis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia- Olano, Diego Perno, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel M. Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jena Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heufeld, Kevin Stone, and et al. The luna 3 herd of models. CoRR, abs/2407.21783, 2024. doi: 10.48550/ARXIV.2407.21783. URL https://doi.org/10.48550/arXiv.2407.21783.

[15] Linxi Fan, Guanzhi Wang, Yunfan Jiang, Ajay Mandlekar, Yuncong Yang, Haoyi Zhu, Andrew Tang, De- An Huang, Yuke Zhu, and Anima Anandkumar. Minedojo: Building open- ended embodied agents with internet- scale knowledge. In Sammi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/74a67268c5ccc5910f64938cac4526a90- Abstract- Datasets_and_Benchmarks.html.

[16] Bingzheng Gan, Yufan Zhao, Tianyi Zhang, Jing Huang, Yusu Li, Shu Xian Teo, Changwang Zhang, and Wei Shi. MASTER: A multi- agent system with LLM specialized MCTS. In Luis Chiruzzo, Alan Pitter, and Lu Wang, editors, Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL 2025 - Volume 1: Long Papers, Albuquerque, New Mexico, USA, April 29 - May 4, 2025, pages 9409- 9426. Association for Computational Linguistics, 2025. doi: 10.18653/V1/2025. NAACL- LONG.476. URL https://doi.org/10.18653/v1/2025. naacl- long.476.

[17] Honglin Guo, Kai Lv, Qipeng Guo, Tianyi Liang, Zhiheng Xi, Demin Song, Qi Zhang, Yu Sun, Kai Chen, Xipeng Qiu, and Tao Gui. Critiq: Mining data quality criteria from human preferences. CoRR, abs/2502.19279, 2025. doi: 10.48550/ARXIV.2502.19279. URL https://doi.org/10.48550/arXiv.2502.19279.

[18] Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, Siyuan Li, Liang Zeng, Tianwen Wei, Cheng Cheng, Bo An, Yang Liu, and Yahui Zhou. Skywork open reasoner 1 technical report. CoRR, abs/2505.22312, 2025. doi: 10.48550/ARXIV.2505.22312. URL https://doi.org/10.48550/arXiv.2505.22312.

[19] Xanh Ho, Anh- Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing A multi- hop QA dataset for comprehensive evaluation of reasoning steps. In Donia Scott, Nuria Bel, and Chengqing Zong, editors, Proceedings of the 28th International Conference on Computational Linguistics, COLING 2020, Barcelona, Spain (Online), December 8- 13, 2020, pages 6609- 6625. International Committee on Computational Linguistics, 2020. doi: 10.18653/V1/2020. COLING- MAIN.580. URL https://doi.org/10.18653/v1/2020. coling- main.580.

[20] Sirui Hong, Mingchen Zhuge, Jonathan Chen, Xiawu Zheng, Yuheng Cheng, Jinlin Wang, Ceyao Zhang, Zili Wang, Steven Ka Shing Yau, Zijuan Lin, Liyang Zhou, Chenyu Ran, Lingfeng Xiao, Chenglin Wu, and Jürgen Schmidhuber. Metagpt: Meta- programming for A multi- agent collaborative framework. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7- 11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=VtmBAGCN7o.

[21] Jian Hu. REINFORCE++: A simple and efficient approach for aligning large language models. CoRR, abs/2501.03262, 2025. doi: 10.48550/ARXIV.2501.03262. URL https://doi.org/10.48550/arXiv.2501.03262. [22] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alax Hayes, Alec Radford, et al. Gpt- 4o system card. arXiv preprint arXiv:2410.21276, 2024. [23] Aaron Jaech, Adam Kalai, Adam Lerer, Adam Richardson, Ahmed El- Kishky, Aiden Low, Alec Helyar, Aleksander Madry, Alex Beutin, Alex Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16520, 2024. [24] Jiaming Ji, Tianyi Qiu, Boyuan Chen, Borong Zhang, Hantao Lou, Kaile Wang, Yawen Duan, Zhonghao He, Jiayi Zhou, Zhaowei Zhang, Fanzhi Zeng, Kwan Yee Ng, Juntao Dai, Xuehai Pan, Aidan O'Gara, Yingshan Lei, Hua Xu, Brian Tse, Jie Fu, Stephen McAleer, Yaodong Yang, Yizhou Wang, Song- Chun Zhu, Yike Guo, and Wen Gao. AI alignment: A comprehensive survey. CoRR, abs/2310.19852, 2023. doi: 10.48550/ARXIV.2310.19852. URL https://doi.org/10.48550/arXiv.2310.19852. [25] Bowen Jin, Jinsung Yoon, Priyanka Kargupta, Sercan O. Arik, and Jiawei Han. An empirical study on reinforcement learning for reasoning- search interleaved LLM agents. CoRR, abs/2505.15117, 2025. doi: 10.48550/ARXIV.2505. 15117. URL https://doi.org/10.48550/arXiv.2505.15117. [26] Bowen Jin, Hansi Zeng, Zhenrui Yue, Dong Wang, Hamed Zamani, and Jiawei Han. Search- r1: Training llms to reason and leverage search engines with reinforcement learning. CoRR, abs/2503.09516, 2025. doi: 10.48550/ARXIV.2503.09516. URL https://doi.org/10.48550/arXiv.2503.09516. [27] Yuu Jinnai, Tetsuro Morimura, Kaito Ariu, and Kenshi Abe. Regularized best- of- n sampling to mitigate reward hacking for language model alignment. CoRR, abs/2404.01054, 2024. doi: 10.48550/ARXIV.2404.01054. URL https://doi.org/10.48550/arXiv.2404.01054. [28] Mandar Joshi, Eunsol Choi, Daniel S. Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. In Regina Barzilay and Min- Yen Kan, editors, Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, ACL 2017, Vancouver, Canada, July 30 - August 4, Volume 1: Long Papers, pages 1601- 1611. Association for Computational Linguistics, 2017. doi: 10.18653/V1/P17- 1147. URL https://doi.org/10.18653/v1/P17- 1147. [29] Ryo Kamoi, Yusen Zhang, Nan Zhang, Jiawei Han, and Rui Zhang. When can llms Actually correct their own mistakes? A critical survey of self- correction of llms. Trans. Assoc. Comput. Linguistics, 12:1417- 1440, 2024. doi: 10.1162/TACL_a_00713. URL https://doi.org/10.1162/tacl_a_00713. [30] Wouter Kool, Herke van Hoof, and Max Welling. Buy 4 REINFORCE samples, get a baseline for free! In Deep Reinforcement Learning Meets Structured Prediction, ICLR 2019 Workshop, New Orleans, Louisiana, United States, May 6, 2019. OpenReview.net, 2019. URL https://openreview.net/forum?id=r11gTGL5DE. [31] Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, John D. Co- Reyes, Avi Singh, Kate Baumli, Shariq Iqbal, Colton Bishop, Rebecca Roelofs, Lei M. Zhang, Kay McKinney, Disha Shrivastava, Cosmin Paduraru, George Tucker, Doina Precup, Feryal M. P. Behbahani, and Aleksandra Faust. Training language models to self- correct via reinforcement learning. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24- 28, 2025. OpenReview.net, 2025. URL https://openreview.net/forum?id=CjwERcAU7w. [32] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur P. Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming- Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. Natural questions: a benchmark for question answering research. Trans. Assoc. Comput. Linguistics, 7:452- 466, 2019. doi: 10.1162/TACL_a_00276. URL https://doi.org/10.1162/tacl_a_00276. [33] Junyou Li, Qin Zhang, Yangbin Yu, Qiang Fu, and Decheng Ye. More agents is all you need. Trans. Mach. Learn. Res., 2024, 2024. URL https://openreview.net/forum?id=bgzUSZ8aeg. [34] Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Shuming Shi, and Zhaopeng Tu. Encouraging divergent thinking in large language models through multi- agent debate. In Yaser Al- Onaizan, Mohit Bansal, and Yun- Nung Chen, editors, Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP 2024, Miami, FL, USA, November 12- 15, 2024, pages 17889- 17904. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024. EMNLP- MAIN.992. URL https://doi.org/10.18653/v1/2024. emnlp- main.992. [35] Tianqi Liu, Yao Zhao, Rishabh Joshi, Misha Khalman, Mohammad Saleh, Peter J. Liu, and Jialu Liu. Statistical rejection sampling improves preference optimization. In The Twelfth International Conference on Learning

Representations, ICLR 2024, Vienna, Austria, May 7- 11, 2024. OpenReview.net, 2024. URL https://openrevi ew.net/forum?id  \(\equiv\)  xbjSwwrQ0e.

[36] Zhiwei Liu, Weiran Yao, Jianguo Zhang, Le Xue, Shelby Heinecke, Rithesh Murthy, Yihao Feng, Zeyuan Chen, Juan Carlos Niebles, Devansh Arpit, Ran Xu, Phil Mui, Huan Wang, Caiming Xiong, and Silvio Savarese. BOLAA: benchmarking and orchestrating llm- augmented autonomous agents. CoRR, abs/2308.05960, 2023. doi: 10.48550/ARXIV.2308.05960. URL https://doi.org/10.48550/arXiv.2308.05960. [37] Michael Luo, Sijun Tan, Justin Wong, Xiaoxiang Shi, William Y. Tang, Manan Roongta, Colin Cai, Jeffrey Luo, Tianjun Zhang, Li Erran Li, Raluca Ada Popa, and Ion Stoica. Deepscaler: Surpassing o1- preview with a 1.5b model by scaling rl. https://pretty- radio- b75. nction.site/DeepScaleR- Surpassing- 01- Preview- with- a 1.5b- Model- by- Scalging- NL- 10681992- 1468005bedata3202134c2, 2025. Nation- Blog. [38] Alex Mallen, Akari Asai, Victor Zhong, Rajarshi Das, Hannaneh Hajishirzi, and Daniel Khashabi. When not to trust language models: Investigating effectiveness and limitations of parametric and non- parametric memories. CoRR, abs/2212.10511, 2022. doi: 10.48550/ARXIV.2212.10511. URL https://doi.org/10.48550/arXiv.2212.10511. [39] Moonshot AI. Kimi k2: Open agentic intelligence. https://moonshotai.github.io/Kimi- K2/, 2025. URL https://moonshotai.github.io/Kimi- K2/. Accessed: 2025- 07- 15. [40] Siddharth Nayak, Adelmo Morrison Orozco, Marina Ten Have, Jackson Zhang, Vittal Thirumalai, Darren Chen, Aditya Kapoor, Eric Robinson, Karthik Gopalakrishnan, James Harrison, Anuj Mahajan, Brian Ichter, and Hamsa Balakrishnan. Long- horizon planning for multi- agent robots in partially observable environments. In Amir Glandersons, Lester Mackey, Danielle Belgrave, Angela Fan, Ulrich Paquet, Jakub M. Tomczak, and Cheng Zhang, editors, Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024, 2024. URL http://papers.nips.cc/paper_files/paper/2024/hash/7d6e85e88495104442af94c983899659- Abstrac t- Conference.html. [41] OpenAI. GPT- 4 technical report. CoRR, abs/2303.08774, 2023. doi: 10.48550/ARXIV.2303.08774. URL https://doi.org/10.48550/arXiv.2303.08774. [42] OpenAI. Openai o3 and o4- mini system card. https://cdn.openai.com/pdf/2221c875- 02dc- 4789- 800b- e7758 f3722c1/o3- and- o4- mini- system- card.pdf, 2025. URL https://cdn.openai.com/pdf/2221c875- 02dc- 478 9- 800b- e7758f3722c1/o3- and- o4- mini- system- card.pdf. [43] Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe. Training language models to follow instructions with human feedback. In Sanmi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a0 01731- Abstract- Conference.html. [44] Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. Instruction tuning with GPT- 4. CoRR, abs/2304.03277, 2023. doi: 10.48550/ARXIV.2304.03277. URL https://doi.org/10.48550/arXiv.2304.03277. [45] Archiki Prasad, Alexander Koller, Mareike Hartmann, Peter Clark, Ashish Sabharwal, Mohit Bansal, and Tushar Khot. Adapt: As- needed decomposition and planning with language models. In Kevin Duh, Helena Gomez- Adorno, and Steven Bethard, editors, Findings of the Association for Computational Linguistics: NAACL 2024, Mexico City, Mexico, June 16- 21, 2024, pages 4226- 4252. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024.FINDINGS- NAACL.264. URL https://doi.org/10.18653/v1/2024.findings- naacl.264. [46] Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A. Smith, and Mike Lewis. Measuring and narrowing the compositionality gap in language models. In Houda Bouamor, Juan Pino, and Kalika Bali, editors, Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6- 10, 2023, pages 5687- 5711. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.FINDINGS- EMNLP.378. URL https://doi.org/10.18653/v1/2023.findings- emnlp.378. [47] Zehan Qi, Xiao Liu, Iat Long Iong, Hanyu Lai, Xueqiao Sun, Jiadai Sun, Xinyue Yang, Yu Yang, Shuntian Yao, Wei Xu, Jie Tang, and Yuxiao Dong. Webrl: Training LLM web agents via self- evolving online curriculum reinforcement learning. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24- 28, 2025. OpenReview.net, 2025. URL https://openreview.net/forum?id=oVKEAFjEqv.

[48] Yujia Qin, Shengding Hu, Yankai Lin, Weize Chen, Ning Ding, Ganqu Cui, Zheni Zeng, Xuanhe Zhou, Yufei Huang, Chaojun Xiao, Chi Han, Yi R. Fung, Yusheng Su, Huadong Wang, Cheng Qian, Runchu Tian, Kunlun Zhu, Shihao Liang, Xingyu Shen, Bokai Xu, Zhen Zhang, Yining Ye, Bowen Li, Ziwei Tang, Jing Yi, Yuzhang Zhu, Zhenning Dai, Lan Yan, Xin Cong, Yaxi Lu, Weilin Zhao, Yuxiang Huang, Junxi Yan, Xu Han, Xian Sun, Dahai Li, Jason Phang, Cheng Yang, Tongshuang Wu, Heng Ji, Guoliang Li, Zhiyuan Liu, and Maosong Sun. Tool learning with foundation models. ACM Comput. Surv., 57(4):101:1- 101:40, 2025. doi: 10.1145/3704435. URL https://doi.org/10.1145/3704435. [49] Qwen Team. Qwq- 32b: Embracing the power of reinforcement learning, March 2025. URL https://qwenlm.git hub.io/blog/qwq- 32b/. [50] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Erman, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/a85b4 05ed65c6477a4fe8302b5e06ce7- Abstract- Conference.html. [51] Matthew Renze and Erhan Guven. Self- reflection in LLM agents: Effects on problem- solving performance. CoRR, abs/2405.06682, 2024. doi: 10.48550/ARXIV.2405.06682. URL https://doi.org/10.48550/arXiv.2405.06682. [52] John Schulman, Sergey Levine, Pieter Abbeel, Michael I. Jordan, and Philipp Moritz. Trust region policy optimization. In Francis R. Bach and David M. Blei, editors, Proceedings of the 32nd International Conference on Machine Learning, ICML 2015, Lille, France, 6- 11 July 2015, volume 37 of JMLR Workshop and Conference Proceedings, pages 1889- 1897. JMLR.org, 2015. URL http://proceedings.mlr.press/v37/schulman15. html. [53] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Redford, and Oleg Klimov. Proximal policy optimization algorithms. CoRR, abs/1707.06347, 2017. URL http://arxiv.org/abs/1707.06347. [54] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. CoRR, abs/2402.03300, 2024. doi: 10.48550/ARXIV.2402.03300. URL https://doi.org/10.48550/arXiv.2402.03300. [55] Junhong Shen, Hao Bai, Lunjun Zhang, Yifei Zhou, Amrith Setlur, Shengbang Tong, Diego Caples, Nan Jiang, Tong Zhang, Ameet Talwalkar, and Aviral Kumar. Thinking vs. doing: Agents that reason by scaling test- time interaction. CoRR, abs/2506.07976, 2025. doi: 10.48550/ARXIV.2506.07976. URL https://doi.org/10.48550 /arXiv.2506.07976. [56] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and efficient RLHF framework. In Proceedings of the Twentieth European Conference on Computer Systems, EuroSys 2025, Rotterdam, The Netherlands, 30 March 2025 - 3 April 2025, pages 1279- 1297. ACM, 2025. doi: 10.1145/3689031.3696075. URL https://doi.org/10.1145/3689031.3696075. [57] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: language agents with verbal reinforcement learning. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper_files/papers/2023/hash/1b44b878bb782e6954cd88628510e90- A bstract- Conference.html. [58] Mohit Shridhar, Xingdi Yuan, Marc- Alexandre Cote, Yonatan Bisk, Adam Trischler, and Matthew J. Hausknecht. Allworld: Aligning text and embodied environments for interactive learning. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3- 7, 2021. OpenReview.net, 2021. URL https://openreview.net/forum?id=010X0YcCdTn. [59] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM test- time compute optimally can be more effective than scaling model parameters. CoRR, abs/2408.03314, 2024. doi: 10.48550/ARXIV.2408.03314. URL https://doi.org/10.48550/arXiv.2408.03314. [60] Giulio Starace, Oliver Jaffe, Dane Sherburn, James Aung, Jun Shern Chan, Leon Maksin, Rachel Dias, Evan Mays, Benjamin Kinsella, Wyatt Thompson, Johannes Heidecke, Amelia Glaese, and Tejal Patwardhan. Paperbench: Evaluating ai's ability to replicate AI research. CoRR, abs/2504.01848, 2025. doi: 10.48550/ARXIV.2504.01848. URL https://doi.org/10.48550/arXiv.2504.01848.

[61] Haotian Sun, Yuchen Zhuang, Lingkai Kong, Bo Dai, and Chao Zhang. Adaplanner: Adaptive planning from feedback with language models. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/b5c8c1c117618267944b2617add0a766- Abstrac t- Conference.html.[62] Richard S. Sutton and Andrew G. Barto. Reinforcement learning - an introduction, 2nd Edition. MIT Press, 2018. URL http://www.incompleteideas.net/book/the- book- 2nd.html.[63] Richard S. Sutton, David A. McAllester, Satinder Singh, and Yishay Mansour. Policy gradient methods for reinforcement learning with function approximation. In Sara A. Soll, Todd K. Loar, and Klaus- Robert Müller, editors, Advances in Neural Information Processing Systems 12, [NIPS Conference, Denver, Colorado, USA, November 29 - December 4, 1999], pages 1057- 1063. The MIT Press, 1999. URL http://papers.nips.cc/paper /1713- policy- gradient- methods- for- reinforcement- learning- with- function- approximation.[64] Yashar Talebirad and Amirhossein Nadiri. Multi- agent collaboration: Harnessing the power of intelligent LLM agents. CoRR, abs/2306.03314, 2023. doi: 10.48550/ARXIV.2306.03314. URL https://doi.org/10.48550/arX iv.2306.03314. [65] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean- Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. [66] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, Chuning Tang, Congcong Wang, Dehao Zhang, Enming Yuan, Enzhe Lu, Fengxiang Tang, Flood Sung, Guangda Wei, Guokun Lai, Haiqing Guo, Han Zhu, Hao Ding, Hao Hu, Hao Yang, Hao Zhang, Haotian Yao, Haotian Zhao, Haoyu Lu, Haoze Li, Haozhen Yu, Hongcheng Gao, Huabin Zheng, Huan Yuan, Jia Chen, Jianhang Guo, Jianlin Su, Jianzhou Wang, Jie Zhao, Jin Zhang, Jingyuan Liu, Junjie Yan, Junyan Wu, Lidong Shi, Ling Ye, Longhui Yu, Menghan Dong, Neo Zhang, Ningchen Ma, Qiwei Pan, Qucheng Gong, Shaoweil Liu, Shengling Ma, Shupeng Wei, Sihan Cao, Siying Huang, Tao Jiang, Weihao Gao, Weimin Xiong, Weiran He, Weixiao Huang, Wenhao Wu, Wenyang He, Xianghui Wei, Xianqing Jia, Xingzhe Wu, Xinran Xu, Xinxing Zu, Xinyu Zhou, Xuehai Pan, Y. Charles, Yang Li, Yangyang Hu, Yangyang Liu, Yanru Chen, Yejie Wang, Yibo Liu, Yidao Qin, Yifeng Liu, Ying Yang, Yiping Bao, Yulun Du, Yuxin Wu, Yuzhi Wang, Zaida Zhou, Zhaoji Wang, Zhaowei Li, Zhen Zhu, Zheng Zhang, Zenxu Wang, Zhilin Yang, Zhiqi Huang, Zihao Huang, Ziyao Xu, and Zonghan Yang. Kimi k1.5: Scaling reinforcement learning with llms. CoRR, abs/2501.12599, 2025. doi: 10.48550/ARXIV.2501.12599. URL https://doi.org/10.48550/arXiv.2501.12599. [67] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique. Multihop questions via single- hop question composition. Trans. Assoc. Comput. Linguistics, 10:539- 554, 2022. doi: 10.1162/TACL A_00475. URL https://doi.org/10.1162/tacl_a_00475. [68] Luong Quoc Trung, Xinbo Zhang, Zhanming Jie, Peng Sun, Xiaoran Jin, and Hang Li. Reft: Reasoning with reinforced fine- tuning. In Lun- Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11- 16, 2024, pages 7601- 7610. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024ACL- LONG.410. URL https://doi.org/10.18653/v1/2024. acl- long 410. [69] Ruoyao Wang, Peter A. Jansen, Marc- Alexandre Cote, and Prithviraj Ammanabrolu. Scienceworld: Is your agent smarter than a 5th grader? In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang, editors, Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7- 11, 2022, pages 11279- 11298. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022. EMNLP- MAIN.775. URL https://doi.org/10.18653/v1/2022. emnlp- main.775. [70] Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self- consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1- 5, 2023. OpenReview.net, 2023. URL https://openreview.net/forum?id=1PL1NIMMrw.[71] Zihan Wang, Kangrui Wang, Qineng Wang, Pingyue Zhang, Linjie Li, Zhengyuan Yang, Xing Jin, Kefan Yu, Minh Nhat Nguyen, Licheng Liu, Eli Gottlieb, Yiping Lu, Kyunghyun Cho, Jiajun Wu, Li Fei- Fei, Lijuan Wang, Yejin Choi, and Manling Li. RAGEN: understanding self- evolution in LLM agents via multi- turn

reinforcement learning. CoRR, abs/2504.20073, 2025. doi: 10.48550/ARXIV.2504.20073. URL https://doi.org/10.48550/arXiv.2504.20073.

[72] Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, and Amelia Glaese. Browsecomp: A simple yet challenging benchmark for browsing agents. CoRR, abs/2504.12516, 2025. doi: 10.48550/ARXIV.2504.12516. URL https://doi.org/10.48550/arXiv.2504.12516.

[73] Ronald J. Williams. Simple statistical gradient- following algorithms for connectionist reinforcement learning. Mach. Learn., 8:229- 256, 1992. doi: 10.1007/BF00992696. URL https://doi.org/10.1007/BF00992696.

[74] Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Shaokun Zhang, Erkang Zhu, Beibin Li, Li Jiang, Xiaoyun Zhang, and Chi Wang. Autogen: Enabling next- gen LLM applications via multi- agent conversation framework. CoRR, abs/2308.08155, 2023. doi: 10.48550/ARXIV.2308.08155. URL https://doi.org/10.48550/arXiv.2308.08155.

[75] xAI. Grok 4. https://x.ai/news/grok- 4, 2025.

[76] Zhiheng Xi, Wenxiang Chen, Boyang Hong, Senjie Jin, Rui Zheng, Wei He, Yiwen Ding, Shichun Liu, Xin Guo, Junzhe Wang, Honglin Guo, Wei Shen, Xiaoran Fan, Yuhao Zhou, Shihan Dou, Xiao Wang, Xinbo Zhang, Peng Sun, Tao Gui, Qi Zhang, and Xuanjing Huang. Training large language models for reasoning through reverse curriculum reinforcement learning. In Forty- first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21- 27, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=t82Y31mRtk.

[77] Zhiheng Xi, Yiwen Ding, Wenxiang Chen, Boyang Hong, Honglin Guo, Junzhe Wang, Dingwen Yang, Chenyang Liao, Xin Guo, Wei He, Songyang Gao, Lu Chen, Rui Zheng, Yicheng Zou, Tao Gui, Qi Zhang, Xipeng Qiu, Xuanjing Huang, Zuxuan Wu, and Yu- Gang Jiang. Agentgym: Evolving large language model- based agents across diverse environments. CoRR, abs/2406.04151, 2024. doi: 10.48550/ARXIV.2406.04151. URL https://doi.org/10.48550/arXiv.2406.04151.

[78] Zhiheng Xi, Dingwen Yang, Jixuan Huang, Jiafu Tang, Guanyu Li, Yiwen Ding, Wei He, Boyang Hong, Shihan Dou, Wenyu Zhan, Xiao Wang, Rui Zheng, Tao Ji, Xiaowei Shi, Yitao Zhai, Rongxiang Wang, Jingang Wang, Xunliang Cai, Tao Gui, Zuxuan Wu, Qi Zhang, Xipeng Qiu, Xuanjing Huang, and Yu- Gang Jiang. Enhancing LLM reasoning via critique models with test- time and training- time supervision. CoRR, abs/2411.16579, 2024. doi: 10.48550/ARXIV.2411.16579. URL https://doi.org/10.48550/arXiv.2411.16579.

[79] Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, Rui Zheng, Xiaoran Fan, Xiao Wang, Limao Xiong, Yuhao Zhou, Weiran Wang, Changhao Jiang, Yicheng Zou, Xiangyang Liu, Zhangyue Yin, Shihan Dou, Rongxiang Weng, Wenjuan Qin, Yongyan Zheng, Xipeng Qiu, Xuanjing Huang, Qi Zhang, and Tao Gui. The rise and potential of large language model based agents: a survey. Sci. China Inf. Sci., 68(2), 2025. doi: 10.1007/S11432- 024- 4222- 0. URL https://doi.org/10.1007/s11432- 024- 4222- 0.

[80] Han Xia, Songyang Gao, Qiming Ge, Zhiheng Xi, Qi Zhang, and Xuanjing Huang. Inverse- q*: Token level reinforcement learning for aligning large language models without preference data. In Yaser Al- Onaizan, Mohit Bansal, and Yun- Nung Chen, editors, Findings of the Association for Computational Linguistics: EMNLP 2024, Miami, Florida, USA, November 12- 16, 2024, pages 8178- 8188. Association for Computational Linguistics, 2024. doi: 10.18653/V1/2024. FINDINGS- EMNLP.478. URL https://doi.org/10.18653/v1/2024. Findings- emnlp.478.

[81] Yuxi Xie, Kenji Kawaguchi, Yiran Zhao, James Xu Zhao, Min- Yen Kan, Junxian He, and Michael Qizhe Xie. Self- evaluation guided beam search for reasoning. In Alice Oh, Tristan Naumann, Amir Globerson, Kate Saenko, Moritz Hardt, and Sergey Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023, 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/81fde95c4dc79188a69ce5b24d63010b- Abstract- Conference.html.

[82] Zhihui Xie, Jie Chen, Liyu Chen, Weichao Mao, Jingjing Xu, and Lingpeng Kong. Teaching language models to critique via reinforcement learning. CoRR, abs/2602.03492, 2025. doi: 10.48550/ARXIV.2502.03492. URL https://doi.org/10.48550/arXiv.2502.03492.

[83] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang,

Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. CoRR, abs/2412.15115, 2024. doi: 10.48550/ARXIV.2412.15115. URL https://doi.org/10.48550/arXiv.2412.15115.

[84] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jian Yang, Jiaxi Yang, Jingren Zhou, Jingren Zhou, Junyang Liu, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingdong Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, and Zihan Qiu. Qwen3 technical report. CoRR, abs/2505.09388, 2025. doi: 10.48550/ARXIV.2505.09388. URL https://doi.org/10.48550/arXiv.2505.09388.

[85] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. Hotpotqa: A dataset for diverse, explainable multi- hop question answering. In Ellen Riloff, David Chiang, Julia Hockenmaier, and Jun'ichi Tsujii, editors, Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, October 31 - November 4, 2018, pages 2369- 2380. Association for Computational Linguistics, 2018. doi: 10.18653/v1/D18- 1259. URL https://doi.org/10.18653/v1/d18- 1259.

[86] Shunyu Yao. Language Agents: From Next- Token Prediction to Digital Automation. PhD thesis, Princeton University, 2024.

[87] Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. Webshop: Towards scalable real- world web interaction with grounded language agents. In Sammi Koyejo, S. Mohamed, A. Agarwal, Danielle Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022, 2022. URL http://papers.nips.cc/paper/files/paper/2022/hash/82ad13ec01f9fe44c01cb91814fd7b8c- Abstract- Conference.html.

[88] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1- 5, 2023. OpenReview.net, 2023. URL https://openreview.net/forum?id=JE_vluYUL- X.

[89] Junjie Ye, Yilong Wu, Sixian Li, Yuming Yang, Tao Gui, Qi Zhang, Xuanjing Huang, Peng Wang, Zhongchao Shi, Jianping Fan, and Zhengyin Du. Tl- training: A task- feature- based framework for training large language models in tool use. CoRR, abs/2412.15495, 2024. doi: 10.48550/ARXIV.2412.15495. URL https://doi.org/10.48550/arXiv.2412.15495.

[90] Junjie Ye, Zhengyin Du, Xuesong Yao, Weijian Lin, Yufei Xu, Zehui Chen, Zaiyuan Wang, Sining Zhu, Zhiheng Xi, Siyu Yuan, Tao Gui, Qi Zhang, Xuanjing Huang, and Jiecao Chen. Toolhop: A query- driven benchmark for evaluating large language models in multi- hop tool use. CoRR, abs/2501.02506, 2025. doi: 10.48550/ARXIV.2501.02506. URL https://doi.org/10.48550/arXiv.2501.02506.

[91] Aohan Zeng, Mingbao Liu, Rui Lu, Bowen Wang, Xiao Lu, Yuxiao Dong, and Jie Tang. Agentuing: Enabling generalized agent abilities for llms. In Lun- Wei Ku, Andre Martins, and Vivek Srikumar, editors, Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11- 16, 2024, pages 3053- 3077. Association for Computational Linguistics, 2024. doi: 10.18653/v1/2024. FINDINGS- A CL.181. URL https://doi.org/10.18653/v1/2024. findings- aci.181.

[92] Jianguo Zhang, Tian Lan, Rithesh Murthy, Zhiwei Liu, Weiran Yao, Juntao Tan, Thai Hoang, Liangwei Yang, Yihao Feng, Zuxin Liu, Tulika Manoj Awalgaonkar, Juan Carlos Niebles, Silvio Savarese, Shelby Heinecke, Huan Wang, and Caiming Xiong. Agentohana: Design unified data and training pipeline for effective agent learning. CoRR, abs/2402.15506, 2024. doi: 10.48550/ARXIV.2402.15506. URL https://doi.org/10.48550/arXiv.2402.15506.

[93] Rui Zheng, Shihan Dou, Songyang Gao, Yuan Hua, Wei Shen, Binghai Wang, Yan Liu, Senjie Jin, Qin Liu, Yuhao Zhou, Limao Xiong, Lu Chen, Zhiheng Xi, Nuo Xu, Wenbin Lai, Minghao Zhu, Cheng Chang, Zhangyue Yin, Rongxiang Weng, Wensen Cheng, Haoran Huang, Tianxiang Sun, Hang Yan, Tao Gui, Qi Zhang, Xipeng Qiu, and Xuanjing Huang. Secrets of RLHF in large language models part I: PPO. CoRR, abs/2307.04964, 2023. doi: 10.48550/ARXIV.2307.04964. URL https://doi.org/10.48550/arXiv.2307.04964.

[94] Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. Webarena: A realistic web environment for building autonomous agents. In The Twelfth International Conference on Learning Representations. ICLR 2024, Vienna, Austria, May 7- 11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=oKn9c6ytLx.[95] Yifei Zhou, Andrea Zanette, Jiayi Pan, Sergey Levine, and Aviral Kumar. Archer: Training language model agents via hierarchical multi- turn RL. In Forty- first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21- 27, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=prA0xAHT1. [96] King Zhu, Hanhao Li, Siwei Wu, Tianshun Xing, Dehua Ma, Xiangru Tang, Minghao Liu, Jian Yang, Jiaheng Liu, Yuchen Eleanor Jiang, Changwang Zhang, Chenghua Lin, Jun Wang, Ge Zhang, and Wangchunshu Zhou. Scaling test- time compute for LLM agents. CoRR, abs/2506.12928, 2025. doi: 10.48550/ARXIV.2506.12928. URL https://doi.org/10.48550/arXiv.2506.12928. [97] Tinghui Zhu, Kai Zhang, Jian Xie, and Yu Su. Deductive beam search: Decoding deducible rationale for chain- of- thought reasoning. CoRR, abs/2401.17686, 2024. doi: 10.48550/ARXIV.2401.17686. URL https://doi.org/10.48550/arXiv.2401.17686.

# Appendix

## A Details of the AgentGym-RL Architecture

This appendix provides a detailed description of the AgentGym- RL architecture, complementing the highlevel overview in the main text (Figure 2). We present the implementation details of the three core modules- Environment, Agent, and Training. These details highlight the engineering considerations that ensure scalability, flexibility, and reproducibility in large- scale RL experiments.

Environment module. In this module, each environment is packaged as an independent service with the option of deploying multiple replicas to support parallel requests. An environment client communicates with the environment server via HTTP and exposes APIs to the agent, including /observation to get the current observation from the environment, /available_actions to get the currently available actions, /step to perform an action, and /reset to reset the environment. Currently, AgentGym- RL covers five major scenario categories. This modular server- client design allows new environments to provide comprehensive environment and data support for LLM agent training.

Agent module. The agent module encapsulates the reasoning- action loop of LLM- based agents. It receives observations from the environment, performs reasoning over multiple turns, and outputs actions (e.g., invoking provided APIs). The module supports different prompting strategies, sampling configurations, and reward functions.

Training module. The training module provides a unified reinforcement learning (RL) pipeline that supports both online and offline algorithms, offering researchers a flexible foundation for large- scale LLM agent training. The module manages the entire RL lifecycle: trajectory collection, advantage estimation, policy optimization, and reward shaping, while also supporting curriculum learning and staged interaction scaling (i.e., ScalingInter- RL).

The entire training pipeline can be distributed across multiple nodes, leveraging both multi- process and multi- node parallelism. Efficient batching and asynchronous logging utilities ensure that system throughput scales with additional compute resources. Diagnostics tools are integrated to provide fine- grained metrics, including policy entropy, KL divergence, reward curves, and rollout statistics, which are automatically recorded for later analysis and reproducibility.

## B Implementation Details and Settings of Each Environment

We conduct all the experiments on NVIDIA A100 GPUs and Ascend 910B NPUs. The remaining part of this section shows detailed setting of different environments.

### B.1 Web Navigation Scenario

Tools and APIs. In web navigation scenario, the agent simulates human interaction with web pages to ultimately complete the task. WebArena[94] supports these interactioins through a set of tool APIs, allowing agents to perform a variety of real- world tasks, including online shopping, engaging in discussions on Reddit, collaborating on software development via GitLab, and managing store content through a CMS. In addition to these online platforms, WebArena also provides three utility- style tools: a map for navigation and location- based information search, a calculator, and a scratchpad for note- taking.

A query case of web navigation is shown below:

Web Navigation Example

Web Navigation ExampleYou are an autonomous intelligent agent tasked with navigating a web browser. You will be given web- based tasks. These tasks will be accomplished through the use of specific actions you can issue.

Available Information:

Available Information:- User's objective: The task to complete- Accessibility tree: Simplified webpage representation, providing key information.- Current URL: The active page's address- Open tabs: Currently available tabs- Previous action: Last performed action

Action Categories:

Page Operations:

click [id]:Click element with ID type [id] [content] [0|1]: Input text (1  \(=\)  press Enter) hover [id]: Hover over element press [key_comb]: Simulate key press (e.g., Ctrl+v) scroll [down|up]: Scroll page direction

Tab Management:

new_tab:Open new tab tab_focus [tab_index]:Switch to tab close_tab: Close current tab

URL Navigation:

goto [url]:Navigate to URL go_back: Return to previous page go_forward: Advance to next page

Completion:

stop [answer]: Submit final answer (or "N/A" if you believe the task is impossible to complete)

Homepage: If you want to visit other websites, check out the homepage at http://homepage.com.

Objective: Among the top 10 post in "books" forum, show me the book names from posts that recommand a single book.

Settings. We include five subtasks: E- commence, Reddit, Gitlab, OpenStreetMap (Map), and online store content management system (CMS), comprising a total of 372 training queries and 50 testing queries. These are selected from the origin WebArena dataset, which contains 812 queries across three categories: Information Seeking, Site Navigation, and Content & Config. To facilitate efficient parallel rollout, we exclude the Content & Config tasks, which involve insert, update and delete operations that change the state of the websites. We set the maximum number of agent- environment interactions to 15 turns. For the SFT baselines, we set the learning rate to  \(1 \times 10^{- 4}\) . We employ GRPO as the main RL algorithm with a learning rate of  \(5 \times 10^{- 7}\)  and a KL coefficient of  \(1 \times 10^{- 3}\) . For each query, we sample 4 distinct trajectories using a temperature of 1.0.

### B.2 Deep Search Scenario

Tools and APIs. The deep search senario features a search engine- based environment equipped with specialized tools and APIs supporting the interaction with search engines. These APIs enable agents to dynamically generate search queries during the reasoning process, retrieve relevant information from external sources, and incorporate the retrieved information into subsequent reasoning steps. This setting allows agents to engage in complex reasoning processes that involve iterative searching and information integration, thereby enhancing their capability to solve intricate problems where external knowledge is essential.

A query case of Deep Search is shown below:

Deep Search Example

You must always reason inside <think>...</think> first; if you lack knowledge, issue a <search>...</search> and then stop; do not generate <information> or <answer> yet; wait for external input between <information>...</information> before continuing; resume only when new <information> is given; do not skip steps or anticipate answers early.

Question: Who got the first Nobel Prize in Physics?

Settings. We include queries from 7 datasets following the setup of Search- R1 [26]: NQ [32], TriviaQA [28], PopQA [38], HotpotQA [85], 2wiki [19], Musique [67], and Bamboogle [46]. To ensure fair comparison and balanced evaluation, we randomly sample 400 examples from the development sets of NQ. TriviaQA, PopQA, HotpotQA, 2wiki, Musique, and Bamboogle. The maximum number of agent- environment interactions is set to 4 turns. For the SFT baselines, the learning rate is set to  \(1 \times 10^{- 4}\) . We employ GPRO as the main algorithm for reinforcement learning setups with a learning rate of  \(1 \times 10^{- 6}\) , a KL coefficient of  \(1 \times 10^{- 3}\) , and a sampling temperature of 1.0. We sample 8 distinct trajectories for a single query.

### B.3 Digital Games Scenario

B.3 Digital Games ScenarioEnvironments, Tools and APIs. As for digital games, we introduce TextCraft[45], a text-based game environment mirroring Minecraft. The APIs in TextCraft include crafting, inventory management, and dynamic narrative generation. These APIs allow agents to execute predefined crafting recipes, manipulate inventory contents, navigate virtual spaces, dynamically generate quests and sub-tasks based on natural language objectives, and recursively decompose complex tasks into achievable sub-goals.A query case of TextCraft can be seen below:

A query case of TextCraft can be seen below:

TextCraft Example

You are given few useful crafting recipes to craft items in Minecraft. Crafting commands are of the format "craft [arget object] using [input ingredients]". Every round I will give you an observation, you have to respond an action based on the state and instruction. You can "get" an object (ingredients) from the inventory or the environment, look- up the game inventory by "inventory", or "craft" (target) using any of the crafting commands. You can use ONLY these crafting commands provided, do not use your own crafting commands. However, if the crafting command uses a generic ingredient like "planks", you can use special types of the same ingredient e.g. "dark oak planks" in the command instead.

Goal: Craft flint and steel.

Settings. In TextCraft, task difficulty is measured by the maximum depth of the corresponding crafting tree. In practice, the benchmark contains tasks with crafting trees of depths 1, 2, 3, and 4. Accordingly, we divide the entire task set into four subsets based on these depths. We set the maximum number of interactions to 20 turns. In the SFT baselines, we set the learning rate to  \(1 \times 10^{- 4}\) . We employ GPRO as the main RL algorithm with a learning rate of  \(1 \times 10^{- 6}\) , a KL coefficient of  \(1 \times 10^{- 3}\) , and a sampling temperature of 1.0. We sample 8 distinct trajectories for a single query.

### B.4 Embodied Scenario

B.4 Embodied ScenarioTools and APIs. We introduce the BabyAI environment as a representative setting for embodied tasks. It provides APIs that allow agents to navigate a controllable grid world using natural language instructions. Through these APIs, agents can perform actions such as moving objects, unlocking doors, and interacting with the environment in response to textual commands.

A query case of BabyAI can be seen below:

BabyAI Example

You are an exploration master that wants to finish every goal you are given. Every round I will give you an observation, and you have to respond an action and your thought based on the observation to finish the given task. You are placed in a room and you need to accomplish the given goal with actions. You can use the following actions:

- turn right 
- turn left 
- move forward 
- go to obj id 
- pick up obj id 
- go through door id: door must be an open door. 
- toggle and go through door id: door can be a closed door or a locked door. If you want to open a locked door, you need to carry a key that is of the same color as the locked door. 
- toggle: there is a closed or locked door right in front of you and you can toggle it.

Your goal: Go to the red ball.

Settings. Following the original implementation, we divide the tasks into six subsets based on the final goal. We set the maximum number of interactions to 20 turns. In SFT baselines, we set the learning rate to  \(1 \times 10^{- 4}\) . We employ GRPO as the main RL algorithm with a learning rate of  \(1 \times 10^{- 6}\) , a KL coefficient of  \(1 \times 10^{- 3}\) , and a sampling temperature of 1.0. We sample 8 distinct trajectories for a single query.

### B.5 Scientific Scenario

Tools and APIs. SciWorld[69] is an agent environment for scientific tasks. It provides APIs that are designed to support scientific exploration through text- driven reasoning cycles. These APIs empower agents to conduct experiments by interacting with various scientific apparatus and performing actions like measuring temperature, connecting electrical circuits, and mixing chemicals.

A query case of SciWorld can be seen below:

SciWorld Example

You are an agent for science world. Every round I will give you an observation, you have to respond an action based on the observation to finish the given task.

Your task is to boil water. For compounds without a boiling point, combusting the substance is also acceptable. First, focus on the substance. Then, take actions that will cause it to change its state of matter.

Settings. We select 8 subsets of tasks from the original SciWorld environment. We set the maximum number of interactions between the agent and the environment to 20 turns. In SFT baselines, we set the learning rate to  \(1 \times 10^{- 4}\) . We employ GRPO as the main RL algorithm with a learning rate of  \(1 \times 10^{- 6}\) , a KL coefficient of  \(1 \times 10^{- 3}\) , and a sampling temperature of 1.0. We sample 8 distinct trajectories for a single query.

## C Trajectory examples and visualizations of our RL agent

This appendix provides additional trajectory visualizations across multiple environments. The figures illustrate the behaviors of both baseline and RL- trained agents, highlighting the RL model's superior performance in exploration, task execution, and interaction patterns, while also revealing common failure modes that remain.

### BabyAI

#### Task: go to the blue box

obs 0: In front of you in this room, you can see several objects: There is a yellow closed door 1 2 steps in front of you and 1 steps to your right. There is a green closed door 1 3 steps in front of you and 3 steps to your left. The room has walls around you. You are facing a wall 3 steps away. You are not carrying anything.

Available actions: ["turn left", "turn right", "move forward", "toggle and go through yellow closed door 1", "toggle and go through green closed door 1", "go to yellow closed door 1", "go to green closed door 1", "check available actions"]

![](images/8beec13e3a962b70a672372816853476daf1ffaf85258833bcfb5f1e83496001.jpg)

#### Base Model

#### Our RL Model

Thought 1: To reach the blue box, I need to move forward, but there is a yellow closed door in front of me, blocking the way. ...it's more efficient to go through the yellow closed door first. Act 1: toggle and go through yellow closed door 1

Obs 1: There is a green closed door 1 6 steps in front of you and 2 steps to your right. The room has walls around you. You are facing a wall 6 steps away.

Thought 2: ... Now that I have gone through the yellow closed door, Since I don't have any keys, I need to toggle and go through this door to find the blue box. Act 2: toggle and go through green closed door 1

Thought 3: I now face a grey closed door directly in front of me. I need to toggle and go through the grey closed door to continue my journey to find the blue box. Act 3: toggle and go through grey closed door 1

Thought 4: ...my only option is to toggle it and attempt to go through.

Act 4: toggle and go through red closed door 1 Obs 4: The room has walls around you. You are facing a wall 6 steps away.

Act 5: move forward Act 6: go through red open door 1 Act 7: go through grey open door 1

Act 13: go through grey open door 1 Act 14: go through red open door 1

Thought 15: Given that the room has walls around me and I am not carrying anything, if it appears I have reached the end of this path. I have already explored this area thoroughly and there are no more objects or doors to interact with. Act 15: check available actions

Thought 20: Turning left again has brought me to a wall, confirming that I have explored all possible directions in this room.

Act 20: check available actions

Thought 1: ... find the blue box

Act 1: toggle and go through yellow closed door 1 Obs 1: There is a green closed door 1 6 steps in front of you and 2 steps to your right. The room has walls around you. You are facing a wall 6 steps away.

Thought 2: ... find the blue box

Obs 2: There is a green closed door 1 right in front of you 1 steps away. The room has walls around you.

Thought 3: ... find the blue box

Act 3: toggle and go through green closed door 1 Obs 3: There is a grey closed door 1 4 steps in front of you and 3 steps to your left. The room has walls around you. You are facing a wall 6 steps away.

Act 4: go to grey closed door 1

Act 5: toggle and go through grey closed door 1

Act 6: go to red closed door 1

Act 7: turn right

Act 8: turn right

Act 9: go through grey open door 1

Act 10: turn right

Act 11: go through green open door 1

Act 12: go through yellow open door 1

Act 13: toggle and go through green closed door 1 Obs 13: There is a blue box 1 4 steps in front of you and 1 steps to your left. There is a green closed door 1 6 steps in front of you and 3 steps to your right. The room has walls around you. You are facing a wall 6 steps away.

Act 14: go to blue box 1

#### Score: 0.0

Score: 1.0

Figure 12 Smart case of our RL agent compared to the base agent on BabyAI task.

![](images/436babb6c8a80766cf7a6f2eae9b0c59108f61271790d7c3e53d599a40b65531.jpg)  
Figure 13 Trajectory visualization of our RL agent on BabyAI task.

### Sciworld

Task: Find a(n) living thing. First, focus on the thing. Then, move it to the purple box in the bedroom.

obs 8: This room is called the greenhouse. In it, you see: the agent, a substance called air, a bee hive. The bee hive door is closed. a flower pot 3 (containing nothing), a flower pot 4 (containing a peach tree in the adult stage, soil), a flower pot 5 (containing a pea plant in the adult stage with a tall height, soil), a flower pot 8 (containing nothing), a flower pot 9 (containing a banana tree in the adult stage, soil), a jug (containing nothing), a shovel, a sink, which is turned off. In the sink is: nothing. You also see: A door to the hallway (that is closed) A door to the outside (that is closed)

#### Base Model

#### Our RL Model

Thought 1: The task is to find a living thing and move it to the purple box in the bedroom. In the greenhouse, a bee hive is the only potential living thing listed, but it is closed. The peach tree, pea plant, and banana tree are all living things, but they are in flower pots. I'll start by opening the bee hive to check if there are any bees inside. Act 1: open bee hive door. Obs 1: No known Act matches that input.

Thought 2: It seems there was a misunderstanding. Since there are living things in the flower pots (peach tree, pea plant, and banana tree), I should start by focusing on one of them. Let's begin with the peach tree, as it is in a recognizable stage and might be more obvious.

Act 2: focus on flower pot 4. Obs 2: You focus on the flower pot 4.

Thought 1: My task is to find a living thing, focus on it, and move it to the purple box in the bedroom. Act 1: look around. Obs 1: ...

Thought 2: Around the greenhouse, I found a peach tree, a pea plant, and a banana tree. Let's pick up the banana tree.

Act 2: pick up the banana tree. Obs 2: You move the plant to the inventory.

Thought 3: I picked up the banana tree and added it to my inventory. Let's focus on the banana tree. Act 3: focus on the banana tree. Obs 3: I use pots on the banana tree.

Thought 7: Around the hallway, I found the agent, a picture, and some other rooms. Let's go to the bedroom to find the purple box. Act 7: go to the bedroom. Obs 7: The door is not open.

Thought 8: The door to the bedroom is not open. Let's open it before going through. Act 8: open the door to the bedroom. Obs 8: The door is now open.

Thought9: Around the hallway, I found the agent, a picture, and some other rooms. Let's go to the bedroom to find the purple box. Act 9: go to the bedroom. Obs 9: You move to the bedroom.

Thought 11: Around the bedroom, I found a purple box. Let's move the banana tree into the purple box. Act 11: put the banana tree in the purple box. Obs 11: You move the banana tree to the purple box.

Score: 0.0

Score: 100

### Sciworld

Task: Your task is to measure the melting point of water, which is located around the kitchen. First, focus on the thermometer. Next, focus on the water. If the melting point of water is above - 10.0 degrees celsius, focus on the red box. If the melting point of water is below - 10.0 degrees celsius, focus on the green box. The boxes are located around the kitchen. Obs 0: This room is called the bedroom. In it, you see: the agent a substance called air a bed. On the bed is: a mattress. On the mattress is: a white pillow.. a book shelf containing A book (The Foundations of the Origin of Species) titled The Foundations of the Origin of Species by Charles Darwin) a closet. The closet door is closed. a painting a table. On the table is: nothing. You also see: A door to the hallway (that is closed)

Thought 6: Around here, I found a finger painting. Let's go to the kitchen to find a thermometer. Obs 6: go to the kitchen

Obs 6: The door is not open.

Thought 7: The door to the kitchen is not open. Let's open it before going through. Act 7: open the door to the kitchen

Obs 7: The door is now open.

Thought 8: The door to the kitchen is now open. Let's go through to the kitchen.

Act 8: go to the kitchen

Obs 8: You move to the kitchen.

Thought 10: Around the kitchen, I found a thermometer and a red box. Let's pick up the thermometer and add it to my inventory.

Act 10: pick up the thermometer

Obs 10: You move the thermometer to the inventory.

Thought 11: I picked up the thermometer and added it to my inventory. Let's focus on the thermometer.

Act 11: Focus on the thermometer.

Obs 11: You focus on the thermometer.

Thought 12: I focused on the thermometer. Let's focus on the water to measure the melting point.

Act 12: focus on the water

Obs 12: No known action matches that input.

Thought 13: I focused on the thermometer. Let's go to the sink to focus on the water and measure the melting point.

Act 13: Focus on the water

Obs 13: No known action matches that input

Thought 14: The melting point of water is 0 degrees

celsius. Since the melting point of water is above - 10.0 degrees celsius, I will focus on the red box.

Act 14: focus on the red box.

Obs 14: You focus on the red box.

Score: 0.0

#### Sciworld

Task: Your task is to find the animal with the longest life span. The animals are in the 'outside' location. Focus on the animal with the longest life span.

Obs 0: This room is called the hallway. In it, you see: the agent a substance called air a finger painting You also see: A door to the art studio (that is closed) A door to the bedroom (that is closed) A door to the greenhouse (that is closed) A door to the kitchen (that is closed) A door to the living room (that is closed) A door to the workshop (that is closed)

Act 6: go to the outside

Obs 6: The door is not open.

Thought 7: The door to the outside is not open. Let's open it before going through.

Act 7: open the door to the outside.

Obs 7: The door is now open.

Thought 8: The door to the outside is now open. Let's go through to the outside.

Act 8: go to the outside

Obs 8: You move to the outside.

Thought 9: I have gone to the outside. Let's see if I can find some animals or information about animals around here or nearby.

Act 9: look around

Obs 9: This outside location is called the outside. Here you see: the agent a substance called air an axe a chameleon egg a fire pit (containing nothing) a fountain (containing a substance called water) the ground a parrot egg a baby rabbit a substance called wood You also see: A door to the foundry (that is closed) A door to the greenhouse (that is open) A door to the kitchen (that is closed)

Thought 10: Around here, I found a chameleon egg, a parrot egg, and a baby rabbit. Let's find out the life span of these animals.

Act 10: pick up the chameleon egg

Obs 10: You move the chameleon to the inventory. Thought 11: I picked up the chameleon egg and added it to my inventory. Let's focus on the chameleon egg.

Act 11: Focus on the chameleon egg.

Obs 11: You focus on the chameleon egg.

Score: 0.0

### WebArena

Task: What are the key aspects that the customers don't like about Zing Jump Rope Obs 0:[1] RootWebArea 'Dashboard / Magento Admin' focused:True

[108] link 'ue699 MARKETING' [361] link 'Catalog Price Rule'

Thought 1: The information provided does not seem to contain this data. We can navigate to the 'MARKETING' section to see if there is any customer feedback or survey data that might be relevant.

Act 1: click [108]

Obs 1: [376] link 'Email Templates'

Thought 2: Given the objective, we should focus on the Communications section, as it might contain customer feedback or survey data.

Act 2: click [376]

Obs 2: [1198] RootWebArea 'Email Templates / Communications / Marketing / Magento Admin' focused:True

Thought 3: We can navigate to the 'Catalog Price Rule' or 'Cart Price Rules' under the "Promotions" section to see if there is any related information.

Act 3: click [361]

Obs 3: no matching element found.

Thought 4: The "Marketing" section might have other relevant sections that we have not explored yet.

Act 4: click [108]

Obs 4: [1649] StaticText 'SEO & Search'

Thought 5: The "SEO & Search" section might have relevant data, such as search terms or URL rewrites that could indicate customer dissatisfaction.

Act 5: click [1649]

Obs 5: [1670] StaticText 'User Content' Thought 6: we can navigate to the "User Content" section to see if there is any relevant data,

Act 6: click [1670]

Obs 6: [1674] link 'All Reviews'

Thought 7: The reviews section is a potential source of customer feedback.

Act 7: click [1674]

Obs 7: click [1674] 'ue699 MARKETING'

Thought 8: We need to navigate to the reviews for Zing Jump Rope to find the relevant information.

Act 8: click [2354]

Obs 8: [2818] link 'All Reviews'

Act 9: click [2818]

Obs 9: [5172] gridcell 'Any' required: False

[5496] combobox 'AnyPopup: menu expanded: False\

Act 10: click [5172]

Act 11: click [5496]

Act 12: type [5496] [Zing Jump Rope] [5]

Thought 13: the data might not be available for Zing Jump Rope.

Act 13: stop [N/A]

#### Score: 0.0

#### WebArena

Task: Who gave 4 or 5 stars for phone cases from EYZUTAK

Obs 0: [1] RootWebArea 'One Stop Market' focused: True [972] menuitem 'ue622 Cell Phones & Accessories' hasPopup: menu

Thought 1: We need to navigate to a page where we can see the ratings for EYZUTAK's phone cases

Act 1: click [972]

Obs 1: [4955] link 'Cases, Holsters & Sleeves (457

Thought 2: "Cases, Holsters & Sleeves" which likely contains the phone cases from EYZUTAK.

Act 2: click [4955]

Obs 2: [4954] link 'Accessories (1924 item)'

Thought 3: We need to navigate to a page where we can see the ratings for these phone cases.

Act 3: click [4954]

Obs 3: [6748] combobox 'ue615 Search' autocomplete: both hasPopup: listbox required: False expanded: False

Thought 4: We need to find the specific section for

EYZUTAK phone cases and then check the ratings.

Act 4: type [6748] [EYZUTAK phone case] [1]

Obs 4: [9389] DescriptionListDetail ' [9489] link

'EYZUTAK phone case' [10306] img 'Image'

Thought 5: The search results are displayed in a grid format, and the ratings are likely to be found on the product pages.

Act 5: click [10306]

Thought 6: On a product page, the ratings and reviews are typically found in a section dedicated to customer reviews.

Act 6: hover [10309]

Thought 7: However, the ratings and reviews are not

immediately visible. We need to navigate to a specific product page to see the customer reviews and ratings.

Act 7: click [10309]

Thought 9: If the customer reviews section is not immediately visible, scrolling down will help to reveal it.

Act 9: scroll [down]

Obs 9: StaticText 'Rating: [10513] generic '75%'

Act 10: type [10513] [4 or 5] [0]

Obs 10: [10415] LayoutTable

Thought 11: we need to check each product's reviews to find the relevant information. Act 11: click [10415]

Thought 15: The ratings and reviews for this specific product are not relevant to the objective. we need to navigate back to the search results page to continue the search.

Act 15: tab_focus [0]

Score: 0.0
