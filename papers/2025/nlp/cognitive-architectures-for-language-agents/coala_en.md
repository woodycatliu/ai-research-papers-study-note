# Cognitive Architectures for Language Agents (CoALA)

## Paper Information
- **Authors**: Theodore R. Summers, Shunyu Yao, Karthik Narasimhan, Thomas L. Griffiths
- **Institution**: Princeton University
- **Published**: 2025
- **Field**: Natural Language Processing / AI Agents
- **Paper Link**: [OpenReview](https://openreview.net/forum?id=1i6ZCvflQJ)

## Abstract

Recent work has enhanced Large Language Models (LLMs) with external resources (e.g., the internet) or internal control flows (e.g., prompt chaining) to perform tasks requiring grounding or reasoning, giving rise to a new class of language agents. Despite their empirical successes, we lack a framework for organizing existing agents and planning future developments. In this paper, we draw on the rich history of cognitive science and symbolic AI to propose **Cognitive Architectures for Language Agents (CoALA)**. CoALA describes language agents as having:

- **Modular memory components**
- **Structured action spaces** for interacting with internal memories and external environments
- **Generalized decision processes** for action selection

## Key Contributions

1. **Conceptual Framework**: Introduces CoALA as a unifying framework for language agents
2. **Historical Grounding**: Connects modern language agents to cognitive architectures and production systems
3. **Systematic Organization**: Provides a structured way to analyze and compare existing agents
4. **Future Directions**: Identifies pathways toward more powerful language-based agents

## Methodology

### CoALA Framework Components

#### 1. Memory Modules
- **Working Memory**: Temporary storage for current context and processing
- **Long-term Memory**: Persistent storage for knowledge and experiences
- **Episodic Memory**: Storage of specific experiences and events
- **Semantic Memory**: Storage of factual knowledge and concepts

#### 2. Action Spaces
- **Internal Actions**: Operations on memory systems
- **External Actions**: Interactions with the environment
- **Structured Interfaces**: Defined protocols for action execution

#### 3. Decision Processes
- **Action Selection**: Mechanisms for choosing appropriate actions
- **Control Flow**: Managing the sequence of operations
- **Meta-cognition**: Higher-level reasoning about reasoning

### Connection to Production Systems
- **LLMs as Probabilistic Production Systems**: LLMs define probability distributions over string extensions
- **Prompt Engineering as Control Flow**: Guiding LLM behavior through structured inputs
- **Agent Evolution**: From predefined prompt chains to interactive feedback loops

## Experiments

### Agent Analysis
- Retrospective survey of existing language agents
- Classification using CoALA framework
- Identification of common patterns and architectures

### Framework Validation
- Demonstration of CoALA's explanatory power
- Analysis of agent capabilities and limitations
- Comparison across different agent types

## Results

### Taxonomic Organization
- Successful categorization of diverse language agents
- Clear identification of architectural patterns
- Insights into design trade-offs and choices

### Future Directions
- Pathways toward more sophisticated agents
- Integration of cognitive principles
- Scaling to general intelligence

## Personal Notes

### Key Insights
- **Historical Perspective**: Valuable connection to cognitive architectures and production systems
- **Unifying Framework**: Provides common vocabulary for agent research
- **Practical Applications**: Guides design of new language agents

### Related Work
- Cognitive architectures (Soar, ACT-R)
- Production systems
- Language model architectures
- Agent-based AI systems

### Future Research
- Implementation of CoALA-based agents
- Integration with multi-modal capabilities
- Scaling to complex real-world tasks

## Implementation Notes

- Framework is conceptual rather than prescriptive
- Can be applied to analyze existing agents
- Provides blueprint for future agent development

## Figures and Images

The paper includes several important diagrams and figures stored in the `images/` directory:
- CoALA framework overview
- Agent architecture comparisons
- Memory system diagrams
- Decision process flowcharts

---

*Note: This paper provides a conceptual framework rather than specific technical implementations. The value lies in its organizational and theoretical contributions to the field of language agents.*