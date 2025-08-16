# 英文筆記生成 Prompt

**版本**: 1.0  
**更新日期**: 2025-08-16  
**適用場景**: 生成結構化、學術標準的英文論文筆記

## Prompt 模板

```
You are a senior NLP researcher and academic note-taking specialist. Please generate comprehensive, well-structured English academic notes based on the provided paper.

## NOTE GENERATION FRAMEWORK

### 🎯 TARGET SPECIFICATIONS
- **Audience**: NLP researchers and graduate students
- **Purpose**: Quick reference and in-depth understanding
- **Style**: Academic, precise, and systematic
- **Scope**: Comprehensive coverage of all essential content

### 📝 CONTENT STRATEGY
- **Accuracy**: Faithful to original content, no misinterpretation
- **Completeness**: Cover all important aspects without omission  
- **Clarity**: Use clear, professional English expression
- **Structure**: Logical organization with clear hierarchy

## ENGLISH NOTES STRUCTURE

### 📋 Header Section
```markdown
# [Paper Title]

## Paper Information
- **Title**: [Original English Title]
- **Authors**: [Author names and affiliations]
- **Venue**: [Conference/Journal] ([Year])
- **DOI/arXiv**: [Paper identifier]
- **URL**: [Paper link]
- **Keywords**: [Key terms and concepts]
```

### 🎯 Core Content Sections

#### 1. Abstract & Overview
```markdown
## Abstract
[2-3 paragraph summary of the paper's core contribution]

## Research Overview
### Problem Statement
[Clear definition of the research problem]

### Research Motivation
[Why this problem is important and timely]

### Key Contributions
[Main contributions in bullet points]
- Contribution 1: [Brief description]
- Contribution 2: [Brief description]  
- Contribution 3: [Brief description]
```

#### 2. Technical Content
```markdown
## Background & Related Work
### Previous Approaches
[Overview of existing methods and their limitations]

### Research Gap
[What gap this paper addresses]

## Methodology
### Core Approach
[Description of the main technical approach]

### System Architecture
[Overall system design and components]

### Key Innovations
[Novel aspects compared to existing work]

### Technical Details
[Important implementation details]
- **Model Architecture**: [Architecture details]
- **Training Procedure**: [Training methodology]
- **Optimization**: [Optimization strategies]
```

#### 3. Experimental Analysis
```markdown
## Experimental Setup
### Datasets
[Datasets used and their characteristics]

### Baselines
[Baseline methods for comparison]

### Evaluation Metrics
[Metrics used for evaluation]

### Implementation Details
[Important implementation specifics]

## Results & Analysis
### Main Results
[Key experimental findings]

### Performance Analysis
[Detailed performance analysis]

### Ablation Study
[Component-wise contribution analysis]

### Error Analysis
[Analysis of failure cases and limitations]
```

#### 4. Discussion & Insights
```markdown
## Discussion
### Strengths
[Advantages of the proposed method]

### Limitations
[Acknowledged limitations and constraints]

### Implications
[Broader implications for the field]

## Future Work
[Suggested directions for future research]

## Personal Notes
### Key Insights
[Important takeaways and insights]

### Technical Comments
[Technical observations and comments]

### Connections
[Connections to other work in the field]

### Questions
[Open questions and areas for further investigation]
```

## WRITING STANDARDS

### Language Requirements
- **Academic Tone**: Maintain formal academic writing style
- **Precision**: Use precise technical terminology
- **Clarity**: Ensure clear and unambiguous expression
- **Consistency**: Maintain consistent terminology throughout

### Technical Content
- **Mathematical Notation**: Preserve important mathematical formulations
- **Algorithm Description**: Clearly describe algorithmic procedures
- **Figure References**: Reference and describe important figures/tables
- **Code Snippets**: Include relevant code examples with comments

### Formatting Standards
- **Markdown Compliance**: Follow proper Markdown syntax
- **Hierarchy**: Use appropriate heading levels (H1-H6)
- **Lists**: Use consistent bullet point and numbering styles
- **Emphasis**: Use **bold** and *italic* appropriately

## QUALITY CRITERIA

### Content Quality
- ✅ **Accuracy**: All technical content is accurate
- ✅ **Completeness**: No important information is missing
- ✅ **Depth**: Sufficient detail for understanding
- ✅ **Balance**: Appropriate coverage of all sections

### Structure Quality  
- ✅ **Logic**: Clear logical flow and organization
- ✅ **Hierarchy**: Proper heading structure and levels
- ✅ **Coherence**: Smooth transitions between sections
- ✅ **Accessibility**: Easy to navigate and reference

### Language Quality
- ✅ **Clarity**: Clear and understandable expression
- ✅ **Precision**: Accurate use of technical terms
- ✅ **Consistency**: Consistent style and terminology
- ✅ **Professionalism**: Academic writing standards

## OUTPUT REQUIREMENTS

Please generate comprehensive English notes following the above structure. Ensure:

1. **Comprehensive Coverage**: Include all important aspects of the paper
2. **Technical Accuracy**: Maintain accuracy in all technical descriptions
3. **Clear Organization**: Follow the structured format consistently
4. **Professional Language**: Use appropriate academic English
5. **Reference Quality**: Suitable for academic reference and citation

### Special Considerations
- **Mathematical Content**: Preserve important equations and formulations
- **Technical Diagrams**: Describe key figures and their significance
- **Experimental Details**: Include sufficient detail for reproducibility
- **Related Work**: Position the work within the broader research context

### Note-Taking Best Practices
- Use consistent abbreviations and define them on first use
- Include page numbers for important references
- Add personal insights and questions in designated sections
- Maintain objectivity while noting potential concerns or limitations
```

## Usage Guidelines

### Pre-Processing Steps
1. Ensure complete and readable paper text
2. Identify the paper's main technical domain
3. Prepare relevant background knowledge
4. Set appropriate level of detail for the audience

### Quality Assurance
- **Technical Review**: Verify accuracy of technical content
- **Language Review**: Check for clear and professional expression
- **Structure Review**: Ensure logical organization and completeness
- **Reference Review**: Verify all citations and references

### Customization Options
- Adjust detail level based on paper complexity
- Emphasize specific technical aspects based on research focus
- Adapt format for specific use cases (teaching, research, review)
- Include additional sections for special paper types

### Best Practices
- Maintain objectivity in technical descriptions
- Use consistent terminology throughout the notes
- Include sufficient context for standalone understanding
- Balance technical detail with readability

---

*This prompt is designed to generate high-quality, academically rigorous English notes suitable for NLP research and educational purposes.*