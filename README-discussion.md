# Typeform RAG Prototype - Discussion

## 1. Ideation & Scoping

### Problem Framing
The challenge is to create an AI-powered assistant that can understand natural language questions and provide accurate, contextual answers from Typeform's Help Center content.

**Key Prototyping Considerations**

The prototype prioritizes rapid iteration by focusing on end-to-end functionality over optimization. Each experiment run generates a new timestamped RUN_ID, with all artifacts and evaluations saved to a dedicated directory to enable reproducibility, iteration, and debugging. The approach uses a monolithic interactive cell-based file containing the full workflow for easy end-to-end experimentation and debugging, with main workflow steps organized into separate Python modules for basic code organization without excessive modularity. The design strives to remain provider-agnostic by minimizing dependencies on specific LLM providers, using LiteLLM that allow for easy switching between models.

**Scaling Risks & Constraints**

Several constraints limit the prototype's scalability. API rate limits from Vertex AI and Pinecone could bottleneck high traffic, and this prototype makes no attempt to mitigate such limitations. Cost scaling presents another challenge, as both LLM and embedding API costs grow linearly with usage, while Pinecone storage costs increase with content volume. Latency becomes a concern when scaling real-time query embedding and generation to thousands of concurrent users - the prototype processes test queries sequentially rather than in parallel so this would have to change when going to prod. Additionally, in the prototype we make no attempt to automate re-indexing: in prod, this would be mandatory to ensure content freshness. Lastly, the prototype does **not** optimize chunk retrieval/selection/pruning for inclusion in the prompt - this is an inneficient use of prompt tokens and would need to be improved in production not just for cost considerations but also because in prod we would have order of magnitude more articles. 



### Success Metrics
We can measure the success of the RAG system across three dimensions: the **Quality** of the generated answers, the **Performance** of the system at scale, and overall **User Experience**.



**1. Quality and Factual Integrity**

Measures the efficacy of both the retrieval and generation components, primarily utilizing the RAGAS framework metrics - these are metrics computed **per query**. To measure the system over all queries posed to it, we compute distribution metrics (mean, median, quantiles, etc) for each RAGAS metric. 

In the table below, I comment on whether or not we require Ground-Truth (GT) to compute a metric or if LLM-as-a-judge (LLM-J) is sufficient. Note that while GT data provides the highest precision for evaluating **all** RAG metrics below, it is costly and time-intensive to produce. For scalability, the LLM-as-a-Judge (LLM-J) methodology provides a reliable and scalable approximation for some metrics.

| Metric | Definition | Component Assessed | Scoring Methodology |
| :--- | :--- | :--- | :--- |
| **Factual Accuracy** (a.k.a. **Faithfulness**) | Percentage of generated statements that are factually supported *only* by the retrieved context. | **Generator** (mitigates hallucination) | **LLM-J:** Can be assessed without GT by checking the answer solely against the retrieved context using **another** LLM as a judge. |
| **Answer Relevance** | Measures how well the final response directly addresses the user's question, excluding irrelevant content. | **Generator** | **LLM-J:** Can be assessed without GT, against the original question and the generated answer. |
| **Context Precision** | The signal-to-noise ratio of the retrieved documents; the ratio of relevant chunks among the retrieved set. | **Retriever** (measures search precision) | **LLM-J:** Can be assessed (without GT) against the original question and the retrieved context. |
| **Context Recall** | The extent to which the retrieved context covers **all** necessary information required to form a complete and correct answer. | **Retriever** (measures search coverage) | **Ground Truth (GT) Required:** Must be assessed by comparing the retrieved context against a human-written/annotated "ideal" set of facts or a GT answer. |
| **Overall Accuracy** | The final percentage of questions answered correctly against a pre-determined gold standard. | **End-to-End** | **Ground Truth (GT) Required:** Must compare the generated answer directly against a human-written/verified GT answer. |
| **Refusal Rate** (a.k.a. **Fallback Rate**) | The proportion of queries for which the RAG system produces a canned, non-factual response (e.g., "I cannot answer that," "Information is not available"). | **System Safety & Coverage** (measures utility) | **LLM-J or Pattern Matching:** Can be assessed (without GT) by using a specialized LLM-Judge to classify the answer as a correct/incorrect refusal, or by simple pattern matching against known refusal phrases. |

<i>Note: One way to construct a ground truth dataset in a 'cheaper' way than relying solely on human annotation is to use an LLM to 'go backwards': instead of the LLM generating answers to questions from context, we first select a set of source documents and prompt the LLM to perform two steps. First, the LLM generates a question that can be answered using only the content of the selected document. Then, it generates the definitive, context-supported answer to that question. This is called Synthetic Data Generation or Back-Generation and it builds GT-like triplets of {Question}-{Answer}-{Relevant Context}, which are then used as the gold standard for evaluating a RAG pipeline's performance.</i>



**2. Performance and Scalability**

These metrics ensure the system can handle production load efficiently and cost-effectively.

* **End-to-End Latency (Response Time):** The total time elapsed from the user submitting the query to receiving the final response. Target should be sub-second response times for interactive use.
* **Knowledge Coverage:** The percentage of all relevant Help Center content that the retriever can effectively access and utilize when queried. Note that this fundamentally requires some kind of ground truth.
* **Operational Cost:** Tracking the cost per query, specifically monitoring API usage for the Embedding Model and the Large Language Model.



**3. User Experience**

These metrics validate the real-world value and utility of the RAG system.

* **User Satisfaction (Qualitative Feedback):** Measurement of user-provided feedback (e.g., thumbs-up/down, explicit survey data) on the helpfulness, tone, and overall quality of the generated answers.
* **Adoption Rate:** The frequency and volume of successful query resolution achieved by the RAG system, without human customer service rep involvement.



---

## 2. Data Exploration & Chunking Strategy

### Help Center Content Processing
The prototype processes two sample Help Center articles:
- "Create multi-language forms" 
- "Add a Multi-Question Page to your form"



**Preprocessing & Normalization**
1. **HTML Parsing**: Extract main content using BeautifulSoup, removing navigation, ads, and boilerplate

2. **Text Normalization**: 
   - Case normalization (lowercase)
   - Whitespace standardization
   - Unicode normalization (NFC)
   - Placeholder substitution for URLs/dates

3. **Metadata Addition**: Document ID, title, URL, timestamp, word count, language



**Chunking Method**

The prototype employs a simple recursive character text splitter from LangChain as its chunking strategy. Chunks are configured to 512 tokens (configurable based on embedding model limits, context size of the answerer LLM, and retrieval SLAs), with a 100-token overlap (also configurable) to maintain context continuity across chunk boundaries. Each chunk carries comprehensive metadata including a unique `chunk_id`, `chunk_position_index` indicating its order within the document, `parent_document_id` for source document reference, the actual `chunk_text` content, and the `doc_title` from the parent document.

This simple approach is adequate for the initial articles, which have relatively flat structures. However, a proper analysis across a large set of help articles, looking at the full document hierarchy of each (likely involving HTML/Markdown-element-based or hierarchical chunking) will be mandatory for a production-grade system to ensure retrieval accuracy across diverse Help Center content types.

<br> 

---

## 3. Embedding & Vector Search

**Embedding Model Choice**

The prototype uses Google's `text-embedding-005` via Vertex AI, selected for its high-quality semantic understanding and optimization for retrieval tasks. The model produces 768-dimensional vectors, providing a good balance between performance and accuracy. Its native Google Cloud integration simplifies deployment and management within the existing infrastructure.



**Indexing & Search Strategy** - uses Pinecone
- **Index Configuration**:
  - Dimension: 768 (matching embedding model)
  - Metric: Cosine similarity
  - Cloud: AWS us-east-1
- **Search Process**:
  1. Embed user query with same model
  2. Query Pinecone with top_k (configurable)
  3. Return chunks with (cosine) similarity scores and metadata



**Search Relevance Evaluation**
- **Current Approach**: Manual evaluation of retrieved chunks. We track the number of citations, we check that citations are valid (are a subset of retrieved chunks).
- **Future**: Automated evaluation using something like RAGAS (Retrieval-Augmented Generation Assessment) framework discussed above.



**Experimenting with chunking/embedding**

The prototype uses a **delete-and-recreate** strategy for Pinecone indexes: each experiment run deletes the existing index and creates a fresh one with the same name. This is simple for rapid prototyping, makes sure we stay within the limits of Pinecone's free tier and avoids index management complexity. 

However, in a more developed system we need **run-specific index versioning**. A simple system would be using a standardized naming format that matches our experiment ID tracking: `{base-name}-{run_id}` (e.g., `typeform-help-rag-20250118_143022_UTC_abc123`). This would preserve experiment data for comparison, rollback, and reproducibility and would support A/B testing of different embedding models and chunking strategies.



---

## 4. Prompt Design & AI Response Generation

In this prototype I used Gemini Flash via Vertex AI (accessed through LiteLLM) to generate responses. It has a reputation of very good instruction following, and incorporating context, and it is fairly cost-effective.

In designing the prompt, we want to be explicit about what information the model can and cannot use. We do this by providing context (documents) and requiring the LLM to cite its sources for every fact listed. See the `prompts/` directory for the templates used. 

We incorporate context in the prompt by retrieving the top k most relevant chunks and concatenating them with clear separators for the LLM to process. The system requires citations for each factual claim to ensure traceability and accuracy. When context is insufficient or irrelevant to the question, we specify a clear fallback message to maintain system reliability.

Since we have a 100 (configurable) token overlap, some information gets duplicated, increasing the number of tokens passed to the LLM at query time - this should be improved in a production environment (eg: remove exact repetition) in order to reduce the number of tokens used and free up context capacity. 

The current prototype will fail when the LLM runs out of context tokens, a critical issue for long articles or high top-k retrieval. In a production system, we must handle this and can do so through several strategies. 
- Context shortening and summarization techniques can run conditionally before the final generation step 
- Use a Re-Ranker Model (often a small transformer model like a cross-encoder) to score the relevance of retrieved chunks to the question, then select only the most vital, non-redundant information to pass to the LLM. 
- If the re-ranked context still exceeds a threshold (e.g., 75% of the LLM's context window), the LLM can be instructed to perform an intermediate summarization step, extracting only the core facts relevant to the user's question to create a more concise, token-efficient prompt. 
- Finally, we could use adaptive retrieval: adjust the top_k value dynamically based on query complexity or length, reducing context window usage for simpler questions.



**Prompt Reliability & Fallbacks**
*What do you think about prompt reliability and fallbacks?*

Prompt Reliability is the LLM's ability to consistently adhere to the instructions. The current prototype relies on the System Prompt for instruction following (e.g., citing sources, answering based only on context, and answer formatting instructions). 

Reliability can be increased through several approaches. 
- Constraint enforcement: using structured output formats (Pydantic models, native model-specific API structured output capabilities) to ensure citations and refusal messages are consistently present. 
- Model selection: continuously benchmark instruction-following models (e.g., Gemini vs. others) on complex, nuanced prompts and periodically switching models when needed. 
- Architectural changes: introduce a checking step before an answer is shown to the user, providing an additional quality gate. 

The current fallback is a simple hardcoded refusal message for when the retrieved context is insufficient or irrelevant to the question. We can expand this with several strategies. A low confidence fallback could trigger when the LLM-Judge (or an internal confidence score, or the cosine similarity metric) rates a generated answer's Faithfulness below a set threshold (e.g., <0.6), causing the system to default to the safe refusal message instead of providing an answer below the desired confidence level. For non-Typeform-specific questions, the system could fall back to a general web search or knowledge base outside the Help Center content. Human handoff mechanisms could trigger for queries leading to repeated refusals or low-confidence scores, routing users to human customer service representatives.



---

## 5. Evaluation & Quality Metrics

**Quality Measurement - Heuristics & Metrics**
*How do you measure the quality of answers? What heuristics or metrics would you use?*

Please see Question 1 - Success Metrics for a discussion of this.



**Stress Testing Strategy**
*How would you stress test for accuracy, hallucination, or robustness?*

Stress testing would involve creating a diverse test dataset and running targeted evaluation sprints.

| Stress Test Category | Query Strategy | Metric Focus | Rationale |
| :--- | :--- | :--- | :--- |
| **Accuracy & Coverage** | **Question Generation (Synthetic Data):** Generate Q\&A pairs from all indexed documents. | **Context Recall** & **Overall Accuracy** | Validates the system's ability to answer questions it *should* be able to answer. |
| **Hallucination** | **Out-of-Scope (OOS) Queries:** Ask questions about non-Typeform topics (e.g., competitor products, general history). | **Faithfulness** & **Refusal Rate** | Tests the system's ability to correctly *refuse* to answer or, if it does, that the answer has low Faithfulness (indicating potential hallucination). |
| **Robustness** | **Ambiguous/Complex Queries:** Queries requiring multi-hop reasoning or containing deliberately vague/misleading language. | **Context Precision** & **Answer Relevance** | Tests the retriever's ability to find and filter multiple, complex pieces of information and the generator's ability to synthesize them. |
| **"Needle in a Haystack"** | Embed a single critical fact in a very long, irrelevant chunk of text. | **Context Precision** | Tests the system's ability to focus on the key information despite high noise (common challenge). |

---



**Production Observability**
*How would you implement observability and evaluation of an AI feature in a production environment?*

In a production environment, observability should be continuous and driven by real user traffic.

The system currently implements:
- Request/response logging with timestamps
- Retrieval traces with similarity scores
- Generation metrics (tokens, time, model used)

We can add:
1.  **Production Quality Tracing:** Integrate the core RAGAS metrics to run asynchronously on a sample of live user queries (using an LLM-J in the background). This provides real-time quality scores.
2.  **Drift Detection:** Monitor the distribution (mean/median) of RAGAS metrics and have automated alerts trigger when these metrics drop by a significant margin signaling a potential drift in user queries or an unexpected change in LLM behavior.
3.  **Human Feedback Loop (Systematized):** Every user interaction should include an optional Thumbs-Up/Thumbs-Down user feedback mechanism. Although this would collect a biased and sparse signal, it is essential in capturing.. AICIC AICIC
4.  **Cost Monitoring/Alerting:** Implement dedicated logging to track model usage on a per-query and daily/weekly basis, with alerts to prevent unexpected cost overruns due to model or traffic surges.



---

## Improvements

I discuss three core areas for improvement: scaling, establishing a measurable quality benchmark, and implementing other RAG techniques to maximize answer accuracy.

### 1. Scale
* **Asynchronous Query Answering:** Implement async capabilities across query embedding, chunk retrieval, and question answering to allow for parallel processing of incoming queries and support a large number of concurrent users.
* **Caching Strategy:** Implement Redis caching for frequently requested queries and generated embeddings to reduce latency and minimize reliance on expensive API calls.
* **Cost Monitoring:** Establish a system to actively track API usage and underlying infrastructure costs for sustained operational efficiency.

### 2. Continuous Evaluation

* **Evaluation Dataset Creation:** Build a robust test set of user questions paired with ground truth Q\&A pairs to serve as the benchmark for system performance.
* **RAGAS Implementation:** Integrate the RAGAS framework to implement comprehensive metrics, including **Faithfulness**, **Answer Relevance**, **Context Precision**, and **Context Recall**.
* **Automated Evaluation Pipeline:** Build a continuous testing loop using a combination of the evaluation dataset, Human-in-the-Loop feedback (a systematic collection process), and LLM-as-a-Judge mechanisms.
* **Alerting System:** Develop an automated alerting system to signal immediate quality degradation or critical system failures in production.

### 3. Answer Quality Improvement

#### **Retrieval & Context Optimization**

* **Hybrid Search:** Implement a combination of vector search and keyword search to improve document recall and handle various query styles.
* **Advanced Chunking Strategies:** Utilize Parent-Child Chunking (ParentDocumentRetriever) to retrieve small, relevant chunks while feeding a larger, full context to the LLM for improved generation.
* **Source Hierarchy Tracking:** Capture and leverage the document's section structure (e.g., headings) during retrieval for better contextual grouping.
* **Embedding Analysis:** Use visualization techniques like t-SNE/UMAP to visually analyze chunk embeddings, helping diagnose clustering issues and optimize the embedding model.

#### **Generation & Content Improvement**

* **Chain-of-Thought Prompting:** Apply advanced prompting for complex queries to guide the LLM through step-by-step reasoning, especially if a user query is about troubleshooting an error or a very complex task.
* **Confidence Scoring:** Integrate a system to provide confidence levels for generated answers, enabling fallback strategies for low-confidence outputs.
* **State-graph Integration:** Implement a state-graph-like architecture for iterative reasoning, including context sufficiency checks, query rewriting for ambiguity, and safe fallback mechanisms.
* **Intelligent Preprocessing:** Improve text normalization with acronym expansion and add intent classification to route queries to specialized RAG processes (assumes we build various RAG processes for different use cases).

#### **Knowledge Base Management**

* **Content Versioning:** Implement tracking for changes in Help Center articles to trigger automated re-indexing and ensure the knowledge base remains current.
* **Multi-language Support:** Dedicate resources to testing non-English queries and content to assess and build out full multi-lingual support.