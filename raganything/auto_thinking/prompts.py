"""Prompt templates used by the auto-thinking module."""
#存放复杂度分析，function call调用，检索方式判断，子问题分解，最终合成流程的提示词。
COMPLEXITY_PROMPT = """\
You are a routing guard for a RAG system.
Classify the following user query into one of three buckets:
- low: short factoid or single-entity lookup that needs little reasoning.
- medium: requires some composition but still mainly one focus.
- high: contains multiple sub-questions, comparisons, causal or strategic reasoning, or needs combining many facts.

Return a JSON object with keys:
  complexity ("low"|"medium"|"high"),
  confidence (float between 0 and 1),
  reasons (list of short bullet explanations).

Query:
\"\"\"{query}\"\"\"
"""

LOOKUP_ROUTER_PROMPT = """\
You are an efficient function call selector, determining whether to invoke the corresponding function on the parsed PDF content.

Return strict JSON:
{
  "action": "none" | "page_lookup" | "image_position_lookup",
  "page": "page number or range like 17-25 (optional)",
  "position": "top" | "bottom" | "left" | "right" | null,
  "reasons": ["short bullets"]
}

Rules:
- If the query asks about specific page(s) or a page range (e.g., "page 17" or "pages 17-25"), choose "page_lookup" and fill page with the number/range.
- If the query asks about an image at a specific position on a page (e.g., "page 3 top image", "page 5 left-most picture"), choose "image_position_lookup", fill page if given, and choose a position among top/bottom/left/right. If position is not stated, leave it null.
- Otherwise choose "none".
- Do not invent pages or positions that are not mentioned or implied.

Query:
\"\"\"{query}\"\"\"
"""

QUESTION_DECOMPOSITION_PROMPT = """\
You are an advanced query decision system. Your task is to decide which form my query should take. There are two possibilities:
Traditional RAG approach: The query can be directly used for RAG retrieval (computing semantic similarity is sufficient to achieve the most accurate retrieval).
Non-traditional approach: The query needs to be understood and analyzed to determine whether it should be split or transformed into another expression, achieving more accurate retrieval without relying on traditional RAG.
What is the traditional RAG retrieval strategy:
RAG performs retrieval-augmented generation by computing embedding semantic similarity as context. It relies solely on semantic similarity for retrieval and is relatively rigid.
Other strategies:
Plain RAG only computes semantic similarity rather than semantic relatedness. Therefore, you must be able to decompose and understand the question. If the query can be transformed into another expression or split into sub-questions for retrieval—yielding better results than answering via similarity-based RAG then use this approach.
Let's think step by step to see what form this question needs to be decomposed into for retrieval.
Output strict JSON:
{{
  "strategy": "dependent" | "independent",
  "sub_questions": [
    {{
      "id": "step-1",
      "question": "...",
      "rationale": "...",
      "depends_on": ["step-0"]
    }},
    ...
  ],
  "notes": "global remarks or constraints (optional)"
}}

Guidelines:
- Keep ids unique (step-1, step-2, ...). For dependent flows, fill depends_on with prerequisite ids (omit or [] when none).
- Use "dependent" if any later steps truly require earlier results; otherwise choose "independent".
- Sub-questions must stay grounded in the documents; do not invent context.
- Provide concise rationales describing why each step is necessary and how it will be solved.
- When a question references a specific page, figure, image, or paragraph, create an initial locating step before any detail extraction from that section.
- When a task requires counting, analysing, or comparing multiple (or all) images/figures/tables/pictures, create an initial step that locates all required visuals (recording identifiers, page references, and image paths) before any processing or comparison.
- If the query is simple, output exactly one sub-question identical to the original query and set strategy to "independent".
- Do not create duplicate sub-questions when decomposing the query; each sub-question must be semantically distinct and non-redundant.

Original query:
\"\"\"{query}\"\"\"
Maximum steps allowed: {max_steps}
"""

FINAL_SYNTHESIS_PROMPT = """\
Return JSON:
{{
  "final_answer": "complete response for the original user query",
  "confidence": 0.0-1.0,
  "reasoning": "how the evidence supports the answer",
  "citations": [
    {{"sub_id": "step-1", "support": "what was used"}}
  ]
}}

Original query:
\"\"\"{query}\"\"\"

Plan:
{plan}

Sub-question answers:
{answers}

The above are the answers to several sub-questions. Please integrate these sub-question answers and use them to answer the original query, summarizing the key points.

Rules:
1. Prioritize integrating substantial content from the sub-answers. If the combined content from all sub-questions cannot support the final answer, respond with "Not answerable"
2. Keep the final answer in the same language as the original query when possible.
3. Do not invent content on your own; answer based on the content retrieved.
4. The value of "final_answer" must be a concrete, direct answer to the original query (e.g., including key numbers/entities when relevant). Do NOT output generic meta-text such as "complete response for the original user query" or copy the example texts from this specification.
5. If both the target query and the retrieved content contain a specific date, answer only if the evidence explicitly includes the target date stated in the question and it exactly matches at the year/month/day level. If the date (year/month/day) retrieved in the sub-question answer does not exactly match the target date in the query, return "Not answerable". Do not guess, modify, or substitute dates.
6. Return ONLY a valid JSON object with a single key final_answer (string). Do not output any extra text, commentary, or fields.
"""

RETRIEVAL_STRATEGY_PROMPT = """\
You are an efficient retrieval method selector, helping a RAG choose between BM25 retrieval, embedding vector retrieval.

Rules for choosing retrieval methods for entity and relation retrieval:

BM25 Retrieval:
Used for pure text questions that do not require multimodal content.
Best for simple questions containing nouns that can be directly matched, with few synonyms.
Also suitable when the question involves retrieving concrete details from the text.
When the query contains many proper nouns or domain-specific terms, BM25 is preferred because direct lexical matching works well.

Embedding (Vector) Retrieval:
Used for all multimodal retrieval tasks, as well as text-only questions that require semantic understanding.
Also suitable for more complex text questions or questions with fewer proper nouns where lexical matching is insufficient.

Rules for choosing chunk retrieval:

Chunk retrieval always uses embedding-based retrieval.

Directly return strict JSON with keys "entity_retrieval", "relation_retrieval", "chunk_retrieval".
Each value must be one of: "bm25", "embedding",and cannot be empty.

Query:
\"\"\"{query}\"\"\"
"""
