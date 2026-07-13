# ECLRR-v4 Architecture Image Prompt

The README Mermaid diagram is the source of truth for exact labels and flow.
The prompt below is intended for generating a polished explanatory illustration.
Verify or overlay small text in a design tool after generation because image
models may distort technical labels.

## Main prompt

```text
Create a publication-quality 16:9 technical architecture infographic explaining
ECLRR-v4, Evidence-Carrying Long-Chain Relation Refinement, for an AI knowledge
graph and RAG system. Use a clean dark navy background (#05070D), high-contrast
white typography, cyan for verified source facts, amber for evidence packages,
violet for LLM reasoning, green for accepted deterministic checks, red dashed
lines only for promoted inferred relations, and muted gray for rejected paths.
No gradients, no glow clouds, no decorative orbs, no 3D perspective, no stock
illustration, and no anthropomorphic robots.

Compose the image as one left-to-right evidence pipeline with seven numbered
stages and a narrow audit rail along the bottom:

1. FACT GRAPH VIEW. Show a compact knowledge graph containing people, events,
objects, places, and organizations. Solid cyan edges are original source facts.
Fuzzy edges are muted dashed gray triggers, but are clearly excluded from proof
paths. Promoted and legacy synthetic edges are also excluded from recursive
proof.

2. DETERMINISTIC 3-8 HOP BEAM SEARCH. Highlight one continuous path
A -> B -> C -> D -> E across multiple entity types. Show relation direction,
inverse traversal markers, stable ranking, hub penalties, and a minimal endpoint
slice A -> E. Communicate that paths are loop-free and relation-aware.

3. EVIDENCE PACKAGE. For every hop, attach a small document-chunk tile with an
exact quote, chunk_id, start/end offsets, and edge_id. Use one unbroken evidence
chain across different chunks. A missing middle chunk visibly blocks the path.

4. GENERATOR. Show a constrained JSON proposal containing source, target,
canonical relation, relation family, direction, description, and evidence_refs.
Label it PROPOSE ONLY, NOT APPROVE. Indicate temperature 0 and a 14,000-character
prompt budget with primary evidence never truncated.

5. INDEPENDENT JUDGE. Show a separate review module reading the original
EvidencePackage plus Proposal. It outputs accept, revise, or reject and a
10-point score split into evidence coverage 0-4, semantic composability 0-3,
relation direction 0-2, and uncertainty calibration 0-1.

6. DETERMINISTIC GATE. Show code-like verification checkpoints: real nodes,
continuous 3-8 hop fact path, every hop grounded, quote exact substring,
direction valid, canonical relation not duplicated, no conflict, total >= 8,
and evidence coverage = 4. Failed checks flow to NO-OP and never enter the graph.

7. ATOMIC PROMOTION AND RAG RETRIEVAL. Split accepted output into CREATE or
REFINE, then atomically write to Graph Store and Relationship Vector Index with
rollback protection. Show the final promoted relation as a crisp red dashed edge
between A and E. Its vector record contains endpoints, canonical relation,
direction, evidence-chain description, and chunk IDs. A user question retrieves
the promoted edge together with its original document chunks and sends both to
the answer LLM.

Along the bottom, add an AUDIT TRAIL rail receiving copies from ReviewItem,
EvidencePackage, Generator prompt/output, Judge prompt/output, deterministic
Gate, and create/refine/no-op result. Make clear that ReviewItem, Proposal, and
JudgeDecision exist only in memory and audit files; there is no candidate or
pending edge state in the knowledge graph.

Use precise grid alignment, thin connector lines, generous spacing, readable
Chinese-compatible sans-serif typography, WCAG-AA contrast, and restrained
professional styling suitable for a research README or paper figure. The visual
conclusion should be immediately clear: no inferred edge is written unless an
unbroken chain of original chunks survives two LLM stages and deterministic
verification.
```

## Negative prompt

```text
Avoid neural-network brain icons, generic AI circuitry, fantasy imagery,
photorealistic people, excessive neon bloom, gradients, glassmorphism, tiny
unreadable paragraphs, circular process diagrams, random node clouds, candidate
edges stored in the graph, fuzzy edges used as proof, promoted edges recursively
proving new edges, and any implication that the Judge score is a probability.
```
