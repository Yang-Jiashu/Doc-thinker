export interface RawGraphNode {
  id?: unknown;
  entity_id?: unknown;
  label?: unknown;
  type?: unknown;
  entity_type?: unknown;
  description?: unknown;
  source_id?: unknown;
  file_path?: unknown;
  degree?: unknown;
  is_expanded?: unknown;
  is_image_node?: unknown;
}

export interface RawGraphEdge {
  id?: unknown;
  source?: unknown;
  target?: unknown;
  src_id?: unknown;
  tgt_id?: unknown;
  label?: unknown;
  relation?: unknown;
  relation_family?: unknown;
  direction?: unknown;
  description?: unknown;
  source_id?: unknown;
  weight?: unknown;
  is_discovered?: unknown;
  is_promoted?: unknown;
  edge_kind?: unknown;
  review_status?: unknown;
  provenance?: unknown;
  algorithm_version?: unknown;
  relation_id?: unknown;
  canonical_key?: unknown;
  path_used?: unknown;
  supporting_paths?: unknown;
  evidence_chain?: unknown;
  evidence_chunk_ids?: unknown;
  judge_scores?: unknown;
  decision_score?: unknown;
}

export interface RawGraphResponse {
  nodes?: RawGraphNode[];
  edges?: RawGraphEdge[];
  links?: RawGraphEdge[];
  metadata?: Record<string, unknown>;
}

export interface GraphNode {
  id: string;
  label: string;
  type: string;
  description: string;
  sourceId: string;
  filePath: string;
  degree: number;
  group: number;
  color: [number, number, number];
  size: number;
  isExpanded: boolean;
  isImageNode: boolean;
}

export interface GraphEdge {
  id: string;
  source: number;
  target: number;
  label: string;
  description: string;
  sourceId: string;
  weight: number;
  isDiscovered: boolean;
  isPromoted: boolean;
  kind: "original" | "eclrr_v4";
  relationFamily: string;
  direction: string;
  relationId: string;
  canonicalKey: string;
  pathUsed: string[];
  supportingPaths: unknown[];
  evidenceChain: EdgeEvidence[];
  evidenceChunkIds: string[];
  judgeScores: Record<string, number>;
  decisionScore: number | null;
}

export interface EdgeEvidence {
  source?: string;
  target?: string;
  relation?: string;
  chunk_id?: string;
  quote?: string;
  start?: number;
  end?: number;
  edge_id?: string;
}

export interface GraphModel {
  nodes: GraphNode[];
  edges: GraphEdge[];
  nodeIndex: Map<string, number>;
  positions: Float32Array;
  colors: Float32Array;
  sizes: Float32Array;
  degrees: Float32Array;
  groups: Uint8Array;
  hashes: Uint32Array;
  edgePairs: Uint32Array;
  csrOffsets: Uint32Array;
  csrNeighbors: Uint32Array;
  fingerprint: string;
  metadata: Record<string, unknown>;
}

export interface ChunkEvidence {
  chunk_id: string;
  content: string;
  file_path?: string;
  tokens?: number;
  chunk_order_index?: number;
  missing?: boolean;
}

export interface EntityChunkResponse {
  entity_id?: string;
  edge_id?: string;
  source_ids?: string[];
  chunks?: ChunkEvidence[];
  error?: string;
}

export interface SessionRecord {
  id?: string;
  session_id?: string;
  title?: string;
  name?: string;
  updated_at?: string;
  created_at?: string;
}

export interface LabelPlacement {
  nodeIndex: number;
  text: string;
  anchorX: number;
  anchorY: number;
  offsetX: number;
  offsetY: number;
  width: number;
  height: number;
  color: [number, number, number];
  opacity: number;
  forced: boolean;
}
