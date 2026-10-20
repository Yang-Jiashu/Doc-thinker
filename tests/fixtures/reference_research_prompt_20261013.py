# Frozen pre-round-2 formatter for full result differential checks (2026-10-13).
# Not production; uses unchanged external helpers, not an independent semantic
# oracle. No API, persisted private data, or cross-request cache.
"""Fit one ranked evidence plan without re-running retrieval or calling a model.

Evidence and analysis are different inputs. Optional analysis is removed as a
whole before any of its premises can be truncated. UTF-8 bytes are measured, not
misreported as vendor token usage; actual billing depends on the tokenizer.
"""
from __future__ import annotations

import json
import re

from docthinker.server.invite_budget import GROUNDED_OUTPUT_TOKENS, MAX_OUTPUT_TOKENS, validate_invite_messages
from docthinker.server.research_retrieval import grounded_prompt, refresh_query_coverage
from docthinker.server.research_answer_contract import answer_contract_receipt, SYSTEM_POLICY, INSIGHT_POLICY, knowledge_skill_system_policy
from docthinker.server.research_knowledge import compile_task_knowledge
from docthinker.server.knowledge_skills import MAX_PROMPT_BYTES

INSIGHT_DATA_MARKER = "\nINSIGHT_CONTEXT_DATA_JSON:\n"
MAX_INSIGHT_BYTES = 1200


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def empty_assistance(sources, *, reason="not_requested"):
    trace = {"model_calls": 0, "algorithm_modes": [], "assist_bytes": 0,
             "semantic_entailment_verified": False, "skipped_reason": reason}
    return {"sources": sources, "prompt": "", "trace": trace,
            "reasoning": {"mode": "context_assist", "status": "not_requested",
                          "conclusions": [], "claims": [], "gaps": [], "trace": trace}}


def retrieval_question(question, history):
    """Resolve short follow-ups using one prior user turn, never AI prose.

    This is query context, not new evidence. Only enabled conversation history
    is passed by the caller. Concrete new questions retain their own scope.
    """
    followup = re.compile(
        r"^(?:(?:请|帮我)?(?:再|继续)(?:解释|展开|说明|写|讲)|(?:那|那么)?(?:它|这个|这段|这一点|上面|上述|刚才)|"
        r"(?:为什么|为何|怎么说)[？?。.!！]*$|(?:please )?(?:expand|continue|make it |what about that|why[?.!]*$|how so))", re.I)
    if len(question) > 160 or not followup.search(question.strip()):
        return question
    previous = next((row['content'] for row in reversed(history) if row.get('role') == 'user'), '')
    if not previous:
        return question
    return question + '\n前一问题（仅作检索指代）：' + previous[:500]


def requires_document_evidence(question):
    """Do not confuse creating a document with citing an existing one."""
    mentions = re.search(r'论文|原文|文中|文档|资料|文件|\b(?:pdf|papers?|documents?|sources?)\b', question, re.I)
    if not mentions:
        return False
    creating = re.search(r'写|起草|撰写|\b(?:write|draft|compose)\b', question, re.I)
    grounded = re.search(r'根据|基于|依据|按照|这些|上述|上传|这篇|这份|原文|文中|\b(?:according to|based on|from (?:these|the)|uploaded)\b', question, re.I)
    return not creating or bool(grounded)


def fit_research_prompt(question, plan, history, assistance, *, system_policy, memory_as_of=None,
                        knowledge_support=None, insight_runtime=None, trial_contract=None,
                        project_context=None):
    """Keep current question intact; shed whole optional blocks deterministically.

    Rank once upstream. Never repeat BM25, index reads, background operators or
    provider calls for each formatting attempt. Original retrieval evidence wins
    over optional generated hints; hint-only extra sources leave with the hints.
    """
    baseline = plan["sources"]
    if project_context is not None and (not isinstance(project_context, str) or not project_context.strip()
                                        or len(project_context.encode('utf-8')) > 8192):
        raise ValueError('Invalid project context')
    memory = plan["confirmed_memories"]
    attempts = [(assistance["sources"], memory, history, assistance),
                (assistance["sources"], memory, [], assistance),
                (baseline, memory, [], empty_assistance(baseline, reason="prompt_budget")),
                (baseline, [], [], empty_assistance(baseline, reason="prompt_budget"))]
    attempts.extend((baseline[:count], [], [], empty_assistance(baseline[:count], reason="prompt_budget"))
                    for count in range(len(baseline) - 1, -1, -1))
    has_skill = bool((knowledge_support or {}).get('prompt', ''))
    # Whole cards are optional and cannot crowd out original evidence. Try the
    # highest-ranked complete cards with the stage constraint before falling
    # back to the original cards without it. No retrieval is repeated.
    has_insight = bool(insight_runtime is not None and insight_runtime.schema_refs())
    card_count = len((knowledge_support or {}).get('sources', []))
    optional_variants = ([(True, True, None)] + [(True, True, count)
                            for count in range(min(card_count, 3) - 1, 0, -1)] +
                            [(True, False, None), (False, False, None)]
                         if has_skill and has_insight else
                         [(True, False, None), (False, False, None)] if has_skill else
                         [(False, True, None), (False, False, None)] if has_insight else
                         [(False, False, None)])
    attempts = [(sources, memory, history, support, skill, insight, card_limit, component_locators)
                for sources, memory, history, support in attempts
                for skill, insight, card_limit in optional_variants
                # Definition locators are optional attention aids. Drop them
                # intact before dropping evidence, history or another card.
                for component_locators in (True, False)]
    compiled_variants = {}
    insight_variants = {}
    coverage_variants = {}
    for sources, kept_memory, conversation, support, include_skill, include_insight, card_limit, component_locators in attempts:
        source_key = tuple(id(row) for row in sources)
        task_knowledge = knowledge_support
        if include_skill:
            # History/memory budget variants often share exactly the same
            # excerpts. Check each excerpt variant once, without any retrieval.
            method_key = (source_key, card_limit)
            if method_key not in compiled_variants:
                selected = knowledge_support
                if card_limit is not None:
                    selected = {**knowledge_support, 'sources': knowledge_support['sources'][:card_limit],
                                'trace': {**knowledge_support['trace'],
                                    'budget_omitted_count': knowledge_support['trace']['budget_omitted_count']
                                        + card_count - card_limit}}
                compiled_variants[method_key] = compile_task_knowledge(selected, sources)
            task_knowledge = compiled_variants[method_key]
        skill = (task_knowledge or {}).get('prompt', '') if include_skill else ''
        active_method_labels = tuple(row['label'] for row in (task_knowledge or {}).get('sources', []) if skill)
        insight_key = (source_key, active_method_labels)
        if has_insight and insight_key not in insight_variants:
            proposal = insight_runtime.on_stage('after_evidence', {
                'sources': sources, 'task_revision': 1,
                'active_method_labels': list(active_method_labels),
            })
            rendered = insight_runtime.render_context(proposal)
            # A single JSON object is data, never an instruction appended by a
            # resource renderer. Invalid or oversized optional hints disappear
            # intact without changing the underlying evidence.
            try:
                packet = json.loads(rendered) if isinstance(rendered, str) and rendered else None
                rendered = json.dumps(packet, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
                if not isinstance(packet, dict) or len(rendered.encode('utf-8')) > MAX_INSIGHT_BYTES:
                    rendered = ''
            except (ValueError, TypeError):
                rendered = ''
            if not active_method_labels:
                rendered = ''
            insight_variants[insight_key] = (proposal, rendered)
        insight_proposal, available_insight = insight_variants.get(insight_key, (None, ''))
        insight = available_insight if include_insight else ''
        if len(skill.encode('utf-8')) + len(insight.encode('utf-8')) > MAX_PROMPT_BYTES:
            continue
        if source_key not in coverage_variants:
            coverage_variants[source_key] = refresh_query_coverage(plan, sources)
        coverage_plan = coverage_variants[source_key]
        gaps = list(coverage_plan["gaps"])
        if len(conversation) < len(history) or len(kept_memory) < len(memory) or len(sources) < len(baseline):
            gaps.append({"code": "context_budget_omitted", "message": "为遵守输入预算，部分历史、确认记忆或资料未加入本次回答。"})
        fitted = {**coverage_plan, "sources": sources, "selected_chunks": sources,
                  "confirmed_memories": kept_memory, "gaps": gaps}
        try:
            prompt = grounded_prompt(question, fitted, include_component_locators=component_locators)
            if project_context is not None:
                # Kept intact in every budget variant. No dependency may be
                # reported as handed off if formatting silently discarded it.
                prompt += '\nPROJECT_TASK_DATA_JSON:\n' + _json({'context': project_context})
            if conversation:
                prompt += "\n近期对话（可能截短；不是原文证据，只作指代上下文）：\n" + _json(conversation)
            if memory_as_of is not None:
                prompt += "\n记忆按事件有效时间回看（UTC 秒）=" + str(memory_as_of) + "；这是目前所知的历史背景，不代表那时已经获知，也不改变原文的适用日期。"
            prompt += support["prompt"]
            if skill:
                prompt += '\nKNOWLEDGE_SKILL_DATA_JSON:\n' + skill
            if insight:
                prompt += INSIGHT_DATA_MARKER + insight
            if trial_contract:
                # Requested delivery contracts are mandatory host data, not
                # optional method hints. Fit the whole contract on every
                # source variant; never silently drop it and claim enabled.
                prompt += trial_contract['prompt']
            if not sources:
                prompt += ("\n当前没有原文匹配：可作普通交流或参考用户确认的背景，但不能声称研究结论来自论文。"
                           "不要声称自动保存了记忆；长期保存只能由用户在确认记忆页明确提交。")
            used_policy = (SYSTEM_POLICY if not skill and system_policy == knowledge_skill_system_policy(True)
                           else system_policy)
            if insight:
                used_policy += INSIGHT_POLICY
            if trial_contract:
                used_policy += trial_contract['policy']
            messages = [{"role": "system", "content": used_policy}, {"role": "user", "content": prompt}]
            validate_invite_messages(messages)
        except ValueError:
            continue
        budget = {"measurement": "utf8_bytes_not_billed_tokens", "max_input_bytes": 24576,
                  "input_bytes": sum(len(item["role"].encode()) + len(item["content"].encode()) for item in messages),
                  "source_bytes": sum(len(row["content"].encode()) for row in sources),
                  "memory_bytes": sum(len(row["content"].encode()) for row in kept_memory),
                  "history_bytes": len(_json(conversation).encode()) if conversation else 0,
                  "assist_bytes": len(support["prompt"].encode()), "retrieval_passes": 1,
                  "knowledge_skill_bytes": len(skill.encode()),
                  "insight_bytes": len(insight.encode()),
                  "max_output_tokens": GROUNDED_OUTPUT_TOKENS if sources else MAX_OUTPUT_TOKENS,
                  "output_budget_profile": "grounded" if sources else "chat",
                  "automatic_retry": False}
        if trial_contract:
            budget['trial_contract_bytes'] = len(trial_contract['prompt'].encode()) + len(trial_contract['policy'].encode())
        if project_context is not None:
            budget['project_context_bytes'] = len(_json({'context': project_context}).encode('utf-8'))
        receipt = answer_contract_receipt(messages, sources, original_source_count=len(assistance['sources']),
                                         question=question, include_component_locators=component_locators)
        if trial_contract:
            receipt['trial_contract'] = {'version': 1, 'schema_sha256': trial_contract['schema_sha256'],
                                         'included': True, 'scope': 'output_structure_not_semantic_support'}
        reasoning = {**support["reasoning"], "trace": {**support["reasoning"]["trace"],
                     "context_budget": budget, "answer_contract": receipt}}
        if isinstance(plan.get('query_terms'), (list, tuple)):
            had_query_gap = any(row.get('code') == 'uncovered_query_terms'
                                for row in reasoning.get('gaps', []))
            reasoning['gaps'] = [row for row in reasoning.get('gaps', [])
                                 if row.get('code') != 'uncovered_query_terms']
            if had_query_gap:
                # Refresh an existing receipt without adding a new persistent
                # copy of raw question tokens or growing the reasoning budget.
                reasoning['gaps'].extend({'code': row['code'], 'message': row['message']}
                    for row in fitted['gaps'] if row.get('code') == 'uncovered_query_terms')
        return {"plan": fitted, "prompt": prompt, "messages": messages,
                "knowledge_skill_included": bool(skill),
                "knowledge_support": task_knowledge,
                "insight_proposal": insight_proposal, "insight_included": bool(insight),
                "trial_contract_included": bool(trial_contract),
                "reasoning": reasoning, "budget": budget}
    raise ValueError("Research prompt exceeds its budget")
