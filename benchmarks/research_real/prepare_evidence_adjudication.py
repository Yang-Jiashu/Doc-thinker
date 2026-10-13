"""Make evidence-only blind conflict packets; never expose answer/arm/scores.

Safe to run incrementally: only pairs with both completed reviews are read;
existing outputs must be byte-identical. Exact supplied packs for one question
are adjudicated once, with candidate aliases preserved. Original reviews remain
unchanged. This prepares evaluation material, not production retrieval inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def build(packet, reviews):
    expected = {fact["id"] for fact in packet["required_facts"]}
    review_maps = [{row["candidate_id"]: row for row in review["candidates"]} for review in reviews]
    groups = {}
    for candidate in packet["candidates"]:
        sources = candidate["actually_supplied_sources"]
        key = digest(sources)
        group = groups.setdefault(key, {"sources": sources, "candidate_ids": [], "reviews": []})
        identifier = candidate["candidate_id"]
        group["candidate_ids"].append(identifier)
        for review in review_maps:
            row = review[identifier]
            checks = row["retrieved_evidence_checks"]
            statuses = {check["fact_id"]: check["status"] for check in checks}
            if set(statuses) != expected or len(checks) != len(expected):
                raise ValueError("incomplete_fact_review")
            if set(statuses.values()) - {"full", "partial", "absent"}:
                raise ValueError("invalid_fact_status")
            labels = row["irrelevant_source_labels"]
            if len(set(labels)) != len(labels):
                raise ValueError("duplicate_noise_label")
            group["reviews"].append((statuses, sorted(labels)))
    tasks = []
    for key, group in groups.items():
        facts = sorted(fact for fact in expected if len({row[0][fact] for row in group["reviews"]}) > 1)
        noise = len({tuple(row[1]) for row in group["reviews"]}) > 1
        if facts or noise:
            tasks.append({"evidence_id": "evidence-" + key[:16],
                          "candidate_ids": group["candidate_ids"],
                          "actually_supplied_sources": group["sources"],
                          "fact_ids_to_check": facts, "check_irrelevance": noise})
    if not tasks:
        return None
    return {"schema": "blind-evidence-conflict-packet-v1", "packet_id": packet["packet_id"],
            "question": packet["question"], "required_facts": packet["required_facts"],
            "original_pdf_reference_pages": packet["original_pdf_reference_pages"], "evidence_packs": tasks,
            "instructions": "Independently adjudicate only requested fact IDs/noise flags from the actual excerpts. No answer or arm is supplied. Full means every material component, subject, relation, attribution and condition of the required fact is actually in the supplied excerpts, not merely in PDF reference pages. Partial never counts as full. PDF pages verify context, not missing retrieval. For irrelevance flag only wholly irrelevant S labels; preserve mixed/caveat evidence. Exact identical source packs share one evidence_id and one decision. Do not infer arm, seek original reviewers, or read other files.",
            "output_contract": {"packet_id": packet["packet_id"], "reviewer_id": "blind_evidence_arbiter",
                                "evidence_packs": [{"evidence_id": row["evidence_id"],
                                    "fact_checks": [{"fact_id": fact, "status": "full|partial|absent",
                                        "source_labels": [], "reason": "Concrete excerpt-based reason; absent has no labels."}
                                        for fact in row["fact_ids_to_check"]],
                                    "irrelevant_source_labels": [] if row["check_irrelevance"] else None,
                                    "irrelevance_reason": "Concrete reason if requested; otherwise null." if row["check_irrelevance"] else None}
                                    for row in tasks]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    args = parser.parse_args()
    directory = args.analysis / "adjudication/blind"
    directory.mkdir(parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    ready, conflicts, pending = 0, 0, 0
    for path in sorted((args.analysis / "blind").glob("batch_*/*.json")):
        scores = [args.analysis / "scores" / reviewer / path.name for reviewer in ("reviewer_a", "reviewer_b")]
        if not all(score.is_file() for score in scores):
            continue
        packet = json.loads(path.read_bytes())
        reviews = [json.loads(score.read_bytes()) for score in scores]
        if any(review["packet_id"] != packet["packet_id"] for review in reviews):
            raise ValueError("packet_identity_mismatch")
        expected = {row["candidate_id"] for row in packet["candidates"]}
        # An incremental run may observe a reviewer between two saves. Do not
        # adjudicate a partial file or silently interpret its missing arm.
        if any({row["candidate_id"] for row in review["candidates"]} != expected
               or len(review["candidates"]) != len(expected) for review in reviews):
            pending += 1
            continue
        ready += 1
        value = build(packet, reviews)
        if value is None:
            continue
        conflicts += 1
        output, content = directory / path.name, encoded(value)
        if output.exists():
            if output.read_bytes() != content:
                raise ValueError("existing_packet_changed")
        else:
            with os.fdopen(os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
                stream.write(content)
    print(json.dumps({"fully_reviewed_pairs": ready, "conflict_packets": conflicts,
                      "incomplete_review_pairs": pending}))


if __name__ == "__main__":
    main()
