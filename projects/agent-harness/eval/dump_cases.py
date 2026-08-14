"""Dump FULL passages + answers for specific cases to establish ground truth."""
import json
import sys

traces = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
targets = [(24856944, "meds"), (25242454, "summarize")]
for r in traces:
    if (r.get("hadm_id"), r.get("prompt")) not in targets:
        continue
    print("=" * 70)
    print(f"CASE {r['hadm_id']}/{r['prompt']}")
    print("QUESTION:", r["question"])
    print("\nANSWER (FULL):\n", r["answer"])
    print("\nTOOL CALLS:")
    for tc in r.get("tool_calls") or []:
        resp = tc.get("response") or {}
        print(f"  {tc['name']} returned={resp.get('returned')} query={resp.get('query')}")
        for p in resp.get("passages") or []:
            print(f"\n  ===== [{p.get('section')}] ({len(p.get('text',''))} chars) =====")
            print("  " + (p.get("text") or "").replace("\n", "\n  "))
