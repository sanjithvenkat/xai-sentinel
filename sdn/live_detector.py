import csv
from pathlib import Path

INPUT = Path("outputs/sdn_features.csv")
OUTPUT = Path("outputs/live_detection.csv")

def detect(row):
    protocol = row.get("protocol", "OTHER").upper()

    try:
        packets = float(row.get("packets") or 0)
        bytes_count = float(row.get("bytes") or 0)
        pps = float(row.get("packets_per_second") or 0)
        bps = float(row.get("bytes_per_second") or 0)
    except ValueError:
        return "REVIEW", "Invalid numeric feature"

    # Ignore empty/stale flow counters rather than flagging them.
    if packets <= 0 and bytes_count <= 0:
        return "NO_ACTIVITY", "No packets or bytes recorded in this flow"

    # Transparent demo heuristics, not a trained AI model.
    if pps >= 100:
        return "ANOMALY_REVIEW", "High packet rate"
    if bps >= 1_000_000:
        return "ANOMALY_REVIEW", "High byte rate"
    if protocol == "OTHER":
        return "REVIEW", "Unclassified protocol; inspect flow rule"
    return "NO_RULE_TRIGGERED", "No configured heuristic triggered"

def main():
    if not INPUT.exists():
        print(f"Missing input: {INPUT}")
        return

    with INPUT.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("No SDN feature records found.")
        return

    results = []
    for row in rows:
        status, reason = detect(row)
        results.append({
            **row,
            "detection_status": status,
            "explanation": reason,
        })

    OUTPUT.parent.mkdir(exist_ok=True)
    with OUTPUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    from collections import Counter
    print(f"Processed {len(results)} flow records.")
    print("Detection summary:", dict(Counter(
        r["detection_status"] for r in results
    )))
    print(f"Saved results to: {OUTPUT.resolve()}")
    print("Reminder: these are rule-based flags, not validated attack predictions.")

if __name__ == "__main__":
    main()
