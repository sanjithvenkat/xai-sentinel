import csv
import re
import subprocess
from datetime import datetime
from pathlib import Path

OUTPUT = Path("outputs")
OUTPUT.mkdir(exist_ok=True)

def collect_flows():
    result = subprocess.run(
        ["sudo", "ovs-ofctl", "dump-flows", "s1"],
        capture_output=True,
        text=True,
        check=True,
    )

    rows = []

    for line in result.stdout.splitlines():
        packets = re.search(r"n_packets=(\d+)", line)
        byte_count = re.search(r"n_bytes=(\d+)", line)
        priority = re.search(r"priority=(\d+)", line)
        duration = re.search(r"duration=([\d.]+)s", line)

        if not packets or not byte_count:
            continue

        rows.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "duration_seconds": float(duration.group(1)) if duration else 0,
            "priority": int(priority.group(1)) if priority else 0,
            "packets": int(packets.group(1)),
            "bytes": int(byte_count.group(1)),
            "flow_rule": line.strip(),
        })

    output_file = OUTPUT / "flow_stats.csv"
    write_header = not output_file.exists() or output_file.stat().st_size == 0

    with output_file.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "timestamp", "duration_seconds", "priority",
            "packets", "bytes", "flow_rule"
        ])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    print(f"Collected {len(rows)} flow entries.")
    print(f"Saved to: {output_file.resolve()}")

if __name__ == "__main__":
    collect_flows()
