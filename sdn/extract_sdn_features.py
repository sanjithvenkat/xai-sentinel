import csv
import re
from pathlib import Path

INPUT_FILE = Path("outputs/flow_stats.csv")
OUTPUT_FILE = Path("outputs/sdn_features.csv")


def get_value(rule, pattern, default=""):
    match = re.search(pattern, rule)
    return match.group(1) if match else default


def main():
    if not INPUT_FILE.exists():
        print(f"Input file not found: {INPUT_FILE}")
        return

    with INPUT_FILE.open(newline="") as file:
        rows = list(csv.DictReader(file))

    features = []

    for row in rows:
        rule = row["flow_rule"]
        duration = float(row["duration_seconds"])
        packets = int(row["packets"])
        byte_count = int(row["bytes"])

        protocol = (
            "ICMP" if re.search(r"(?:^|,)icmp(?:,| )", rule)
            else "ARP" if re.search(r"(?:^|,)arp(?:,| )", rule)
            else "TCP" if re.search(r"(?:^|,)tcp(?:,| )", rule)
            else "UDP" if re.search(r"(?:^|,)udp(?:,| )", rule)
            else "OTHER"
        )

        source_ip = get_value(rule, r"(?:nw_src|arp_spa)=([^, ]+)")
        destination_ip = get_value(rule, r"(?:nw_dst|arp_tpa)=([^, ]+)")
        icmp_type = get_value(rule, r"icmp_type=(\d+)")
        arp_operation = get_value(rule, r"arp_op=(\d+)")
        source_port = get_value(rule, r"tp_src=(\d+)")
        destination_port = get_value(rule, r"tp_dst=(\d+)")

        features.append({
            "timestamp": row["timestamp"],
            "protocol": protocol,
            "source_ip": source_ip,
            "destination_ip": destination_ip,
            "source_port": source_port,
            "destination_port": destination_port,
	    "icmp_type": icmp_type,
	    "arp_operation": arp_operation,
            "duration_seconds": duration,
            "packets": packets,
            "bytes": byte_count,
            "bytes_per_second": (
                byte_count / duration if duration > 0 else 0
            ),
            "packets_per_second": (
                packets / duration if duration > 0 else 0
            ),
        })

    with OUTPUT_FILE.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(features[0].keys()) if features else [
                "timestamp", "protocol", "source_ip",
                "destination_ip", "icmp_type", "arp_operation", "source_port", "destination_port",
                "duration_seconds", "packets", "bytes",
                "bytes_per_second", "packets_per_second",
            ],
        )
        writer.writeheader()
        writer.writerows(features)

    print(f"Extracted {len(features)} records.")
    print(f"Saved features to: {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
