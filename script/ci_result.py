import os
import re
import csv
import argparse
from datetime import datetime

#input_dir = "~/results"
#output_file = "benchmark_results.csv"
parser = argparse.ArgumentParser(description="Extract benchmark results to CSV.")
parser.add_argument("input_dir", type=str, help="Directory containing result files.")
#parser.add_argument("output_file", type=str, nargs="?", default="benchmark_results.csv", help="Output CSV file name.")
args = parser.parse_args()

current_date = datetime.now().strftime("%Y-%m-%d")
output_dir = os.path.join(os.path.expanduser(args.input_dir), "sorted_results")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, f"{current_date}.csv")


#csv_columns = ["Benchmark", "Waterline", "Library", "Compiler", "Latency", "RMS", "MemUsage"]
csv_columns = [
    "Benchmark", "Waterline", "Library", "Compiler", "Device",
    "NumTest", "LoopCount", "Latency", "RMS"
]

# Benchmark-Waterline-Library-Compiler-Device-NumTest-LoopCount.txt
file_pattern = re.compile(
    r"^(.*?)-(\d+)-(.*?)-(.*?)-(.*?)-(\d+)-(\d+)\.txt$"
)

results = []
issues_missing = []  # latency/rms missing
issues_rms = []      # rms > 1
issues_parse = []    # rms/latency parse fail
#file_pattern = re.compile(r"^(.*?)-(\d+)-(.*?)-(.*?).txt$")
#data_pattern = re.compile(
##    r"latency:\s*(\d+\.\d+).*?rms:\s*([\d\.]+)(?:.*?MemUsage:\s*(\d+\.\d+)GB)?",
#    r"latency:\s*(\d+\.\d+).*?rms:\s*([\deE\.\+-]+)",    
#    re.DOTALL
#)
data_pattern = re.compile(
    r"latency\s*\(s\):\s*([0-9eE\.\+-]+).*?"
    r"rms:\s*([0-9eE\.\+-]+)",
    re.DOTALL
)

for filename in sorted(os.listdir(os.path.expanduser(args.input_dir))):
    m = file_pattern.match(filename)
    if not m:
        continue

    benchmark, waterline, library, compiler, device, num_test, loop_count = m.groups()

    file_path = os.path.join(os.path.expanduser(args.input_dir), filename)
    with open(file_path, "r") as f:
        content = f.read()

    latency = rms = ""
    dm = data_pattern.search(content)
    if dm:
        latency, rms = dm.groups()

    # ---- issue detection ----
    # 1) missing
    if (latency == "" or rms == ""):
        issues_missing.append((benchmark, filename, latency, rms))

    # 2) rms too large
    if rms != "":
        try:
            rms_val = float(rms)
            if rms_val > 1.0:
                issues_rms.append((benchmark, filename, rms))
        except ValueError:
            issues_parse.append((benchmark, filename, "rms", rms))

    # latency parse sanity
    if latency != "":
        try:
            lat_val = float(latency)
            if lat_val < 0:
                issues_parse.append((benchmark, filename, "latency", latency))
        except ValueError:
            issues_parse.append((benchmark, filename, "latency", latency))

    results.append([
        benchmark, waterline, library, compiler, device,
        num_test, loop_count, latency, rms
    ])

results.sort(key=lambda x: (x[2], x[4], x[0], int(x[1]), int(x[5]), int(x[6])))

#results.sort(key=lambda x: x[2])

with open(output_file, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(csv_columns)
    writer.writerows(results)

with open(output_file, "a", newline="") as f:
    f.write("\n")
    f.write(f"# Summary ({current_date})\n")
    f.write(f"# Total rows: {len(results)}\n")

    f.write(f"# Missing Latency/RMS: {len(issues_missing)}\n")
    for b, fn, lat, rms_ in issues_missing:
        f.write(f"#   - {b} (file={fn}) latency='{lat}' rms='{rms_}'\n")

    f.write(f"# RMS > 1.0: {len(issues_rms)}\n")
    for b, fn, rms_ in issues_rms:
        f.write(f"#   - {b} (file={fn}) rms={rms_}\n")

    f.write(f"# Parse/Sanity issues: {len(issues_parse)}\n")
    for b, fn, field, val in issues_parse:
        f.write(f"#   - {b} (file={fn}) {field}='{val}'\n")


print(f"Results have been saved to {output_file}")

import shutil

latest_path = os.path.join(output_dir, "latest.csv")
shutil.copyfile(output_file, latest_path)
print(f"Latest CSV updated to {latest_path}")
