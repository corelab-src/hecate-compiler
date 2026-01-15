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
csv_columns = ["Benchmark", "Waterline", "Library", "Compiler", "Latency", "RMS"]

results = []
file_pattern = re.compile(r"^(.*?)-(\d+)-(.*?)-(.*?).txt$")
data_pattern = re.compile(
#    r"latency:\s*(\d+\.\d+).*?rms:\s*([\d\.]+)(?:.*?MemUsage:\s*(\d+\.\d+)GB)?",
    r"latency:\s*(\d+\.\d+).*?rms:\s*([\deE\.\+-]+)",    
    re.DOTALL
)

for filename in sorted(os.listdir(os.path.expanduser(args.input_dir))):
    match = file_pattern.match(filename)
    if match:
        benchmark, waterline, library, compiler = match.groups()
        file_path = os.path.join(os.path.expanduser(args.input_dir), filename)
        with open(file_path, "r") as f:
            content = f.read()
            data_match = data_pattern.search(content)
            if data_match:
                latency, rms = data_match.groups()
            else:
                latency, rms = "", ""
            results.append([
                benchmark, waterline, library, compiler, latency, rms
            ])

results.sort(key=lambda x: x[2])

with open(output_file, "w", newline="") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(csv_columns)
    writer.writerows(results)

print(f"Results have been saved to {output_file}.")


