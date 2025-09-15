# Example command:
# find /home/andrewor/local/logs/axolotl/saved-9-24 -maxdepth 1 -type d | xargs python parse_it.py

import os
import re
import sys

if len(sys.argv) == 1:
    print("Usage: python parse_it.py [log_dir1] [log_dir2] ...")
    sys.exit(1)
log_dirs = sys.argv[1:]

def extract_wikitext_perplexity(log_file: str) -> float:
    with open(log_file, "r") as f:
        for l in f.readlines():
            m = re.match(".*\|word_perplexity\|↓  \|([\\d.]*)\|.*", l)
            if m is not None:
                return float(m.groups()[0])
    raise ValueError("Did not find wikitext perplexity in %s" % log_file)

for log_dir in log_dirs:
    if "qat" not in log_dir:
        continue
    qat_dir = log_dir
    baseline_dir = qat_dir.replace("_qat", "_baseline")
    assert "qat" in qat_dir
    assert "baseline" in baseline_dir
    qat_name = qat_dir.split("/")[-1]
    baseline_name = baseline_dir.split("/")[-1]

    # extract eval data
    all_data = {}
    try:
        for experiment_dir in [qat_dir, baseline_dir]:
            experiment_name = experiment_dir.split("/")[-1]
            all_data[experiment_name] = {}
            float_eval_file = os.path.join(experiment_dir, "eval_float.log")
            quantized_eval_file = os.path.join(experiment_dir, "eval_quantized.log")
            all_data[experiment_name]["wikitext_word_perplexity_float"] = extract_wikitext_perplexity(float_eval_file)
            all_data[experiment_name]["wikitext_word_perplexity_quantized"] = extract_wikitext_perplexity(quantized_eval_file)
    except FileNotFoundError:
        print(f"Skipping log directory {log_dir}")
        continue

    # print data in a nice format
    metric = "wikitext_word_perplexity"
    baseline_float_value = all_data[baseline_name][metric + "_float"]
    baseline_quantized_value = all_data[baseline_name][metric + "_quantized"]
    qat_quantized_value = all_data[qat_name][metric + "_quantized"]
    recovered = (qat_quantized_value - baseline_quantized_value) / (baseline_float_value - baseline_quantized_value) * 100
    print(
        "%s: %.3f (baseline) -> %.3f (quant) -> %.3f (qat), recovered %.3f%%" % (
            log_dir.split("/")[-1].replace("Llama3.2-3B_", ""),
            baseline_float_value,
            baseline_quantized_value,
            qat_quantized_value,
            recovered,
        )
    )
