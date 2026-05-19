import json
import pandas as pd
from pathlib import Path

from models.local_llm import LocalLLM
from evaluation.advanced_evaluator import evaluate_response
from evaluation.metrics import compute_metrics


ATTACK_DIR = "attacks"
RESULT_FILE = "results/results.csv"

# LOAD ATTACKS
all_attacks = []

for file in Path(ATTACK_DIR).glob("*.json"):

    with open(file, "r") as f:
        data = json.load(f)
        all_attacks.extend(data)

# LOAD MODELS
with open("models_config.json", "r") as f:
    models_config = json.load(f)

all_results = []

for model_info in models_config:

    model_name = model_info["name"]
    hf_name = model_info["hf_name"]

    print(f"\n==========")
    print(f"RUNNING MODEL: {model_name}")
    print(f"==========\n")

    llm = LocalLLM(hf_name)

    model_results = []

    for attack in all_attacks:

        prompt = attack["prompt"]

        print(f"Running Attack {attack['id']}")

        response = llm.query(prompt)

        evaluation = evaluate_response(response)

        result = {
            "model": model_name,
            "attack_id": attack["id"],
            "category": attack["category"],
            "prompt": prompt,
            "response": response,
            **evaluation
        }

        model_results.append(result)
        all_results.append(result)

    df_model = pd.DataFrame(model_results)

    metrics = compute_metrics(df_model)

    print(f"\n=== METRICS: {model_name} ===\n")

    for k, v in metrics.items():
        print(f"{k}: {v}")

df = pd.DataFrame(all_results)

df.to_csv(RESULT_FILE, index=False)

print("\nSaved combined results.")
