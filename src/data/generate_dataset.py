from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

TAXONOMY_PATH = DATA_DIR / "taxonomy.json"
INTENT_PATH = DATA_DIR / "intent_template.json"
OUTPUT_PATH = DATA_DIR / "prompts.csv"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def generate_rows(taxonomy, intent_templates):
    rows = []

    for capability, policy_groups in taxonomy.items():
        for policy, topics in policy_groups.items():
            for topic in topics:
                for intent, templates in intent_templates.items():
                    for template_idx, template in enumerate(templates, start=1):
                        prompt = template.format(topic=topic)

                        rows.append(
                            {
                                "id": f"{capability}_{intent}_{policy}_{template_idx}_{len(rows)+1}",
                                "prompt": prompt,
                                "intent": intent,
                                "capability": capability,
                                "policy": policy,
                                "topic": topic,
                                "template_id": template_idx,
                            }
                        )

    return rows



def main():
    taxonomy = load_json(TAXONOMY_PATH)
    intent_templates = load_json(INTENT_PATH)

    rows = generate_rows(taxonomy, intent_templates)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Generated {len(df)} prompts")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()