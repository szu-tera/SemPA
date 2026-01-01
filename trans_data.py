import json
import pandas as pd

input_csv = "./data/nli_for_simcse.csv"
output_jsonl = "./data/paraphrase_data.jsonl"

df = pd.read_csv(input_csv)

with open(output_jsonl, "w", encoding="utf-8") as out:
    for _, row in df.iterrows():

        sent0 = str(row.get("sent0", "")).strip()
        sent1 = str(row.get("sent1", "")).strip()
        hard_neg = str(row.get("hard_neg", "")).strip()

        # Skip invalid data (empty sentences)
        if not (sent0 and sent1):
            continue

        # design prompt（Paraphrase Generation Task)
        # P1 prompt = f'Keep the same meaning of this sentence: "{sent0}", while making some changes.'
        # P2 prompt = f'Generate a paraphrase of this sentence: "{sent0}" that preserves its meaning, while making some changes.'
        # P3 prompt = f'Keep the main meaning of this sentence: "{sent0}", and rewrite it in a different way.'
        # P4 prompt = f'Generate a paraphrase of the sentence: "{sent0}".'
        # P5 prompt = f'Rewrite the sentence: "{sent0}" while preserving its main meaning, but the wording may be simplified or rephrased.'

        prompt = f'Keep the same meaning of this sentence: "{sent0}", while making some changes.'

        data = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": sent1}  # chosen 
            ],
            "rejected_response": hard_neg  # rejected 
        }

        out.write(json.dumps(data, ensure_ascii=False) + "\n")

print("Done:", output_jsonl)
