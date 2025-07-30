from datasets import load_dataset
import pandas as pd
import json

# dataset = load_dataset("parquet", data_files="campo-math-62k.parquet", split="train")


# df = dataset.to_pandas()

# df['reward_model'] = df['clean_answer'].apply(lambda x: json.dumps({"ground_truth": x, "style": "rule-lighteval/MATH_v2"}))
# df.to_parquet("campo-math-62k-updated.parquet", index=False)

# print("Saved updated dataset with reward_model column to MiroMind-M1-RL-62K-updated.parquet")

import pandas as pd
from datasets import load_dataset
import json

dataset = load_dataset("parquet", data_files="campo-math-62k.parquet", split="train")
df = dataset.to_pandas()
df['reward_model'] = df['clean_answer'].apply(lambda x: {"ground_truth": x, "style": "rule-lighteval/MATH_v2"})

# Add prompt column
def create_prompt(problem_text):
    prompt_template = [
        {
            "content": (
                "Solve the following math problem step by step. The last line of your response should be "
                "of the form Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
                f"{problem_text}\n\n"
                "Remember to put your answer on its own line after \"Answer:\"."
            ),
            "role": "user"
        }
    ]
    return json.dumps(prompt_template)

df['prompt'] = df['problem'].apply(create_prompt)
df['data_source'] = "math_dapo"
df['ability'] = "MATH"

df.to_parquet("campo-math-62k-fixed.parquet", index=False)
print("Added prompt and reward column and saved updated dataset")




### show data

import pandas as pd
from datasets import load_dataset

# Load dataset from your local parquet file
dataset = load_dataset("parquet", data_files="campo-math-62k-updated.parquet", split="train")

# Convert to pandas DataFrame
df = dataset.to_pandas()

# Example query - you can change this as needed
filtered_df = df[df['pass_ratio'] > 0.5]

# Configure pandas to show full content of all columns without truncation
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)

# Show the top row after filtering
top_row = filtered_df.head(1)
print(top_row)
