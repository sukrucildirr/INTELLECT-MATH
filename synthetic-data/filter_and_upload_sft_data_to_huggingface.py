import json
import argparse
from datasets import Dataset
from transformers import AutoTokenizer

def convert_to_chat_format(data, index, tokenizer, max_length):
    print(index)
    encoded = tokenizer(data["response"], return_tensors="pt")
    if encoded["input_ids"].shape[1] > max_length:
        return None  # Return None if the length exceeds the maximum

    return {
        "id": f"math_problem_{index}",
        "messages": [
            {"role": "system", "content": "Solve the following math problem efficiently and clearly. Think carefully and step by step about your response and reason before providing a final response. Conclude your response with: \n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem."},
            {"role": "user", "content": data["prompt"]},
            {"role": "assistant", "content": data["response"]}
        ],
        "ground_truth": data["ground_truth"]
    }

def main(args):
    tokenizer = AutoTokenizer.from_pretrained("Qwen/QwQ-32B-Preview")
    
    with open(args.input_file, "r") as f:
        data = [json.loads(line) for line in f]

    chat_data = [
        result for result in (
            convert_to_chat_format(item, i, tokenizer, args.max_tokens)
            for i, item in enumerate(data)
            if item.get("correct", False)
        )
        if result is not None
    ]

    dataset = Dataset.from_list(chat_data)

    print(f"Total problems: {len(data)}")
    print(f"Correct problems included: {len(chat_data)}")

    dataset.push_to_hub(args.upload_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and upload math problems dataset")
    parser.add_argument("--max_tokens", type=int, default=512, help="Maximum number of tokens for each response")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSONL file")
    parser.add_argument("--upload_path", type=str, required=True, help="Hugging Face upload path (e.g., 'username/dataset_name')")
    args = parser.parse_args()

    main(args)
