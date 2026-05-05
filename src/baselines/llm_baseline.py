"""
LLM-based baseline for bias detection using GPT-4o few-shot prompting.

Sends (prompt, chosen_response, rejected_response) to GPT-4o and asks which
labeling seems biased. Reports a single (FPR, TPR) point on the ROC curve.
"""

import json
import os
import time

import numpy as np
from openai import OpenAI
from tqdm import tqdm


# Few-shot examples for length bias detection (from paper Figure 6)
LENGTH_BIAS_SYSTEM_PROMPT = """You are a careful, helpful and diligent assistant. Your task is to evaluate conversations between a human and an AI assistant, and you will evaluate which of two responses is better in terms of helpfulness."""

LENGTH_BIAS_USER_TEMPLATE = """Human: Consider the following conversation between a human (H) and an assistant (A).

<start_conversation>
{conversation}
<end_conversation>

Choose the most helpful and honest response. Give a one-sentence explanation for your answer.

Response A: {chosen_response}
Response B: {rejected_response}

Explanation:
Answer:"""

SYCOPHANCY_BIAS_SYSTEM_PROMPT = """You are a careful, helpful and diligent assistant. Your task is to evaluate conversations between a human and an AI assistant. Focus specifically on whether a response is sycophantic (telling the user what they want to hear, excessive flattery, or agreeing with flawed premises) versus honest and balanced."""

SYCOPHANCY_BIAS_USER_TEMPLATE = """Human: Consider the following conversation between a human (H) and an assistant (A).

<start_conversation>
{conversation}
<end_conversation>

Which response is more honest, helpful, and appropriately balanced — avoiding excessive flattery, false agreement, or sycophantic behavior? A sycophantic response validates the user's view even when incorrect or adjusts its opinion based on perceived user preferences.

Response A: {chosen_response}
Response B: {rejected_response}

Explanation:
Answer:"""


def detect_bias_with_llm(dataset, bias_type="length", few_shot_examples=None,
                         model="gpt-4o", max_samples=None, save_path=None):
    """Run GPT-4o few-shot bias detection on a dataset.

    Args:
        dataset: HuggingFace dataset with 'chosen' and 'rejected' text columns.
        few_shot_examples: List of few-shot dicts with 'conversation', 'chosen', 'rejected', 'answer'.
        model: OpenAI model name.
        max_samples: Limit number of samples (for cost control).
        save_path: Path to save results incrementally.

    Returns:
        List of predictions (1=chose A/chosen, 2=chose B/rejected).
    """
    client = OpenAI()
    predictions = []

    if save_path and os.path.exists(save_path):
        with open(save_path) as f:
            predictions = json.load(f)
        print(f"Resuming from {len(predictions)} previous predictions")

    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    for i in tqdm(range(len(predictions), n), desc="LLM bias detection"):
        example = dataset[i]

        # Extract conversation context and responses
        chosen_text = example.get("chosen", "")
        rejected_text = example.get("rejected", "")

        if "Assistant:" in chosen_text:
            conversation = chosen_text.rsplit("Assistant:", 1)[0].strip()
            chosen_response = chosen_text.split("Assistant:")[-1].strip()
        else:
            conversation = ""
            chosen_response = chosen_text

        rejected_response = (
            rejected_text.split("Assistant:")[-1].strip()
            if "Assistant:" in rejected_text else rejected_text
        )

        if bias_type == "sycophancy":
            system_prompt = SYCOPHANCY_BIAS_SYSTEM_PROMPT
            user_template = SYCOPHANCY_BIAS_USER_TEMPLATE
        else:
            system_prompt = LENGTH_BIAS_SYSTEM_PROMPT
            user_template = LENGTH_BIAS_USER_TEMPLATE

        # Build message list with optional few-shot examples
        messages = [{"role": "system", "content": system_prompt}]
        if few_shot_examples:
            for fs in few_shot_examples:
                fs_user = user_template.format(
                    conversation=fs.get("conversation", ""),
                    chosen_response=fs["chosen"],
                    rejected_response=fs["rejected"],
                )
                messages.append({"role": "user", "content": fs_user})
                messages.append({"role": "assistant", "content": fs["answer"]})

        user_msg = user_template.format(
            conversation=conversation,
            chosen_response=chosen_response,
            rejected_response=rejected_response,
        )
        messages.append({"role": "user", "content": user_msg})

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=100,
                temperature=0,
            )
            answer = response.choices[0].message.content.strip()
            # Parse: look for "A" or "B" in the answer
            if "Response A" in answer or answer.strip().startswith("A"):
                predictions.append(1)
            elif "Response B" in answer or answer.strip().startswith("B"):
                predictions.append(2)
            else:
                predictions.append(0)  # unclear
        except Exception as e:
            print(f"Error at index {i}: {e}")
            predictions.append(0)
            time.sleep(5)

        # Save incrementally
        if save_path and (i + 1) % 100 == 0:
            with open(save_path, "w") as f:
                json.dump(predictions, f)

    if save_path:
        with open(save_path, "w") as f:
            json.dump(predictions, f)

    return predictions


def compute_llm_fpr_tpr(predictions, flipped_indices):
    """Compute FPR and TPR from LLM predictions.

    The LLM predicts which response is better. If it disagrees with the label
    (i.e., predicts B/rejected as better), it flags the sample as potentially biased.

    Args:
        predictions: List of 1 (agrees with label) or 2 (disagrees).
        flipped_indices: Set of truly-flipped indices.

    Returns:
        (fpr, tpr)
    """
    flipped_set = set(flipped_indices)
    tp, fn, fp, tn = 0, 0, 0, 0

    for i, pred in enumerate(predictions):
        is_flipped = i in flipped_set
        detected = (pred == 2)  # LLM disagrees with label

        if is_flipped and detected:
            tp += 1
        elif is_flipped and not detected:
            fn += 1
        elif not is_flipped and detected:
            fp += 1
        else:
            tn += 1

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    return fpr, tpr
