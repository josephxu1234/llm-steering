"""
activation_steering_demo.py

Demonstrates activation addition (ActAdd) on GPT-2-XL

Usage:
    python steering_demo.py
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Configs
MODEL_PATH = "/scratch/gpfs/TSILVER/jx6/models/gpt2-xl"

# Layer to inject the steering vector into.
# GPT-2-XL has 48 layers (indexed 0–47).
INJECTION_LAYER = 24

MAX_NEW_TOKENS = 60

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded on {device}.\n")


def get_mean_activation(prompt: str) -> torch.Tensor:
    """
    Run a forward pass and return the mean residual stream activation
    at INJECTION_LAYER, averaged over the token dimension.

    Returns a tensor of shape [hidden_dim] (1600 for GPT-2-XL).
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    captured = {}

    def hook_fn(module, input, output):
        # output[0] shape: [batch=1, seq_len, hidden_dim]
        captured["hidden"] = output[0].detach()

    hook = model.transformer.h[INJECTION_LAYER].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(**inputs)
    hook.remove()

    # Mean-pool over the sequence length → [hidden_dim]
    # return captured["hidden"].squeeze(0).mean(dim=0)
    return captured["hidden"][-1, :]  # last token only, shape [hidden_dim]


def layer_sweep(prompt, steering_vec, coeff, layers):
    """Try injecting at each layer and print the result."""
    for layer in layers:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        def make_hook(l):
            def hook_fn(module, input, output):
                output[0][:] += coeff * steering_vec.to(device)
                return output

            return hook_fn

        hook = model.transformer.h[layer].register_forward_hook(make_hook(layer))
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        hook.remove()

        new_tokens = out[0][inputs["input_ids"].shape[1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        print(f"  Layer {layer:2d}: {text}")


def compute_steering_vector(
    positive_prompts: list[str], negative_prompts: list[str]
) -> torch.Tensor:
    """Average over multiple contrastive pairs for a more robust steering vector."""
    assert len(positive_prompts) == len(negative_prompts)
    diffs = []
    for pos, neg in zip(positive_prompts, negative_prompts):
        p = get_mean_activation(pos)
        n = get_mean_activation(neg)
        diffs.append(p - n)
    vec = torch.stack(diffs).mean(dim=0)
    print(f'vector norm: {vec.norm()}')
    return vec


def generate(
    prompt: str,
    steering_vector: torch.Tensor | None = None,
    coeff: float = 0.0,
) -> str:
    """
    Generate text from a prompt, optionally injecting a scaled steering
    vector into the residual stream at every forward pass through INJECTION_LAYER.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    if steering_vector is not None and coeff != 0.0:
        # Expand to [1, 1, hidden_dim] so it broadcasts over batch and seq dims
        # vec = steering_vector.to(device).unsqueeze(0).unsqueeze(0)

        def hook_fn(module, input, output):
            output[0][:] += coeff * steering_vector.to(device)
            return output

        hook = model.transformer.h[INJECTION_LAYER].register_forward_hook(hook_fn)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,  # greedy decoding
            pad_token_id=tokenizer.eos_token_id,
        )

    if steering_vector is not None and coeff != 0.0:
        hook.remove()

    # Decode only the newly generated tokens (not the prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def run_demo(
    demo_name: str,
    positive_prompt: str,
    negative_prompt: str,
    test_prompts: list[str],
    coefficients: list[float],
) -> None:
    """
    Run a single steering demo:
      1. Compute a steering vector from the contrastive pair.
      2. For each test prompt, generate with each coefficient.
      3. Print a formatted comparison table.
    """
    print("=" * 70)
    print(f"DEMO: {demo_name}")
    print(f'  Positive prompt : "{positive_prompt}"')
    print(f'  Negative prompt : "{negative_prompt}"')
    print(f"  Injection layer : {INJECTION_LAYER}")
    print("=" * 70)

    steering_vec = compute_steering_vector(positive_prompt, negative_prompt)
    print(f"Steering vector computed. Norm: {steering_vec.norm().item():.2f}\n")

    for prompt in test_prompts:
        print(f'Prompt: "{prompt}"\n')
        for coeff in coefficients:
            label = f"coeff={coeff:+.0f}"
            if coeff == 0.0:
                label += " (baseline)"
            completion = generate(prompt, steering_vec, coeff)
            print(f"  [{label}]")
            print(f"  {completion}")
            print()
        print("-" * 70 + "\n")


# ── Demo definitions ───────────────────────────────────────────────────────────

if __name__ == "__main__":

    happy_positives = [
        "Everything is amazing, wonderful, and happy.",
        "I'm so extremely happy today."
    ]

    happy_negatives = [
        "Everything is awful, depressing, and miserable.",
        "I'm so extremely sad today."
    ]

    banana_positives = [
        "I talk about bananas constantly",
        "the most important thing in the world is bananas"
    ]

    banana_negatives = [
        "I do not talk about bananas constantly",
        "bananas are not relevant"
    ]

    formal_positives = [
        "I always speak formally in a manner that is prim and proper."
    ]

    formal_negatives = [
        "I constantly use slang and never speak formally."
    ]

    # ── Demo 1: Sentiment steering ────────────────────────────────────────────
    run_demo(
        demo_name="Sentiment Steering",
        positive_prompt=happy_positives,
        negative_prompt=happy_negatives,
        test_prompts=[
            "Today I went to the store and",
            "The weather outside today",
        ],
        coefficients=[-1, 0.0, 1],
    )

    # ── Demo 2: Topic injection ────────────────────────────────────────────────
    # Injects a concept (bananas) regardless of what the prompt is about.
    run_demo(
        demo_name="Topic Injection (bananas)",
        positive_prompt=banana_positives,
        negative_prompt=banana_negatives,
        test_prompts=[
            "The stock market today",
            "Scientists have discovered a new",
        ],
        coefficients=[0.0, 1],
    )