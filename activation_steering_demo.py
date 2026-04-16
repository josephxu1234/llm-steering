"""
activation_steering_demo.py

Demonstrates activation addition (ActAdd) on GPT-2-XL using raw PyTorch hooks.
Runs fully offline — load the model from a local path before submitting to Slurm.

Usage:
    python steering_demo.py

To pre-download the model on the login node (requires internet):
    python -c "
    from transformers import AutoTokenizer, AutoModelForCausalLM
    path = '/scratch/gpfs/<YourNetID>/models/gpt2-xl'
    AutoTokenizer.from_pretrained('gpt2-xl').save_pretrained(path)
    AutoModelForCausalLM.from_pretrained('gpt2-xl').save_pretrained(path)
    "
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Configuration ─────────────────────────────────────────────────────────────

MODEL_PATH = "/scratch/gpfs/<YourNetID>/models/gpt2-xl"

# Layer to inject the steering vector into.
# GPT-2-XL has 48 layers (indexed 0–47).
# Mid-to-late layers (15–25) work best for semantic/tonal steering.
INJECTION_LAYER = 17

MAX_NEW_TOKENS = 60


# ── Model loading ──────────────────────────────────────────────────────────────

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float32)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model loaded on {device}.\n")


# ── Core steering utilities ───────────────────────────────────────────────────

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
    return captured["hidden"].squeeze(0).mean(dim=0)


def compute_steering_vector(
    positive_prompt: str,
    negative_prompt: str,
) -> torch.Tensor:
    """
    Compute a steering vector as the difference in mean activations
    between a positive and negative contrastive prompt pair.

    The resulting vector points from the negative concept toward the
    positive concept in the residual stream space.
    """
    pos = get_mean_activation(positive_prompt)
    neg = get_mean_activation(negative_prompt)
    return pos - neg


def generate(
    prompt: str,
    steering_vector: torch.Tensor | None = None,
    coeff: float = 0.0,
) -> str:
    """
    Generate text from a prompt, optionally injecting a scaled steering
    vector into the residual stream at every forward pass through INJECTION_LAYER.

    Args:
        prompt:          The input text.
        steering_vector: Precomputed vector of shape [hidden_dim].
        coeff:           Scale factor. Positive steers toward the concept,
                         negative steers away. Zero = unsteered baseline.
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    if steering_vector is not None and coeff != 0.0:
        # Expand to [1, 1, hidden_dim] so it broadcasts over batch and seq dims
        vec = steering_vector.to(device).unsqueeze(0).unsqueeze(0)

        def hook_fn(module, input, output):
            output[0][:] += coeff * vec
            return output

        hook = model.transformer.h[INJECTION_LAYER].register_forward_hook(hook_fn)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,       # greedy decoding — deterministic and easier to compare
            pad_token_id=tokenizer.eos_token_id,
        )

    if steering_vector is not None and coeff != 0.0:
        hook.remove()

    # Decode only the newly generated tokens (not the prompt)
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
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
    print(f"  Positive prompt : \"{positive_prompt}\"")
    print(f"  Negative prompt : \"{negative_prompt}\"")
    print(f"  Injection layer : {INJECTION_LAYER}")
    print("=" * 70)

    steering_vec = compute_steering_vector(positive_prompt, negative_prompt)
    print(f"Steering vector computed. Norm: {steering_vec.norm().item():.2f}\n")

    for prompt in test_prompts:
        print(f"Prompt: \"{prompt}\"\n")
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

    # ── Demo 1: Sentiment steering ────────────────────────────────────────────
    # The classic and most reliable demo. The steering vector encodes a direction
    # in activation space that roughly corresponds to positive vs. negative affect.
    run_demo(
        demo_name="Sentiment Steering (positive ↔ negative)",
        positive_prompt="I feel wonderful, joyful, and full of hope.",
        negative_prompt="I feel terrible, miserable, and full of despair.",
        test_prompts=[
            "Today I went to the store and",
            "The weather outside is",
        ],
        coefficients=[-20.0, 0.0, 20.0],
    )

    # ── Demo 2: Topic injection ────────────────────────────────────────────────
    # Injects a concept (bananas) regardless of what the prompt is about.
    # This is the "wedding" example from the original ActAdd paper, adapted.
    # It's visually striking because the effect is clear and a little absurd.
    run_demo(
        demo_name="Topic Injection (bananas)",
        positive_prompt="bananas, yellow fruit, tropical, bunch of bananas",
        negative_prompt="",  # empty string as the neutral/negative baseline
        test_prompts=[
            "The stock market today",
            "Scientists have discovered a new",
        ],
        coefficients=[0.0, 15.0, 30.0],
    )

    # ── Demo 3: Formality steering ────────────────────────────────────────────
    # Steers between casual/colloquial and formal/academic registers.
    run_demo(
        demo_name="Formality Steering (casual ↔ formal)",
        positive_prompt="The study hereby presents empirical evidence demonstrating significant findings.",
        negative_prompt="Yo so basically we found out some pretty wild stuff lol.",
        test_prompts=[
            "Our results show that",
        ],
        coefficients=[-15.0, 0.0, 15.0],
    )

    print("Done.")