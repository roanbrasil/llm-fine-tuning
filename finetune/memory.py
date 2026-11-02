"""Back-of-the-envelope VRAM for fine-tuning, matching the numbers in the article.

These ignore activations, which depend on batch size, sequence length and gradient checkpointing.
"""

GB = 1e9

# Mixed-precision Adam, per trainable parameter:
# bf16 weight (2) + bf16 grad (2) + fp32 master weight (4) + Adam m (4) + Adam v (4) = 16 bytes
FULL_FT_BYTES_PER_PARAM = 16
BF16_BYTES = 2
NF4_BYTES = 0.5  # 4 bits; double quantization keeps the scale constants to ~0.127 bits/param
NF4_SCALE_BITS = 0.127


def full_finetune_gb(n_params: float) -> float:
    return n_params * FULL_FT_BYTES_PER_PARAM / GB


def lora_gb(n_params: float, n_adapter_params: float) -> float:
    """Frozen bf16 base + adapters trained with the full 16-byte Adam budget."""
    return (n_params * BF16_BYTES + n_adapter_params * FULL_FT_BYTES_PER_PARAM) / GB


def qlora_gb(n_params: float, n_adapter_params: float) -> float:
    """Frozen NF4 base (with double-quantized scales) + bf16 adapters trained with Adam."""
    base = n_params * (NF4_BYTES + NF4_SCALE_BITS / 8)
    return (base + n_adapter_params * FULL_FT_BYTES_PER_PARAM) / GB


def llama_like_adapter_params(n_layers: int, d_model: int, d_ff: int, r: int, kv_dim: int | None = None) -> int:
    """LoRA on every linear layer of attention (q, k, v, o) and MLP (gate, up, down)."""
    kv = kv_dim or d_model
    shapes = [(d_model, d_model), (d_model, kv), (d_model, kv), (d_model, d_model),
              (d_model, d_ff), (d_model, d_ff), (d_ff, d_model)]
    per_layer = sum(r * (i + o) for i, o in shapes)
    return n_layers * per_layer
