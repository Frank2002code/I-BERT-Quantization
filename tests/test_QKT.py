import os

import numpy as np

# Try to import torch for GeLU reference, otherwise define it manually
try:
    import torch
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def gelu_numpy(x):
    # Approximation of GeLU if torch is not available
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * np.power(x, 3))))


BASE_DIR = (
    r"d:\DL\Quantization\I-BERT-Quantization\result\encoder\sentence_encoder\layers\0"
)


def load_dequantized(dir_path, input_or_output="output"):
    """
    Loads int and scaling_factor and returns dequantized float array.
    Returns (float_array, int_array, scaling_factor)
    """
    int_path = os.path.join(dir_path, f"{input_or_output}_int.npy")
    scale_path = os.path.join(dir_path, f"{input_or_output}_scaling_factor.npy")

    if not os.path.exists(int_path):
        print(f"[Warn] File not found: {int_path}")
        return None, None, None

    int_val = np.load(int_path).astype(np.float64)

    if os.path.exists(scale_path):
        scale_val = np.load(scale_path)
        # Scale might be a scalar array or scalar
        if scale_val.size == 1:
            scale_val = float(scale_val)
        return int_val * scale_val, int_val, scale_val
    else:
        # If scaling factor is missing, return int values and indicate missing scale
        return int_val, int_val, None


def check_correlation(a, b, name):
    a_flat = a.flatten()
    b_flat = b.flatten()

    # Handle NaNs or Infs
    valid_mask = np.isfinite(a_flat) & np.isfinite(b_flat)
    if not np.all(valid_mask):
        print(f"[{name}] Warning: Contains NaNs or Infs. Filtering...")
        a_flat = a_flat[valid_mask]
        b_flat = b_flat[valid_mask]

    if len(a_flat) == 0:
        print(f"[{name}] Error: No valid data points.")
        return

    corr = np.corrcoef(a_flat, b_flat)[0, 1]
    print(f"[{name}] Pearson Correlation: {corr:.4f}")

    if len(a_flat) > 5:
        print(f"[{name}] Sample values (Ref vs Actual):")
        indices = np.random.choice(len(a_flat), 5, replace=False)
        for idx in indices:
            print(f"  {a_flat[idx]:.4f} vs {b_flat[idx]:.4f}")


def verify_gelu():
    print("\n--- Verifying GeLU ---")
    path = os.path.join(BASE_DIR, "activation_fn_approx")

    # Load input
    input_float, input_int, input_scale = load_dequantized(path, "input")
    # Load actual output
    output_float, output_int, output_scale = load_dequantized(path, "output")

    if input_float is None or output_float is None:
        print("Skipping GeLU verification due to missing files.")
        return

    # Compute reference output
    if HAS_TORCH:
        ref_output = F.gelu(torch.tensor(input_float)).numpy()
    else:
        ref_output = gelu_numpy(input_float)

    check_correlation(ref_output, output_float, "GeLU")


def verify_qkt():
    print("\n--- Verifying QK^T ---")

    # Paths
    q_path = os.path.join(BASE_DIR, "self_attn", "q_proj_act")
    k_path = os.path.join(BASE_DIR, "self_attn", "k_proj_act")
    target_path = os.path.join(BASE_DIR, "self_attn", "softmax")

    # Load Q and K outputs
    q_float, _, _ = load_dequantized(q_path, "output")
    k_float, _, _ = load_dequantized(k_path, "output")

    # Load Target (Softmax Input)
    # Note: Softmax input usually doesn't have scaling factor saved in this dump format seemingly,
    # so we load the int values and check correlation with the computed float product.
    target_float, target_int, target_scale = load_dequantized(target_path, "input")

    if q_float is None or k_float is None or target_int is None:
        print("Skipping QK^T verification due to missing files.")
        return

    # Shapes:
    # Q, K usually: (Batch, SeqLen, HiddenDim)
    # Target: (Batch, Heads, SeqLen, SeqLen)

    # Determine shapes
    # Fairseq usually uses (SeqLen, Batch, Hidden) for Q, K, V
    # Target (Softmax input) usually (Batch*Heads, SeqLen, SeqLen)

    # Trust the target shape for SeqLen first because it's (.., S, S)
    if target_int.ndim == 3:
        # Expected: (Batch*Heads, SeqLen, SeqLen)
        B_combined, S_t1, S_t2 = target_int.shape
        assert S_t1 == S_t2, (
            f"Target last two dims should be equal (SeqLen, SeqLen), got {S_t1}, {S_t2}"
        )
        real_S = S_t1
    elif target_int.ndim == 4:
        # Expected: (Batch, Heads, SeqLen, SeqLen)
        B_t, NumHeads, S_t1, S_t2 = target_int.shape
        assert S_t1 == S_t2
        real_S = S_t1
        B_combined = B_t * NumHeads
    else:
        print(f"Unexpected target shape: {target_int.shape}")
        return

    # Now align Q shape
    dim0, dim1, dim2 = q_float.shape
    if dim0 == real_S:
        # Shape is (SeqLen, Batch, Hidden)
        S, B, H_dim = dim0, dim1, dim2
        print(f"Inferred Q shape: (SeqLen, Batch, Hidden) -> ({S}, {B}, {H_dim})")
        # Transpose to (B, S, H) for easier processing
        q_float = q_float.transpose(1, 0, 2)  # (B, S, H)
        k_float = k_float.transpose(1, 0, 2)  # (B, S, H)
    elif dim1 == real_S:
        # Shape is (Batch, SeqLen, Hidden)
        B, S, H_dim = dim0, dim1, dim2
        print(f"Inferred Q shape: (Batch, SeqLen, Hidden) -> ({B}, {S}, {H_dim})")
    else:
        print(f"Could not match Q shape {q_float.shape} with Target SeqLen {real_S}")
        return

    NumHeads = B_combined // B

    HeadDim = H_dim // NumHeads
    print(
        f"Shapes inferred: Batch={B}, SeqLen={S}, NumHeads={NumHeads}, HeadDim={HeadDim}"
    )

    # Reshape Q and K
    # (B, S, H_dim) -> (B, S, NumHeads, HeadDim) -> (B, NumHeads, S, HeadDim)
    q_reshaped = q_float.reshape(B, S, NumHeads, HeadDim).transpose(0, 2, 1, 3)
    k_reshaped = k_float.reshape(B, S, NumHeads, HeadDim).transpose(0, 2, 1, 3)

    # Compute Q @ K.T
    # (B, NumHeads, S, HeadDim) @ (B, NumHeads, HeadDim, S) -> (B, NumHeads, S, S)
    attn_scores = np.matmul(q_reshaped, k_reshaped.transpose(0, 1, 3, 2))

    # Scaling by 1/sqrt(HeadDim) is standard in Attention
    attn_scores = attn_scores / np.sqrt(HeadDim)

    # Align shapes if mismatch (e.g. target is 3D (B*H, S, S) but attn is 4D (B, H, S, S))
    if target_int.ndim == 3 and attn_scores.ndim == 4:
        attn_scores = attn_scores.reshape(-1, real_S, real_S)

    # Compare with target
    # If target has no scale, we compare attn_scores (float) vs target_int (int).
    # They should be highly correlated (linear relationship).
    check_correlation(
        attn_scores, target_int if target_scale is None else target_float, "QK^T"
    )

    if target_scale is None:
        # Calculate approximate scale from the data
        # Real = Int * Scale => Scale = Real / Int
        # Use valid entries
        valid = (target_int != 0) & np.isfinite(attn_scores)
        if np.any(valid):
            # Take median ratio to avoid outliers
            ratios = attn_scores[valid] / target_int[valid]
            inferred_scale = np.median(ratios)
            print(
                f"[QK^T] Note: Target scale missing. Inferred scale from data: {inferred_scale:.6e}"
            )
            print("       This explains the magnitude difference (Float vs Raw Int).")


def verify_v_softmax():
    print("\n--- Verifying V * Softmax ---")

    softmax_path = os.path.join(BASE_DIR, "self_attn", "softmax")
    v_path = os.path.join(BASE_DIR, "self_attn", "v_proj_act")
    attn_act_path = os.path.join(BASE_DIR, "self_attn", "attn_act")

    # Load Softmax Output (Probs)
    probs_float, _, _ = load_dequantized(softmax_path, "output")
    # Load V Output
    v_float, _, _ = load_dequantized(v_path, "output")
    # Load Attn Act Output (Context)
    context_float, _, _ = load_dequantized(attn_act_path, "output")

    if probs_float is None or v_float is None or context_float is None:
        print("Skipping V * Softmax verification due to missing files.")
        if probs_float is None:
            print(f"Missing {softmax_path}/output_int.npy")
        if v_float is None:
            print(f"Missing {v_path}/output_int.npy")
        if context_float is None:
            print(f"Missing {attn_act_path}/output_int.npy")
        return

    # Dimensions
    # Probs: (B, NumHeads, S, S)
    # V: (B, S, HiddenDim) -> Needs reshape -> (B, NumHeads, S, HeadDim)
    # Context expected: (B, S, HiddenDim) -> or maybe (B, NumHeads, S, HeadDim) before merge?
    # Usually attn_act output is the result BEFORE the final Linear projection (out_proj),
    # so it matches the V shape (B, S, H) or (B, H, S, D).

    # Let's infer shapes

    # Analyze Probs Shape (similar to QK^T verification)
    real_S = 0
    if probs_float.ndim == 3:
        # (Batch*Heads, SeqLen, SeqLen)
        B_combined, S1, S2 = probs_float.shape
        assert S1 == S2, "Probs last two dims must correspond to SeqLen"
        real_S = S1
        # Reshape to (B_combined, 1, S, S) for easier matmul later, or keep as is?
        # Let's keep as (B*NumHeads, S, S)
    elif probs_float.ndim == 4:
        B_t, NumHeads, S1, S2 = probs_float.shape
        B_combined = B_t * NumHeads
        real_S = S1
        # Flatten to (B*NumHeads, S, S)
        probs_float = probs_float.reshape(B_combined, real_S, real_S)
    else:
        print(f"Unexpected probs shape: {probs_float.shape}")
        return

    # Analyze V Shape
    dim0, dim1, dim2 = v_float.shape
    if dim0 == real_S:
        # (S, B, H)
        S, B, H_dim = dim0, dim1, dim2
        # Transpose to (B, S, H)
        v_float = v_float.transpose(1, 0, 2)
    elif dim1 == real_S:
        # (B, S, H)
        B, S, H_dim = dim0, dim1, dim2
    else:
        print(f"Could not match V shape {v_float.shape} with Probs SeqLen {real_S}")
        return

    NumHeads = B_combined // B
    if NumHeads == 0:
        NumHeads = 1  # Safety
    HeadDim = H_dim // NumHeads

    # Reshape V
    # (B, S, H) -> (B, S, NumHeads, HeadDim) -> (B, NumHeads, S, HeadDim)
    v_reshaped = v_float.reshape(B, S, NumHeads, HeadDim).transpose(0, 2, 1, 3)

    # Compute Probs @ V
    # (B, NumHeads, S, S) @ (B, NumHeads, S, HeadDim) -> (B, NumHeads, S, HeadDim)
    context_calc = np.matmul(probs_float, v_reshaped)

    # The stored comparison context might be flattened back to (B, S, H) or kept as (B, H, S, D).
    # Let's check context_float shape
    if context_float.shape == (S, B, H_dim):
        print(
            f"Target shape detected as (SeqLen, Batch, Hidden): {context_float.shape}"
        )
        # Calc is (B, NumHeads, S, HeadDim)
        # We need (S, B, NumHeads * HeadDim)

        # 1. Transpose to (S, B, NumHeads, HeadDim)
        context_calc_transposed = context_calc.transpose(2, 0, 1, 3)

        # 2. Reshape to (S, B, HiddenDim)
        context_calc_final = context_calc_transposed.reshape(S, B, H_dim)

        check_correlation(
            context_calc_final, context_float, "V * Softmax (Corrected Layout)"
        )
    elif context_float.shape == (B, S, H_dim):
        check_correlation(context_calc, context_float, "V * Softmax (Heads)")
    else:
        print(
            f"Shape mismatch: Calc {context_calc.shape} vs Loaded {context_float.shape}"
        )
        # Try comparing flattened
        check_correlation(context_calc, context_float, "V * Softmax (Flattened)")


if __name__ == "__main__":
    if not os.path.exists(BASE_DIR):
        print(f"Directory not found: {BASE_DIR}")
    else:
        verify_gelu()
        verify_qkt()
        verify_v_softmax()
