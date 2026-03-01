import torch
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import IntegratedGradients

def get_model_device(model):
    """
    Robustly get device from model (works for nn.Module and ScriptModule).
    """
    try:
        return next(model.parameters()).device
    except Exception:
        # ScriptModule might not expose .parameters() easily or be empty
        # Fallback: check a dummy attribute or return CPU if fails
        # But ScriptModule usually has .code, .graph.
        # Safest for ScriptModule is to check one of its submodules if possible
        # Or just assume it's on the same device as inputs we want to send?
        # A trick: check a parameter if possible, else default to cpu
        # If model is ScriptModule, we can try accessing a known buffer?
        pass
    
    # Check if it has any parameters (ScriptModule often does via named_parameters)
    try:
        return next(model.game_local_xai()).device # unlikely
    except:
        pass
        
    try: 
        # For TorchScript, list(model.parameters()) usually works
        params = list(model.parameters())
        if len(params) > 0:
            return params[0].device
    except:
        pass
        
    return torch.device("cpu")

class ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ModelWrapper, self).__init__()
        self.model = model

    def forward(self, input):
        output = self.model(input)
        if isinstance(output, (tuple, list)):
            output = output[0]
        
        # Ensure 2D output for Captum: (Batch, NumOutputs)
        # Check if shape is (N,) or (N)
        if len(output.shape) == 1:
             output = output.unsqueeze(1)
            
        return output

def compute_ig_importance_global(model, X_test, feature_cols=None, target=0, n_steps=50, top_k=15, device=None):
    """
    Compute Integrated Gradients (IG) for either:
    - Temporal data: X_test shape (N, T, F)
    - Tabular data: X_test shape (N, F)

    Automatically detects data type based on tensor dimension.

    Args:
        model: PyTorch model
        X_test: numpy array (N, T, F) or (N, F)
        feature_cols: optional feature names
        target: class index for attribution
        n_steps: IG steps
        top_k: top-K features to visualize

    Returns:
        all_attr: IG attributions, same shape as X_test
        global_feat_imp_signed: (F,)
        global_year_imp_abs (only for temporal): (T,) or None
    """

    model.eval()
    wrapped_model = ModelWrapper(model)
    ig = IntegratedGradients(wrapped_model)

    X_np = np.array(X_test)
    dim = X_np.ndim

    if dim == 3:
        # Temporal dataset: (N, T, F)
        N, T, F = X_np.shape
        is_temporal = True
    elif dim == 2:
        # Tabular dataset: (N, F)
        N, F = X_np.shape
        T = None
        is_temporal = False
    else:
        raise ValueError("X_test must be 2D (tabular) or 3D (temporal)")

    all_attr_list = []

    # 1) Compute IG per sample
    if device is None:
        device = get_model_device(model)
    
    for i in range(N):
        x_i = torch.tensor(X_np[i], dtype=torch.float32).unsqueeze(0).to(device)
        baseline_i = torch.zeros_like(x_i).to(device)

        attr_i, _ = ig.attribute(
            x_i,
            baselines=baseline_i,
            target=target,
            n_steps=n_steps,
            return_convergence_delta=True
        )
        all_attr_list.append(attr_i.squeeze(0).detach().cpu())

    all_attr = torch.stack(all_attr_list, dim=0)  # (N, T, F) or (N, F)

    # 2) Feature-level Importance
    if is_temporal:
        global_feat_imp_signed = all_attr.mean(dim=(0, 1)).numpy()  # (F,)
    else:
        global_feat_imp_signed = all_attr.mean(dim=0).numpy()  # (F,)

    # feature labels
    if feature_cols is not None and len(feature_cols) == F:
        global_feat_labels = feature_cols
    else:
        global_feat_labels = [f"f={i}" for i in range(F)]

    # Top-K selection
    sorted_idx = np.argsort(np.abs(global_feat_imp_signed))[::-1]
    top_idx = sorted_idx[:min(top_k, F)]
    top_vals = global_feat_imp_signed[top_idx]
    top_labels = np.array(global_feat_labels)[top_idx]

    # 3) Visualization
    if is_temporal:
        # Two plots: feature level & time level
        global_year_imp_abs = all_attr.abs().mean(dim=(0, 2)).numpy()  # (T,)
        x_years = np.arange(T)

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Left: Feature importance
        colors_feat = np.where(top_vals > 0, '#ff0d57', '#1e88e5')
        axes[0].barh(np.arange(len(top_idx)), top_vals, color=colors_feat, edgecolor='gray')
        axes[0].set_yticks(np.arange(len(top_idx)))
        axes[0].set_yticklabels(top_labels)
        axes[0].invert_yaxis()
        axes[0].set_xlabel("mean IG (signed)")
        axes[0].set_title(f"Top {len(top_idx)} Features")

        # Right: Year importance
        axes[1].bar(x_years, global_year_imp_abs, color='#1e88e5', edgecolor='gray')
        axes[1].set_xticks(x_years)
        axes[1].set_xticklabels([f"t={i}" for i in x_years])
        axes[1].set_xlabel("Time step / Year")
        axes[1].set_ylabel("mean |IG|")
        axes[1].set_title("Year-level Importance (mean |IG|)")

        plt.tight_layout()
        plt.tight_layout()
        # plt.show()

    else:
        # Only one plot
        plt.figure(figsize=(7, 6))
        colors_feat = np.where(top_vals > 0, '#ff0d57', '#1e88e5')
        plt.barh(np.arange(len(top_idx)), top_vals, color=colors_feat, edgecolor='gray')
        plt.yticks(np.arange(len(top_idx)), top_labels)
        plt.gca().invert_yaxis()
        plt.xlabel("mean IG (signed)")
        plt.title(f"Top {len(top_idx)} Features")
        plt.tight_layout()
        global_year_imp_abs = None
        fig = plt.gcf() 

    # 4) Return results
    return all_attr, global_feat_imp_signed, global_year_imp_abs, fig

def compute_ig_importance_local(
    model, 
    X_sample, 
    feature_cols, 

    target=0, 
    n_steps=50, 
    top_k=10,
    padding_mask=None,
    device=None
):
    """
    LEFT  = Waterfall plot (your code)
    RIGHT = Year-level mean |IG| plot (only for temporal input)
    padding_mask: boolean array of shape (T,), True where step is PADDED (to be ignored)
    """

    model.eval()
    wrapped_model = ModelWrapper(model)
    ig = IntegratedGradients(wrapped_model)

    X_np = np.array(X_sample)
    F = X_np.shape[-1]

    # Detect temporal or tabular
    is_temporal = (X_np.ndim == 2)

    # 1. Prepare tensors
    if device is None:
        device = get_model_device(model)
        
    input_tensor = torch.tensor(X_np, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Adaptive Baseline:
    # 0.0 for real data (standard assumption for scaled data)
    # padding_value for padded steps (to ensure input - baseline = 0)
    # We infer padded locations from the input using the mask if provided, or by value
    baseline = torch.zeros_like(input_tensor).to(device)
    
    if padding_mask is not None:
         # padding_mask is (T,) boolean. 
         # We need to broadcast to (1, T, F)
         # Find padding value from input (assuming mask is correct)
         # Or just assume standard padding behavior.
         # Ideally we pass padding_value to this function, but for now let's use the input at those positions
         # input_tensor values at padded positions SHOULD be the padding value.
         
         # Convert numpy mask to torch
         mask_tensor = torch.tensor(padding_mask, device=device) # (T,)
         
         # Expand mask to (1, T, F)
         mask_expanded = mask_tensor.unsqueeze(0).unsqueeze(-1).expand_as(input_tensor)
         
         # Copy padding values from input to baseline
         baseline[mask_expanded] = input_tensor[mask_expanded]

    # 2. Model prediction and baseline output
    with torch.no_grad():
        pred_full = wrapped_model(input_tensor).squeeze()
        final_output = pred_full[target].item() if pred_full.ndim > 0 else pred_full.item()

        base_full = wrapped_model(baseline).squeeze()
        baseline_output = base_full[target].item() if base_full.ndim > 0 else base_full.item()

    # 3. IG attribution
    attributions, delta = ig.attribute(
        input_tensor,
        baselines=baseline,
        target=target,
        n_steps=n_steps,
        return_convergence_delta=True
    )
    attr = attributions.squeeze(0).detach().cpu().numpy()

    # 4. Feature-level importance
    if is_temporal:
         # With adaptive baseline, attr at padded steps is mathematically 0.
         # No need to manually mask, which preserves the integral property (sum + baseline = pred)
         feature_importance_signed = attr.sum(axis=0)
    else:
        feature_importance_signed = attr

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # LEFT： Waterfall
    ax = axes[0]

    contrib_full = feature_importance_signed.copy()
    names_full = np.array(feature_cols)

    idx = np.argsort(np.abs(contrib_full))[::-1]
    top_idx = idx[:min(top_k, F)]
    other_idx = idx[min(top_k, F):]

    contrib = contrib_full[top_idx]
    names = names_full[top_idx]

    contrib_other = contrib_full[other_idx].sum()
    if contrib_other != 0:
        contrib = np.append(contrib, contrib_other)
        names = np.append(names, "Other features")

    cumulative = baseline_output + np.cumsum(contrib)

    y_pos = np.arange(len(names))
    colors = np.where(contrib > 0, "#ff0051", "#008bfb")

    for i in range(len(names)):
        x_start = cumulative[i] - contrib[i]
        x_end = cumulative[i]

        ax.barh(
            i,
            contrib[i],
            left=x_start,
            color=colors[i],
            edgecolor="black",
            height=0.8
        )

        ax.text(
            x_end + 0.01 * np.sign(contrib[i]),
            i,
            f"{contrib[i]:+.3f}",
            va="center",
            fontsize=9
        )

    ax.axvline(baseline_output, color="gray", linestyle="--", label=f"Baseline={baseline_output:.3f}")
    ax.axvline(final_output, color="green", linestyle="-", label=f"Pred={final_output:.3f}")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_title("Local IG: Waterfall (Top Features + Other)")
    ax.set_xlabel("Contribution")
    ax.invert_yaxis()
    ax.legend(loc="lower right")

    xmin, xmax = ax.get_xlim()
    padding = (xmax - xmin) * 0.15   # extend 15% on both sides
    ax.set_xlim(xmin - padding, xmax + padding)
    
    # RIGHT： Year-level |IG| (only for temporal)
    ax2 = axes[1]

    if is_temporal:
        year_imp = np.mean(np.abs(attr), axis=1)  # (T,)
        x_years = np.arange(len(year_imp))

        ax2.bar(x_years, year_imp, color="#1e88e5", edgecolor="black")
        ax2.set_xticks(x_years)
        ax2.set_xticklabels([f"t={i}" for i in x_years])

        ax2.set_title("Year-level Importance (mean |IG|)")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("mean |IG|")
    else:
        ax2.text(0.5, 0.5, "No Year Dimension (Tabular Input)", ha="center", va="center")
        ax2.axis("off")

    plt.tight_layout()
    plt.tight_layout()
    # plt.show()

    return attr, feature_importance_signed, fig