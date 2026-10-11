# run_stability_prune.py
import argparse, json, math, copy, yaml
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy import stats

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# -------------------
# Models
# -------------------
ACTS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation="relu"):
        super().__init__()
        act = ACTS[activation]
        sizes = [input_size] + list(hidden_sizes) + [output_size]
        self.linears = nn.ModuleList([nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)])
        self.acts = nn.ModuleList([act() for _ in range(len(sizes)-2)])
        self.hidden_sizes = hidden_sizes
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

    def forward(self, x):
        for i, lin in enumerate(self.linears):
            x = lin(x)
            if i < len(self.linears)-1:
                x = self.acts[i](x)
        return x

    def get_architecture(self):
        return {
            "input_size": self.input_size,
            "hidden_sizes": self.hidden_sizes,
            "output_size": self.output_size,
            "activation": self.activation
        }


# -------------------
# Utils
# -------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def synthetic_classification(n, d_in, d_out):
    X = torch.randn(n, d_in)
    # Create a simple pattern to learn
    X[:, 0] = X[:, 0] + X[:, 1]  # Create correlation
    y = torch.randint(0, d_out, (n,))
    return X, y

def train_one(model, X, y, iters, lr, verbose=False):
    crit = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=lr)
    losses = []
    
    for i in range(iters):
        opt.zero_grad()
        logits = model(X)
        loss = crit(logits, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        
        if verbose and i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss.item():.4f}")
    
    return model, losses

def extract_params(model: MLP) -> List[Tuple[np.ndarray, np.ndarray]]:
    out = []
    for lin in model.linears:
        out.append((lin.weight.detach().cpu().numpy().copy(),
                    lin.bias.detach().cpu().numpy().copy()))
    return out

def accuracy(model, X, y):
    with torch.no_grad():
        pred = model(X).argmax(dim=1)
        return (pred == y).float().mean().item()

def ce_loss(model, X, y):
    with torch.no_grad():
        return nn.CrossEntropyLoss()(model(X), y).item()


# -------------------
# Visualization Tools
# -------------------
class WeightVisualizer:
    def __init__(self, output_dir="visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def plot_weight_distributions(self, init_params, trained_params, arch_name, layer_idx):
        """Plot weight distributions before and after training"""
        init_weights = init_params[layer_idx][0].flatten()
        trained_weights = trained_params[layer_idx][0].flatten()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot histograms
        ax1.hist(init_weights, bins=50, alpha=0.7, label='Initial', density=True)
        ax1.hist(trained_weights, bins=50, alpha=0.7, label='Trained', density=True)
        ax1.set_xlabel('Weight Value')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Weight Distribution - Layer {layer_idx}')
        ax1.legend()
        
        # Plot box plots
        data = [init_weights, trained_weights]
        ax2.boxplot(data)
        ax2.set_xticklabels(['Initial', 'Trained'])  # Fix for MatplotlibDeprecationWarning
        ax2.set_ylabel('Weight Value')
        ax2.set_title(f'Weight Spread - Layer {layer_idx}')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'weight_dist_{arch_name}_layer{layer_idx}.png')
        plt.close()
        
    def plot_stability_scores(self, scores_hidden, masks_hidden, arch_name, threshold):
        """Plot stability scores for each layer with pruning threshold"""
        n_layers = len(scores_hidden)
        fig, axes = plt.subplots(1, n_layers, figsize=(5*n_layers, 5))
        if n_layers == 1:
            axes = [axes]
            
        for i, (scores, mask) in enumerate(zip(scores_hidden, masks_hidden)):
            axes[i].bar(range(len(scores)), scores, alpha=0.7)
            axes[i].axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold}')
            axes[i].set_xlabel('Neuron Index')
            axes[i].set_ylabel('Stability Score')
            axes[i].set_title(f'Layer {i} - Kept: {mask.sum()}/{len(mask)}')
            axes[i].legend()
            
        plt.tight_layout()
        plt.savefig(self.output_dir / f'stability_scores_{arch_name}.png')
        plt.close()
        
    def plot_weight_changes(self, init_params, trained_params, arch_name):
        """Plot absolute weight changes by layer"""
        n_layers = len(init_params)
        changes = []
        layer_labels = []
        
        for i in range(n_layers):
            init_w = init_params[i][0]
            trained_w = trained_params[i][0]
            change = np.abs(trained_w - init_w).mean()
            changes.append(change)
            layer_labels.append(f'Layer {i}')
            
        plt.figure(figsize=(10, 6))
        plt.bar(layer_labels, changes, alpha=0.7)
        plt.xlabel('Layer')
        plt.ylabel('Mean Absolute Weight Change')
        plt.title(f'Weight Changes During Training - {arch_name}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / f'weight_changes_{arch_name}.png')
        plt.close()
        
    def plot_neuron_correlations(self, params_list, arch_name):
        """Plot correlation matrices between neurons across different runs"""
        n_layers = len(params_list[0])
        n_runs = len(params_list)
        
        for layer_idx in range(n_layers):
            # Extract weights for this layer from all runs
            layer_weights = [params[layer_idx][0] for params in params_list]
            
            # Calculate mean weights for each neuron across runs
            mean_neurons = []
            for run_weights in layer_weights:
                mean_neurons.append(run_weights.mean(axis=1))  # Mean across input dimensions
                
            # Create correlation matrix
            corr_matrix = np.corrcoef(mean_neurons)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                       xticklabels=[f'Run {i}' for i in range(n_runs)],
                       yticklabels=[f'Run {i}' for i in range(n_runs)])
            plt.title(f'Neuron Correlation Across Runs - Layer {layer_idx}\n{arch_name}')
            plt.tight_layout()
            plt.savefig(self.output_dir / f'neuron_corr_{arch_name}_layer{layer_idx}.png')
            plt.close()
            
    def plot_pruning_impact(self, original_acc, pruned_acc, original_size, pruned_size, arch_names):
        """Plot comparison of accuracy vs model size before and after pruning"""
        plt.figure(figsize=(10, 6))
        
        # Calculate compression ratio
        compression = [o/p for o, p in zip(original_size, pruned_size)]
        
        for i, name in enumerate(arch_names):
            plt.scatter(pruned_size[i], pruned_acc[i], label=f'{name} (pruned)', s=100, alpha=0.7)
            plt.scatter(original_size[i], original_acc[i], label=f'{name} (original)', s=100, alpha=0.7)
            plt.plot([original_size[i], pruned_size[i]], 
                    [original_acc[i], pruned_acc[i]], 'k--', alpha=0.5)
            
            # Annotate with compression ratio
            plt.annotate(f'{compression[i]:.1f}x', 
                        (pruned_size[i], pruned_acc[i]),
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Model Size (number of parameters)')
        plt.ylabel('Accuracy')
        plt.title('Pruning Impact: Accuracy vs Model Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pruning_impact_comparison.png')
        plt.close()


# -------------------
# Stability measurement (three modes)
# -------------------
def neuron_stability_mask(init_W, init_b, tr_W, tr_b, mode: str, thr: float):
    """
    Returns boolean mask over output neurons (rows of W).
    mode: "relative" | "absolute" | "l2"
    """
    dW = tr_W - init_W
    db = tr_b - init_b

    if mode == "relative":
        w_rel = np.abs(dW) / (np.abs(init_W) + 1e-8)
        b_rel = np.abs(db) / (np.abs(init_b) + 1e-8)
        return (np.all(w_rel < thr, axis=1) & (b_rel < thr))

    if mode == "absolute":
        w_abs = np.abs(dW)
        b_abs = np.abs(db)
        return (np.all(w_abs < thr, axis=1) & (b_abs < thr))

    if mode == "l2":
        # L2 over incoming weights per neuron + bias magnitude
        w_l2 = np.linalg.norm(dW, axis=1)          # shape (out,)
        # scale by fan-in to be size-agnostic
        w_l2 /= (np.linalg.norm(init_W, axis=1) + 1e-8)
        b_l2 = np.abs(db) / (np.abs(init_b) + 1e-8)
        return ((w_l2 < thr) & (b_l2 < thr))

    raise ValueError(f"Unknown diff_mode: {mode}")


# -------------------
# Consensus pruned builder
# -------------------
def build_pruned_consensus(arch, masks_hidden, stable_flags_runs, init_params_runs):
    H = len(arch["hidden_sizes"])
    pruned_hidden = [int(m.sum()) for m in masks_hidden]
    pruned = MLP(arch["input_size"], pruned_hidden, arch["output_size"], activation=arch["activation"])

    num_runs = len(init_params_runs)

    def live_in_idxs(layer):
        return np.arange(arch["input_size"]) if layer == 0 else np.where(masks_hidden[layer-1])[0]

    # Hidden layers (rows are neurons)
    for l in range(H):
        live_out = np.where(masks_hidden[l])[0]
        live_in = live_in_idxs(l)

        W_rows, b_vals = [], []
        for out_j in live_out:
            runs = [r for r in range(num_runs) if stable_flags_runs[r][l][out_j]]
            assert runs, "Kept neuron must be stable in at least one run"
            # uniform among stable runs; you can weight by stability score if you like
            Wc = [init_params_runs[r][l][0][out_j][:, ...][live_in] for r in runs]
            bc = [init_params_runs[r][l][1][out_j] for r in runs]
            W_rows.append(np.mean(np.stack(Wc, 0), axis=0))
            b_vals.append(np.mean(np.array(bc), axis=0))

        lin = pruned.linears[l]
        lin.weight.data = torch.tensor(np.stack(W_rows, 0), dtype=torch.float32)
        lin.bias.data = torch.tensor(np.array(b_vals), dtype=torch.float32)

    # Output layer: shrink columns to last hidden kept neurons, average those columns
    live_in_last = live_in_idxs(H)
    cols = []
    for in_j in live_in_last:
        runs = [r for r in range(num_runs) if stable_flags_runs[r][H-1][in_j]]
        assert runs
        col_stack = np.stack([init_params_runs[r][H][0][:, in_j] for r in runs], 0)  # (runs, out)
        cols.append(np.mean(col_stack, axis=0))
    W_out = np.stack(cols, axis=1)  # (out, pruned_last_hidden)
    b_out = np.mean(np.stack([init_params_runs[r][H][1] for r in range(num_runs)], 0), axis=0)

    lin_out = pruned.linears[H]
    lin_out.weight.data = torch.tensor(W_out, dtype=torch.float32)
    lin_out.bias.data = torch.tensor(b_out, dtype=torch.float32)
    return pruned


# -------------------
# Experiment
# -------------------
def run_arch(arch, cfg, visualizer=None):
    device = cfg["train"]["device"]
    set_seed(cfg["train"]["seed"])

    # Extract only the parameters needed for MLP
    arch_params = {
        "input_size": arch["input_size"],
        "hidden_sizes": arch["hidden_sizes"],
        "output_size": arch["output_size"],
        "activation": arch["activation"]
    }

    # Data (swap with your real loader)
    if cfg["data"]["task"] == "synthetic-classification":
        X, y = synthetic_classification(cfg["data"]["n_samples"], arch["input_size"], arch["output_size"])
    else:
        raise ValueError("Only synthetic demo provided. Plug in your DataLoader here.")
    X, y = X.to(device), y.to(device)

    # Train multiple randomized nets
    init_params_runs, trained_params_runs, stable_flags_runs = [], [], []
    training_curves = []
    
    for r in range(cfg["train"]["num_networks"]):
        model_init = MLP(**arch_params).to(device)
        model_tr = copy.deepcopy(model_init)
        model_tr, losses = train_one(model_tr, X, y, cfg["train"]["num_iterations"], cfg["train"]["learning_rate"])
        training_curves.append(losses)

        init_params = extract_params(model_init)
        trained_params = extract_params(model_tr)
        init_params_runs.append(init_params)
        trained_params_runs.append(trained_params)

        # per-hidden-layer flags
        flags_per_layer = []
        for l in range(len(arch["hidden_sizes"])):
            mask = neuron_stability_mask(
                init_params[l][0], init_params[l][1],
                trained_params[l][0], trained_params[l][1],
                cfg["prune"]["diff_mode"], cfg["prune"]["threshold"]
            )
            flags_per_layer.append(mask)
        stable_flags_runs.append(flags_per_layer)
        
        # Visualize weight distributions for the first run
        if visualizer and r == 0:
            for l in range(len(arch["hidden_sizes"])):
                visualizer.plot_weight_distributions(init_params, trained_params, arch["name"], l)

    # Aggregate stability -> scores & final masks
    masks_hidden, scores_hidden = [], []
    for l in range(len(arch["hidden_sizes"])):
        layer_stack = np.stack([stable_flags_runs[r][l] for r in range(cfg["train"]["num_networks"])], 0)  # (runs, out_l)
        scores = np.mean(layer_stack, 0)
        mask = scores >= cfg["prune"]["stability_percentage"]
        if cfg["prune"]["keep_at_least_one_per_layer"] and not np.any(mask):
            mask[np.argmax(scores)] = True
        masks_hidden.append(mask)
        scores_hidden.append(scores)
        
    # Visualize stability scores
    if visualizer:
        visualizer.plot_stability_scores(scores_hidden, masks_hidden, arch["name"], cfg["prune"]["stability_percentage"])
        visualizer.plot_weight_changes(init_params_runs[0], trained_params_runs[0], arch["name"])
        visualizer.plot_neuron_correlations(init_params_runs, arch["name"])

    # Build pruned consensus
    pruned = build_pruned_consensus(arch_params, masks_hidden, stable_flags_runs, init_params_runs).to(device)

    # Calculate model sizes
    original_size = sum(p.numel() for p in MLP(**arch_params).parameters())
    pruned_size = sum(p.numel() for p in pruned.parameters())

    # Optional eval
    results = {
        "kept_per_layer": [int(m.sum()) for m in masks_hidden],
        "scores_per_layer": [s.tolist() for s in scores_hidden],
        "original_size": original_size,
        "pruned_size": pruned_size,
        "compression_ratio": original_size / pruned_size,
        "eval": {}
    }
    
    if cfg["eval"]["report_accuracy"]:
        # Calculate average accuracy of initial models
        acc_init = []
        for r in range(cfg["train"]["num_networks"]):
            model = MLP(**arch_params).to(device)
            # Load the initial parameters
            state_dict = {}
            for i, (w, b) in enumerate(init_params_runs[r]):
                state_dict[f"linears.{i}.weight"] = torch.tensor(w)
                state_dict[f"linears.{i}.bias"] = torch.tensor(b)
            model.load_state_dict(state_dict)
            acc_init.append(accuracy(model, X, y))
            
        results["acc_init_mean"] = float(np.mean(acc_init))
        results["acc_init_std"] = float(np.std(acc_init))
        results["acc_pruned"] = accuracy(pruned, X, y)

    if cfg["eval"]["report_ce_loss"]:
        results["loss_pruned"] = ce_loss(pruned, X, y)
        
    if cfg["eval"]["report_training_curves"]:
        results["training_curves"] = training_curves

    return pruned, results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--out", type=str, default="artifacts.json")
    ap.add_argument("--visualize", action="store_true", help="Generate visualizations")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
        
    # Create visualizer if requested
    visualizer = WeightVisualizer() if args.visualize else None

    artifacts: Dict[str, Dict] = {}
    original_accs, pruned_accs = [], []
    original_sizes, pruned_sizes = [], []
    arch_names = []
    
    for arch in cfg["architectures"]:
        model, res = run_arch(arch, cfg, visualizer)
        
        # Save results for comparison plot
        if "acc_init_mean" in res and "acc_pruned" in res:
            original_accs.append(res["acc_init_mean"])
            pruned_accs.append(res["acc_pruned"])
            original_sizes.append(res["original_size"])
            pruned_sizes.append(res["pruned_size"])
            arch_names.append(arch["name"])
        
        # save minimal checkpoint
        artifacts[arch["name"]] = {
            "kept_per_layer": res["kept_per_layer"],
            "scores_per_layer": res["scores_per_layer"],
            "eval": {k: v for k, v in res.items() if k.startswith(("acc_", "loss_"))},
            "sizes": {
                "original": res["original_size"],
                "pruned": res["pruned_size"],
                "compression_ratio": res["compression_ratio"]
            }
        }
        torch.save(model.state_dict(), f"{arch['name']}_pruned.pt")
        print(f"[{arch['name']}] kept {res['kept_per_layer']}, compression: {res['compression_ratio']:.2f}x, acc: {res['eval'].get('acc_init_mean', 0):.3f} -> {res['eval'].get('acc_pruned', 0):.3f}")

    # Create comparison plot if we have multiple architectures
    if args.visualize and len(arch_names) > 1:
        visualizer.plot_pruning_impact(original_accs, pruned_accs, original_sizes, pruned_sizes, arch_names)

    with open(args.out, "w") as f:
        json.dump(artifacts, f, indent=2)
    print(f"Saved JSON summary to {args.out} and model weights per-arch to <name>_pruned.pt")

if __name__ == "__main__":
    main()