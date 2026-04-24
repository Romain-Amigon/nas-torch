import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from abc import ABC, abstractmethod
from sklearn.datasets import make_classification, make_regression, make_moons
from torchvision import datasets, transforms
from layer_classes import Conv2dCfg, DropoutCfg, FlattenCfg, LinearCfg, MaxPool2dCfg, GlobalAvgPoolCfg, BatchNorm1dCfg, BatchNorm2dCfg, ResBlockCfg
from model import DynamicNet
from optimizer import SAOptimizer, GeneticOptimizer, ABCOptimizer, RLOptimizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
N_SAMPLES = 500 #nmb de points pour rég et classif linéa
N_STATS_RUNS = 10 # nmb de run pour les stats
ITERATIONS_OPTIM = 30 # nmb d'ité pour recherche architec



def get_dataset(task_type):
    if task_type == 'linear_regression':
        X, y = make_regression(n_samples=N_SAMPLES, n_features=20, noise=0.1, random_state=42)
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y).view(-1, 1) 
        return DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE), (1, 20), 1
    elif task_type == 'linear_classification':
        X, y = make_moons(n_samples=N_SAMPLES, noise=0.1, random_state=42)
        X = torch.FloatTensor(X)
        y = torch.LongTensor(y)
        return DataLoader(TensorDataset(X, y), batch_size=BATCH_SIZE), (1, 2), 2
    elif 'cnn' in task_type:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)) 
        ])
        try:
            full_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        except:
            return None, None, None
        indices = torch.randperm(len(full_dataset))[:N_SAMPLES]
        subset = Subset(full_dataset, indices)
        return DataLoader(subset, batch_size=BATCH_SIZE), (1, 28, 28), 10

def get_initial_arch(task_type, input_shape, output_dim):
    layers = []
    if task_type == 'cnn_resblock':
        layers.append(Conv2dCfg(in_channels=0, out_channels=16, kernel_size=3, padding=1, activation=nn.ReLU))
        layers.append(BatchNorm2dCfg(num_features=16)) 
        sub_block = [
            Conv2dCfg(in_channels=0, out_channels=16, kernel_size=3, padding=1, activation=nn.ReLU),
            BatchNorm2dCfg(num_features=16),
            Conv2dCfg(in_channels=0, out_channels=16, kernel_size=3, padding=1, activation=None) 
        ]
        layers.append(ResBlockCfg(sub_layers=sub_block))
        layers.append(BatchNorm2dCfg(num_features=16)) 
        layers.append(GlobalAvgPoolCfg())
        layers.append(LinearCfg(in_features=0, out_features=output_dim, activation=None))
    elif task_type == 'cnn_simple':
        layers.append(Conv2dCfg(in_channels=0, out_channels=16, kernel_size=3, padding=1, activation=nn.ReLU))
        layers.append(BatchNorm2dCfg(num_features=16))
        layers.append(Conv2dCfg(in_channels=0, out_channels=16, kernel_size=3, padding=1, activation=nn.ReLU))
        layers.append(BatchNorm2dCfg(num_features=16))
        layers.append(Conv2dCfg(in_channels=0, out_channels=16, kernel_size=3, padding=1, activation=nn.ReLU))
        layers.append(BatchNorm2dCfg(num_features=16)) 
        layers.append(GlobalAvgPoolCfg())
        layers.append(LinearCfg(in_features=0, out_features=output_dim, activation=None))
    elif task_type == "linear_regression":
        layers.append(FlattenCfg()) 
        layers.append(LinearCfg(in_features=0, out_features=32, activation=nn.ReLU))
        layers.append(LinearCfg(in_features=0, out_features=output_dim, activation=None)) 
    else: 
        layers.append(FlattenCfg()) 
        layers.append(LinearCfg(in_features=0, out_features=32, activation=nn.ReLU))
        layers.append(LinearCfg(in_features=0, out_features=output_dim, activation=None))
    return layers

class BenchmarkWrapper:
    def __init__(self, optimizer_cls, task_type, **kwargs):
        self.optimizer_cls = optimizer_cls
        self.task_type = task_type
        self.kwargs = kwargs
        self.loader, self.in_shape, self.out_dim = get_dataset(task_type)
        self.init_layers = get_initial_arch(task_type, self.in_shape, self.out_dim)

    def measure_inference_time(self, model, n_warmup=5, n_measures=20):
        if len(self.in_shape) == 2:
            dummy_input = torch.randn(1, self.in_shape[1]).to(DEVICE)
        else:
            dummy_input = torch.randn(1, *self.in_shape).to(DEVICE)
        model.eval()
        try:
            with torch.no_grad():
                for _ in range(n_warmup): _ = model(dummy_input)
                times = []
                for _ in range(n_measures):
                    if DEVICE.type == 'cuda': torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    _ = model(dummy_input)
                    if DEVICE.type == 'cuda': torch.cuda.synchronize()
                    times.append((time.perf_counter() - t0) * 1000)
            return np.min(times), np.mean(times), np.max(times)
        except Exception:
            return 0, 0, 0

    def run(self, n_iterations):
        opt = self.optimizer_cls(
            layers=copy.deepcopy(self.init_layers), 
            dataset=self.loader, 
            **self.kwargs
        )
        
        def adaptive_evaluate(genome, train_epochs=3):
            try:
                model = DynamicNet(genome, input_shape=self.in_shape)
                model.to(DEVICE)
                if self.task_type == 'linear_regression': criterion = nn.MSELoss()
                else: criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
                model.train()
                for epoch in range(train_epochs):
                    for X_batch, y_batch in self.loader:
                        if len(self.in_shape) == 2 and X_batch.dim() > 2: X_batch = X_batch.view(X_batch.size(0), -1)
                        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                        optimizer.zero_grad()
                        out = model(X_batch)
                        loss = criterion(out, y_batch)
                        loss.backward()
                        optimizer.step()
                model.eval()
                if self.task_type == 'linear_regression':
                    total_loss = 0
                    with torch.no_grad():
                        for X_b, y_b in self.loader:
                            if len(self.in_shape) == 2 and X_b.dim() > 2: X_b = X_b.view(X_b.size(0), -1)
                            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                            out = model(X_b)
                            total_loss += criterion(out, y_b).item()
                    return -(total_loss / len(self.loader))
                else:
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for X_b, y_b in self.loader:
                            if len(self.in_shape) == 2 and X_b.dim() > 2: X_b = X_b.view(X_b.size(0), -1)
                            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                            out = model(X_b)
                            _, predicted = torch.max(out.data, 1)
                            total += y_b.size(0)
                            correct += (predicted == y_b).sum().item()
                    return 100.0 * correct / max(1, total)
            except Exception:
                return -float('inf')

        opt.evaluate = adaptive_evaluate
        start_time = time.time()
        
        best_sol, optim_stats = opt.run(n_iterations)
        
        search_duration = time.time() - start_time
        try:
            model_final = DynamicNet(best_sol, input_shape=self.in_shape).to(DEVICE)
            inf_min, inf_mean, inf_max = self.measure_inference_time(model_final)
            n_params = model_final.count_parameters()
        except:
            inf_min, inf_mean, inf_max = 0, 0, 0
            n_params = 0
            
        initial_depth = len(self.init_layers)
        final_depth = len(best_sol)
        depth_delta = final_depth - initial_depth

        return {
            "score": opt.best_score,
            "search_time": search_duration,
            "inf_mean": inf_mean,
            "params": n_params,
            "gain": optim_stats["gain"],
            "best_iter": optim_stats["best_iter"],
            "depth_delta": depth_delta
        }

if __name__ == "__main__":
    benchmarks = ["linear_regression", "linear_classification", "cnn_simple", "cnn_resblock"]
    final_stats = {}
    print("\n" + "="*95)
    print("BENCHMARK START")
    print("="*95)
    for task in benchmarks:
        print(f"\n>> {task.upper()} ")
        metrics = {
            "scores": [], "times": [], "inf": [], "params": [],
            "gains": [], "iters": [], "depths": []
        }   
        for i in range(N_STATS_RUNS):
            print(f"  > Run {i+1}...", end="", flush=True)
            #runner = BenchmarkWrapper(SAOptimizer, task, temp_init=100, cooling_rate=0.7)
            #runner = BenchmarkWrapper(GeneticOptimizer, task, pop_size=50)
            #runner = BenchmarkWrapper(ABCOptimizer, task, pop_size=20)
            runner = BenchmarkWrapper(RLOptimizer,task)
            res = runner.run(ITERATIONS_OPTIM)
            if res["score"] == -float('inf'):
                print(" FAILED")
            else:
                metrics["scores"].append(res["score"])
                metrics["times"].append(res["search_time"])
                metrics["inf"].append(res["inf_mean"])
                metrics["params"].append(res["params"])
                metrics["gains"].append(res["gain"])
                metrics["iters"].append(res["best_iter"])
                metrics["depths"].append(res["depth_delta"])
                print(f" Done. Score: {res['score']:.4f} | Gain: {res['gain']:.2f}")
                
        if len(metrics["scores"]) > 0:
            final_stats[task] = {
                "score_avg": np.mean(metrics["scores"]),
                "score_std": np.std(metrics["scores"]),
                "score_var": np.var(metrics["scores"]),
                "time_avg": np.mean(metrics["times"]),
                "inf_avg": np.mean(metrics["inf"]),
                "params_avg": np.mean(metrics["params"]),
                "gain_avg": np.mean(metrics["gains"]),
                "iter_avg": np.mean(metrics["iters"]),
                "depth_avg": np.mean(metrics["depths"]),
                "best_score" : max(metrics["scores"])
            }
        else:
            final_stats[task] = None

    print("\n" + "="*115)
    header = f"{'TASK':<22} | {'SCORE (Avg±Std)':<18} | {'GAIN':<8} | {'BEST ITER':<10} | {'DEPTH Δ':<8} | {'INFER':<8} | {'BEST SCORE':<4}"
    print(header)
    print("-" * 115)
    for task, st in final_stats.items():
        if st:
            score_str = f"{st['score_avg']:.2f} ± {st['score_std']:.2f}"
            print(f"{task:<22} | {score_str:<18} | {st['gain_avg']:<8.2f} | {st['iter_avg']:<10.1f} | {st['depth_avg']:<+8.1f} | {st['inf_avg']:.2f} ms | {st['best_score']:.4f}")
        else:
            print(f"{task:<22} | FAILED ALL RUNS")