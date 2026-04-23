import time
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms
from layer_classes import Conv2dCfg, DropoutCfg, FlattenCfg, LinearCfg, MaxPool2dCfg, GlobalAvgPoolCfg, BatchNorm1dCfg, BatchNorm2dCfg, ResBlockCfg
from model import DynamicNet
from optimizer import SAOptimizer, GeneticOptimizer, ABCOptimizer, RLOptimizer, TransformerOptimizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
N_SAMPLES_TRAIN_IMG = 2000
N_SAMPLES_TEST_IMG = 500
N_STATS_RUNS = 5
ITERATIONS_OPTIM = 40

def get_dataset(task_type):
    if task_type == 'california_housing':
        data = fetch_california_housing()
        X, y = data.data, data.target
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train).view(-1, 1)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test).view(-1, 1)
        
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)
        return train_loader, test_loader, (1, 8), 1

    elif task_type == 'breast_cancer':
        data = load_breast_cancer()
        X, y = data.data, data.target
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.LongTensor(y_test)
        
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=BATCH_SIZE)
        return train_loader, test_loader, (1, 30), 2

    elif 'fashion_mnist' in task_type:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        try:
            train_data = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
            test_data = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
        except:
            return None, None, None, None

        train_indices = torch.randperm(len(train_data))[:N_SAMPLES_TRAIN_IMG]
        test_indices = torch.randperm(len(test_data))[:N_SAMPLES_TEST_IMG]
        
        train_subset = Subset(train_data, train_indices)
        test_subset = Subset(test_data, test_indices)
        
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE)
        return train_loader, test_loader, (1, 28, 28), 10

def get_initial_arch(task_type, input_shape, output_dim):
    layers = []
    if 'resblock' in task_type:
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
    elif 'fashion_mnist' in task_type:
        layers.append(Conv2dCfg(in_channels=0, out_channels=16, kernel_size=3, padding=1, activation=nn.ReLU))
        layers.append(BatchNorm2dCfg(num_features=16))
        layers.append(MaxPool2dCfg(kernel_size=2, stride=2, padding=0))
        layers.append(FlattenCfg())
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
        self.train_loader, self.test_loader, self.in_shape, self.out_dim = get_dataset(task_type)
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
            dataset=self.train_loader, 
            **self.kwargs
        )
        
        def adaptive_evaluate(genome, train_epochs=5):
            try:
                model = DynamicNet(genome, input_shape=self.in_shape)
                model.to(DEVICE)
                
                base_dataset = self.train_loader.dataset
                batch_size = self.train_loader.batch_size
                train_size = int(0.8 * len(base_dataset))
                val_size = len(base_dataset) - train_size
                
                train_subset, val_subset = torch.utils.data.random_split(base_dataset, [train_size, val_size])
                inner_train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
                inner_val_loader = DataLoader(val_subset, batch_size=batch_size)
                
                if 'regression' in self.task_type or 'california' in self.task_type:
                    criterion = nn.MSELoss()
                else:
                    criterion = nn.CrossEntropyLoss()
                
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                
                model.train()
                for epoch in range(train_epochs):
                    for X_batch, y_batch in inner_train_loader:
                        if len(self.in_shape) == 2 and X_batch.dim() > 2:
                            X_batch = X_batch.view(X_batch.size(0), -1)
                        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
                        optimizer.zero_grad()
                        out = model(X_batch)
                        loss = criterion(out, y_batch)
                        loss.backward()
                        optimizer.step()

                model.eval()
                if 'regression' in self.task_type or 'california' in self.task_type:
                    total_loss = 0
                    total_samples = 0
                    with torch.no_grad():
                        for X_b, y_b in inner_val_loader:
                            if len(self.in_shape) == 2 and X_b.dim() > 2: 
                                X_b = X_b.view(X_b.size(0), -1)
                            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
                            out = model(X_b)
                            total_loss += criterion(out, y_b).item() * X_b.size(0)
                            total_samples += X_b.size(0)
                    return -(total_loss / max(1, total_samples))
                else:
                    correct = 0
                    total = 0
                    with torch.no_grad():
                        for X_b, y_b in inner_val_loader:
                            if len(self.in_shape) == 2 and X_b.dim() > 2: 
                                X_b = X_b.view(X_b.size(0), -1)
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
    tasks = [ "breast_cancer", "fashion_mnist_simple"]#"california_housing",, "fashion_mnist_resblock"
    
    optimizers = [
        #("Simulated Annealing", SAOptimizer, {"temp_init": 100, "cooling_rate": 0.8}),
        #("Genetic Algorithm", GeneticOptimizer, {"pop_size": 10, "mutation_rate": 0.3}),
        #("ABC Algorithm", ABCOptimizer, {"pop_size": 10, "limit": 4}),
        #("RL Controller", RLOptimizer, {"max_layers": 50})
        ("Transformer", TransformerOptimizer, {"max_layers":50, "entropy_fct":"default"})
    ]

    print("\n" + "="*130)
    print(f"REAL DATASETS BENCHMARK (Runs: {N_STATS_RUNS} | Iterations/Gen: {ITERATIONS_OPTIM})")
    print("="*130)

    results = []

    for task in tasks:
        print(f"\n>>> TASK: {task.upper()}")
        
        for opt_name, opt_cls, opt_params in optimizers:
            print(f"  > Running {opt_name}...", end="", flush=True)
            
            metrics = {"scores": [], "times": [], "gains": [], "depths": [], "iters": []}

            for i in range(N_STATS_RUNS):
                runner = BenchmarkWrapper(opt_cls, task, **opt_params)
                res = runner.run(ITERATIONS_OPTIM)
                
                if res["score"] > -float('inf'):
                    metrics["scores"].append(res["score"])
                    metrics["times"].append(res["search_time"])
                    metrics["gains"].append(res["gain"])
                    metrics["depths"].append(res["depth_delta"])
                    metrics["iters"].append(res["best_iter"])
            
            if len(metrics["scores"]) > 0:
                results.append({
                    "task": task,
                    "algo": opt_name,
                    "score_str": f"{np.mean(metrics['scores']):.2f} ± {np.std(metrics['scores']):.2f}",
                    "gain": np.mean(metrics["gains"]),
                    "iter": np.mean(metrics["iters"]),
                    "depth": np.mean(metrics["depths"]),
                    "time": np.mean(metrics["times"])
                })
                print(" Done.")
            else:
                print(" FAILED.")

    print("\n" + "="*130)
    header = f"{'TASK':<25} | {'ALGORITHM':<22} | {'TEST SCORE (Avg±Std)':<20} | {'GAIN':<8} | {'ITER':<6} | {'Δ DEPTH':<8} | {'TIME(s)':<8}"
    print(header)
    print("-" * 130)
    for r in results:
        print(f"{r['task']:<25} | {r['algo']:<22} | {r['score_str']:<20} | {r['gain']:<8.2f} | {r['iter']:<6.1f} | {r['depth']:<+8.1f} | {r['time']:<8.2f}")