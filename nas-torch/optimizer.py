import copy
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import DataLoader, TensorDataset
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Optional, Any, Union, Callable

from layer_classes import (
    Conv2dCfg, DropoutCfg, FlattenCfg, LinearCfg, 
    MaxPool2dCfg, GlobalAvgPoolCfg, BatchNorm1dCfg, 
    BatchNorm2dCfg, ResBlockCfg
)
from model import DynamicNet

class Optimizer(ABC):
    """
    Abstract base class for all Neural Architecture Search (NAS) optimizers.
    Handles evaluation, topological mutation, and dataset introspection.
    """
    def __init__(self, layers: List[Any], search_space: Optional[Any] = None, dataset: Any = None):
        """
        Args:
            layers (list): Initial architecture configuration (list of layer configs).
            search_space (Any, optional): Defines the boundaries of the search space.
            dataset (Any): The dataset used for training and evaluation.
        """
        self.layers = layers
        self.search_space = search_space
        self.dataset = dataset
        self.best_score = -float('inf')
        self.best_arch = None
        
        # Introspect the dataset to determine the number of output features required
        base_dataset = self.dataset.dataset if hasattr(self.dataset, 'dataset') else self.dataset
        _, sample_target = base_dataset[0]
        
        if hasattr(base_dataset, 'classes'):
            self.out_features = len(base_dataset.classes)
        elif isinstance(sample_target, torch.Tensor):
            if sample_target.dtype in [torch.float16, torch.float32, torch.float64]:
                self.out_features = sample_target.numel()
            else:
                all_targets = [y for _, y in base_dataset]
                self.out_features = int(torch.tensor(all_targets).max().item()) + 1
        elif isinstance(sample_target, int):
            all_targets = [y for _, y in base_dataset]
            self.out_features = max(all_targets) + 1
        else:
            self.out_features = 1
    
    def evaluate(self, genome: List[Any], train_epochs: int = 10, patience: int = 5) -> float:
        """
        Evaluates an architecture (genome) by training it on a data subset.
        Acts as the proxy evaluator for the NAS.
        
        Args:
            genome (list): The architecture configuration to evaluate.
            train_epochs (int): Maximum number of training epochs.
            patience (int): Early stopping patience if validation accuracy stalls.
            
        Returns:
            float: The best validation score achieved, or -inf if the topology is invalid.
        """
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            base_dataset = self.dataset.dataset if hasattr(self.dataset, 'dataset') else self.dataset
            batch_size = self.dataset.batch_size if hasattr(self.dataset, 'batch_size') else 32
            
            sample_input, sample_target = base_dataset[0]
            input_shape = sample_input.shape

            # Dynamically select loss function based on task type (Binary vs Multiclass)
            if isinstance(sample_target, torch.Tensor):
                if sample_target.dtype in [torch.float16, torch.float32, torch.float64] and sample_target.numel() == 1:
                    criterion = nn.BCEWithLogitsLoss()
                    task = "binary"
                else:
                    criterion = nn.CrossEntropyLoss()
                    task = "multiclass"
            else:
                criterion = nn.CrossEntropyLoss()
                task = "multiclass"

            model = DynamicNet(genome, input_shape=input_shape)
            model.to(device)

            # Split dataset into train and validation for proxy evaluation
            train_size = int(0.7 * len(base_dataset))
            val_size = len(base_dataset) - train_size
            
            generator = torch.Generator().manual_seed(42) # Seed for reproducibility
            train_dataset, val_dataset = torch.utils.data.random_split(base_dataset, [train_size, val_size], generator=generator)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            best_val_acc = 0.0
            patience_counter = 0

            for epoch in range(train_epochs):
                model.train()
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    
                    if task == "binary":
                        targets = targets.view(-1, 1).float()
                    else:
                        targets = targets.long()
                        if targets.dim() > 1 and targets.shape[1] == 1:
                            targets = targets.squeeze(1)

                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()

                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        
                        if task == "binary":
                            targets = targets.view(-1, 1).float()
                            predicted = (outputs > 0).float()
                        else:
                            targets = targets.long()
                            if targets.dim() > 1 and targets.shape[1] == 1:
                                targets = targets.squeeze(1)
                            _, predicted = outputs.max(1)
                            
                        total += targets.size(0)
                        correct += predicted.eq(targets).sum().item()

                current_acc = 0.0 if total == 0 else 100. * correct / total

                # Early stopping logic
                if current_acc > best_val_acc:
                    best_val_acc = current_acc
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    break

            return best_val_acc

        except Exception:
            # Return -inf if architecture generates a PyTorch compilation/tensor error
            return -float('inf')

    def neighbor(self, current_configs: List[Any]) -> List[Any]:
        """
        Topological mutation operator. Mutates the network by adding, removing, 
        swapping activations, or tweaking hyperparameters.
        """
        new_configs = copy.deepcopy(current_configs)
        options = ["param", "add_layer", "remove_layer", "swap_activation"]
        mutation_type = random.choice(options)

        def is_linear_context_check(target_list: List[Any], idx: int) -> bool:
            """Checks if the current index is in a flattened/linear context."""
            if idx == 0: 
                if len(target_list) > 0 and isinstance(target_list[0], (LinearCfg, FlattenCfg)):
                    return True
                return False
            
            prev_layer = target_list[idx - 1]
            if isinstance(prev_layer, (LinearCfg, FlattenCfg, GlobalAvgPoolCfg)):
                return True
            return False

        def get_mutable_layers(current_list: List[Any], is_root: bool = True) -> List[Any]:
            """Recursively retrieves layers that can be mutated."""
            candidates = []
            limit = len(current_list) - 1 if is_root else len(current_list)
            for i in range(limit):
                layer = current_list[i]
                if isinstance(layer, ResBlockCfg):
                    candidates.extend(get_mutable_layers(layer.sub_layers, is_root=False))
                elif hasattr(layer, 'activation') or hasattr(layer, 'out_channels') or hasattr(layer, 'out_features'):
                    candidates.append(layer)
            return candidates

        def get_mutable_lists(current_list: List[Any], is_root: bool = True) -> List[Tuple[List[Any], bool]]:
            """Retrieves sub-lists (like ResBlocks) where layers can be added or removed."""
            candidates = [(current_list, is_root)]
            for layer in current_list:
                if isinstance(layer, ResBlockCfg):
                    candidates.extend(get_mutable_lists(layer.sub_layers, is_root=False))
            return candidates

        if mutation_type == "param":
            candidates = get_mutable_layers(new_configs, is_root=True)
            if candidates:
                target = random.choice(candidates)
                self._mutate_layer_param(target)

        elif mutation_type == "swap_activation":
            candidates = get_mutable_layers(new_configs, is_root=True)
            valid = [l for l in candidates if hasattr(l, 'activation')]
            if valid:
                target = random.choice(valid)
                acts = [nn.ReLU, nn.Tanh, nn.LeakyReLU, None] 
                target.activation = random.choice(acts)

        elif mutation_type == "add_layer":
            list_candidates = get_mutable_lists(new_configs, is_root=True)
            if list_candidates:
                target_list, is_root = random.choice(list_candidates)
                
                if is_root:
                    idx = random.randint(0, len(target_list) - 1) if len(target_list) >= 1 else 0
                else:
                    idx = random.randint(0, len(target_list))

                is_linear = is_linear_context_check(target_list, idx)
                new_layer = self._get_random_layer(linear_context=is_linear)
                target_list.insert(idx, new_layer)

        elif mutation_type == "remove_layer":
            list_candidates = get_mutable_lists(new_configs, is_root=True)
            valid_candidates = []
            for lst, is_root in list_candidates:
                if is_root:
                    if len(lst) > 2: valid_candidates.append((lst, True))
                else:
                    if len(lst) > 0: valid_candidates.append((lst, False))
            
            if valid_candidates:
                target_list, is_root = random.choice(valid_candidates)
                if is_root:
                    idx = random.randint(0, len(target_list) - 2)
                else:
                    idx = random.randint(0, len(target_list) - 1)
                del target_list[idx]

        return new_configs

    def _mutate_layer_param(self, layer: Any):
        """Slightly alters a layer's hyperparameter (e.g., kernel size, channels)."""
        if isinstance(layer, Conv2dCfg):
            choice = random.choice(["kernel", "channels"])
            if choice == "kernel":
                delta = random.choice([-2, 2])
                new_k = layer.kernel_size + delta
                layer.kernel_size = int(np.clip(new_k, 1, 7))
                layer.padding = layer.kernel_size // 2
            elif choice == "channels":
                delta = random.choice([-8, -4, 4, 8])
                layer.out_channels = max(4, int(layer.out_channels + delta))

        elif isinstance(layer, LinearCfg):
            delta = random.choice([-16, 8, 8, 16])
            layer.out_features = max(4, int(layer.out_features + delta))

        elif isinstance(layer, DropoutCfg):
            delta = random.uniform(-0.1, 0.1)
            layer.p = np.clip(layer.p + delta, 0.0, 0.8)

    def _get_random_layer(self, linear_context: bool = False) -> Any:
        """Generates a random valid layer based on spatial vs linear context."""
        if linear_context:
            type_ = random.choice(["linear", "dropout", "bn1d"])
            if type_ == "linear":
                return LinearCfg(in_features=0, out_features=random.randint(16, 128), activation=nn.ReLU)
            elif type_ == "dropout":
                return DropoutCfg(p=0.3)
            elif type_ == "bn1d":
                return BatchNorm1dCfg(num_features=0)
        else:
            type_ = random.choice(["conv", "pool", "bn2d", "dropout"])
            if type_ == "conv":
                k = random.choice([3, 5])
                return Conv2dCfg(in_channels=0, out_channels=random.randint(8, 64), 
                                 kernel_size=k, padding=k//2, activation=nn.ReLU)
            elif type_ == "pool":
                return MaxPool2dCfg(kernel_size=2, stride=2, padding=0)
            elif type_ == "bn2d":
                return BatchNorm2dCfg(num_features=0)
            elif type_ == "dropout":
                return DropoutCfg(p=0.2)
        return DropoutCfg(p=0.1)

    @abstractmethod
    def run(self, n_iterations: int) -> Tuple[List[Any], Dict[str, Any]]:
        """Executes the search algorithm."""
        pass


class SAOptimizer(Optimizer):
    """
    Simulated Annealing Optimizer.
    Stochastic local search that probabilistically accepts worse solutions to escape local minima.
    """
    def __init__(self, layers=None, search_space=None, temp_init=100, cooling_rate=0.95, **kwargs):
        super().__init__(layers, search_space, **kwargs)
        self.T = temp_init
        self.alpha = cooling_rate

    def run(self, n_iterations: int) -> Tuple[List[Any], Dict[str, Any]]:
        current_sol = copy.deepcopy(self.layers)
        current_score = self.evaluate(current_sol)
        
        if current_score == -float('inf'):
            current_score = 0.0 
        
        initial_score = current_score
        self.best_arch = current_sol
        self.best_score = current_score
        
        best_iter = -1

        for i in range(n_iterations):
            neighbor = self.neighbor(current_sol)
            neighbor_score = self.evaluate(neighbor)
            
            if neighbor_score == -float('inf'): continue 

            delta = neighbor_score - current_score
            if delta > 0 or np.random.rand() < np.exp(delta / self.T):
                current_sol = neighbor
                current_score = neighbor_score
                if current_score > self.best_score:
                    self.best_score = current_score
                    self.best_arch = copy.deepcopy(current_sol)
                    best_iter = i
                    print(f"Iter {i}: New Best! Score {self.best_score:.2f}")
            self.T *= self.alpha
            
        stats = {
            "initial_score": initial_score,
            "best_score": self.best_score,
            "best_iter": best_iter,
            "gain": self.best_score - initial_score
        }
        return self.best_arch, stats


class GeneticOptimizer(Optimizer):
    """
    Genetic Algorithm Optimizer.
    Uses tournament selection and crossover to evolve a population of architectures.
    """
    def __init__(self, layers=None, search_space=None, pop_size=10, mutation_rate=0.1, **kwargs):
        super().__init__(layers, search_space, **kwargs)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.population = []

    def crossover(self, parent1: List[Any], parent2: List[Any]) -> List[Any]:
        if len(parent1) < 3 or len(parent2) < 3:
            return copy.deepcopy(parent1)
        idx1 = random.randint(1, len(parent1) - 2)
        idx2 = random.randint(1, len(parent2) - 2)

        child_layers = parent1[:idx1] + parent2[idx2:]
        return child_layers

    def tournament_selection(self, valid_pop: List[Any], scores: List[float], k: int = 3) -> List[Any]:
        k = min(k, len(valid_pop))
        selected_indices = random.sample(range(len(valid_pop)), k)
        best_idx = max(selected_indices, key=lambda idx: scores[idx])
        return copy.deepcopy(valid_pop[best_idx])

    def run(self, n_generations: int) -> Tuple[List[Any], Dict[str, Any]]:
        init_arch = copy.deepcopy(self.layers)
        self.population = [self.neighbor(init_arch) for _ in range(self.pop_size)]
        
        initial_score = self.evaluate(init_arch)
        if initial_score == -float('inf'): initial_score = 0.0
        
        self.best_arch = init_arch
        self.best_score = initial_score
        best_gen = -1

        for g in range(n_generations):
            scores = []
            valid_pop = []
            
            for ind in self.population:
                score = self.evaluate(ind)
                if score > -float('inf'):
                    scores.append(score)
                    valid_pop.append(ind)
                    
                    if score > self.best_score:
                        self.best_score = score
                        self.best_arch = copy.deepcopy(ind)
                        best_gen = g
                        print(f"Gen {g}: New Best! Score {self.best_score:.2f}")
            
            if not valid_pop:
                self.population = [self.neighbor(init_arch) for _ in range(self.pop_size)]
                continue

            sorted_indices = np.argsort(scores)[::-1]
            top_k = max(2, int(len(valid_pop) * 0.2))
            next_gen = [valid_pop[i] for i in sorted_indices[:top_k]]
            
            while len(next_gen) < self.pop_size:
                p1 = self.tournament_selection(valid_pop, scores)
                p2 = self.tournament_selection(valid_pop, scores)
                
                child = self.crossover(p1, p2)
                
                if random.random() < self.mutation_rate:
                    child = self.neighbor(child)
                
                next_gen.append(child)
            
            self.population = next_gen
            
        stats = {
            "initial_score": initial_score,
            "best_score": self.best_score,
            "best_iter": best_gen, 
            "gain": self.best_score - initial_score
        }
        return self.best_arch, stats


class ABCOptimizer(Optimizer):
    """
    Artificial Bee Colony (ABC) Optimizer.
    Swarm intelligence algorithm utilizing employed, onlooker, and scout bees for robust search.
    """
    def __init__(self, layers=None, search_space=None, pop_size=10, limit=5, patience=0, **kwargs):
        super().__init__(layers, search_space, **kwargs)
        self.pop_size = pop_size
        self.limit = limit
        self.patience = patience
        self.population = []
        self.fitness = []
        self.trials = []

    def run(self, n_iterations: int) -> Tuple[List[Any], Dict[str, Any]]:
        init_arch = copy.deepcopy(self.layers)
        initial_score = self.evaluate(init_arch)
        if initial_score == -float('inf'): initial_score = 0.0
        self.population = [self.neighbor(init_arch) for _ in range(self.pop_size)]
        self.fitness = [self.evaluate(ind) for ind in self.population]
        self.trials = [0] * self.pop_size

        for i in range(self.pop_size):
            if self.fitness[i] > self.best_score and self.fitness[i] != -float('inf'):
                self.best_score = self.fitness[i]
                self.best_arch = copy.deepcopy(self.population[i])
        
        iters_without_improvement = 0
        best_gen = 0
        
        for it in range(n_iterations):
            previous_best = self.best_score
            # Employed Bees Phase
            for i in range(self.pop_size):
                new_arch = self.neighbor(self.population[i])
                new_fit = self.evaluate(new_arch)

                if new_fit > self.fitness[i]:
                    self.population[i] = new_arch
                    self.fitness[i] = new_fit
                    self.trials[i] = 0
                    if new_fit > self.best_score:
                        self.best_score = new_fit
                        self.best_arch = copy.deepcopy(new_arch)
                else:
                    self.trials[i] += 1

            valid_fits = [f for f in self.fitness if f != -float('inf')]
            if not valid_fits:
                continue

            # Onlooker Bees Phase
            min_fit = min(valid_fits)
            shifted_fits = [f - min_fit + 1e-5 if f != -float('inf') else 0 for f in self.fitness]
            total_fit = sum(shifted_fits)
            probs = [f / total_fit for f in shifted_fits]

            m = 0
            i = 0
            while m < self.pop_size:
                if random.random() < probs[i]:
                    m += 1
                    new_arch = self.neighbor(self.population[i])
                    new_fit = self.evaluate(new_arch)

                    if new_fit > self.fitness[i]:
                        self.population[i] = new_arch
                        self.fitness[i] = new_fit
                        self.trials[i] = 0
                        if new_fit > self.best_score:
                            self.best_score = new_fit
                            self.best_arch = copy.deepcopy(new_arch)
                    else:
                        self.trials[i] += 1
                i = (i + 1) % self.pop_size

            # Scout Bees Phase (Reset exhausted food sources)
            for i in range(self.pop_size):
                if self.trials[i] >= self.limit:
                    self.population[i] = self.neighbor(init_arch)
                    self.fitness[i] = self.evaluate(self.population[i])
                    self.trials[i] = 0

            print(f"ABC Iter {it}: Best Score {self.best_score:.2f}")
            
            if self.best_score > previous_best + 1e-4:
                iters_without_improvement = 0
                best_gen = it
            else:
                iters_without_improvement += 1
            
            if self.patience > 0 and iters_without_improvement >= self.patience:
                print(f"Early stopping triggered at iteration {it}: No improvement for {self.patience} iterations.")
                break

        stats = {
            "initial_score": initial_score,
            "best_score": self.best_score,
            "best_iter": best_gen,
            "gain": self.best_score - initial_score
        }
        return self.best_arch, stats


class ControllerRNN(nn.Module):
    """Autoregressive RNN (LSTM) controller used for Reinforcement Learning NAS."""
    def __init__(self, num_tokens: int, hidden_size: int = 64):
        super().__init__()
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(num_tokens, hidden_size)
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, num_tokens)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor):
        embed = self.embedding(x)
        h, c = self.lstm(embed, (h, c))
        logits = self.decoder(h)
        return logits, h, c


class RLOptimizer(Optimizer):
    """
    Reinforcement Learning Optimizer using an LSTM Controller.
    Optimizes via Policy Gradient (REINFORCE) with a multi-objective reward.
    """
    def __init__(self, layers=None, search_space=None, dataset=None, max_layers=8, hidden_size=64, lr=0.01, **kwargs):
        super().__init__(layers, search_space, dataset)
        self.max_layers = max_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define the discrete token vocabulary
        self.vocab = [
            "conv_3_16", "conv_3_32", "conv_5_16", 
            "pool_2", 
            "linear_32", "linear_64", 
            "dropout_0.2", "dropout_0.5",
            "bn2d", "bn1d", "flatten", "avgpool",
            "stop"
        ]
        self.num_tokens = len(self.vocab)
        
        self.controller = ControllerRNN(self.num_tokens, hidden_size).to(self.device)
        self.ctrl_optimizer = optim.Adam(self.controller.parameters(), lr=lr)
        self.baseline = 0.0
        self.entropy_weight = 0.05  # Exploration weight

    def _token_to_cfg(self, token: str, is_linear_context: bool) -> Optional[Any]:
        if token == "conv_3_16" and not is_linear_context:
            return Conv2dCfg(in_channels=0, out_channels=16, kernel_size=3, padding=1, activation=nn.ReLU)
        elif token == "conv_3_32" and not is_linear_context:
            return Conv2dCfg(in_channels=0, out_channels=32, kernel_size=3, padding=1, activation=nn.ReLU)
        elif token == "conv_5_16" and not is_linear_context:
            return Conv2dCfg(in_channels=0, out_channels=16, kernel_size=5, padding=2, activation=nn.ReLU)
        elif token == "pool_2" and not is_linear_context:
            return MaxPool2dCfg(kernel_size=2, stride=2, padding=0)
        elif token == "bn2d" and not is_linear_context:
            return BatchNorm2dCfg(num_features=0)
        elif token == "avgpool" and not is_linear_context:
            return GlobalAvgPoolCfg()
        elif token == "flatten":
            return FlattenCfg()
        elif token == "linear_32":
            return LinearCfg(in_features=0, out_features=32, activation=nn.ReLU)
        elif token == "linear_64":
            return LinearCfg(in_features=0, out_features=64, activation=nn.ReLU)
        elif token == "dropout_0.2":
            return DropoutCfg(p=0.2)
        elif token == "dropout_0.5":
            return DropoutCfg(p=0.5)
        elif token == "bn1d" and is_linear_context:
            return BatchNorm1dCfg(num_features=0)
        return None

    def generate_architecture(self) -> Tuple[List[Any], torch.Tensor, torch.Tensor]:
        h = torch.zeros(1, self.controller.hidden_size).to(self.device)
        c = torch.zeros(1, self.controller.hidden_size).to(self.device)
        
        inp = torch.tensor([0]).to(self.device) 
        
        log_probs = []
        entropy = 0
        generated_cfg = []
        is_linear_context = False
        
        for i in range(self.max_layers):
            logits, h, c = self.controller(inp, h, c)
            probs = F.softmax(logits, dim=-1)
            m = Categorical(probs)
            
            action = m.sample()
            log_prob = m.log_prob(action)
            
            log_probs.append(log_prob)
            entropy += m.entropy().mean()
            
            token_str = self.vocab[action.item()]
            
            # Early stopping if "stop" token is selected
            if token_str == "stop":
                break
                
            if token_str == "flatten" or token_str == "avgpool" or token_str.startswith("linear"):
                is_linear_context = True
                
            cfg = self._token_to_cfg(token_str, is_linear_context)
            if cfg is not None:
                generated_cfg.append(cfg)
                
            inp = action

        # Force a valid termination block
        if not any(isinstance(layer, LinearCfg) for layer in generated_cfg):
            generated_cfg.append(FlattenCfg())
            generated_cfg.append(LinearCfg(in_features=0, out_features=2, activation=None))

        return generated_cfg, torch.cat(log_probs).sum(), entropy

    def run(self, n_iterations: int) -> Tuple[List[Any], Dict[str, Any]]:
        best_gen = -1
        
        initial_score = self.evaluate(self.layers)
        if initial_score == -float('inf'):
            initial_score = 0.0 
        self.baseline = initial_score
        
        regression = (initial_score <= 0.0)
        crash_score = -1000.0 if regression else 0.0

        batch_size = 16 

        for i in range(n_iterations):
            self.controller.train()
            self.ctrl_optimizer.zero_grad()
            
            batch_loss = 0
            
            for _ in range(batch_size):
                arch_cfg, log_prob, entropy = self.generate_architecture()
                
                raw_score = self.evaluate(arch_cfg)
                
                if raw_score == -float('inf'):
                    reward = crash_score
                else:
                    # Multi-objective penalty: Subtract a penalty based on network depth
                    # Adjust the 0.5 factor based on the importance of network size
                    length_penalty = 0.5 * len(arch_cfg) if not regression else 0.0
                    reward = raw_score - length_penalty
                
                advantage = reward - self.baseline
                
                # Add entropy to the loss function to encourage exploration
                batch_loss += (-log_prob * advantage) - (self.entropy_weight * entropy)
                
                self.baseline = 0.95 * self.baseline + 0.05 * reward

                # Save based on the true score (without depth penalty)
                if raw_score > self.best_score and raw_score != crash_score:
                    self.best_score = raw_score
                    self.best_arch = copy.deepcopy(arch_cfg)
                    best_gen = i
                    print(f"RL Iter {i}: New Best Score {self.best_score:.2f} (Depth: {len(arch_cfg)})")
            
            batch_loss = batch_loss / batch_size
            batch_loss.backward()
            self.ctrl_optimizer.step()

        stats = {
            "initial_score": initial_score,
            "best_score": self.best_score if self.best_score != crash_score else initial_score,
            "best_iter": best_gen,
            "gain": (self.best_score - initial_score) if self.best_score != crash_score else 0.0
        }
        return self.best_arch, stats


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for Transformer input."""
    def __init__(self, d_model: int, max_len: int = 50):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return x


class ControllerTransformer(nn.Module):
    """Autoregressive Transformer controller used for Reinforcement Learning NAS."""
    def __init__(self, num_tokens: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, max_len: int = 20):
        super().__init__()
        self.num_tokens = num_tokens
        self.d_model = d_model
        
        self.embedding = nn.Embedding(num_tokens, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model*2, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.decoder = nn.Linear(d_model, num_tokens)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generates an upper-triangular matrix to prevent attending to future tokens."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        seq_len = src.size(0)
        mask = self.generate_square_subsequent_mask(seq_len).to(src.device)
        
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        output = self.transformer(src, mask=mask)
        logits = self.decoder(output)
        return logits


class TransformerOptimizer(Optimizer):
    """
    Reinforcement Learning Optimizer using a Transformer Controller.
    Employs dynamic entropy adjustment to prevent premature convergence.
    """
    def __init__(self, layers=None, search_space=None, dataset=None, max_layers=8, entropy_weight=0.05, entropy_fct=None, d_model=64, nhead=4, num_layers=2, lr=0.01, **kwargs):
        super().__init__(layers, search_space, dataset)
        self.max_layers = max_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.vocab = [
            "conv_3_16", "conv_3_32", "conv_5_16", 
            "resblock_16", "resblock_32",
            "pool_2", 
            "linear_32", "linear_64", 
            "dropout_0.2", "dropout_0.5",
            "bn2d", "bn1d", "flatten", "avgpool",
            "stop"
        ]
        self.num_tokens = len(self.vocab)
        
        self.controller = ControllerTransformer(self.num_tokens, d_model=d_model, nhead=nhead, num_layers=num_layers, max_len=max_layers+5).to(self.device)
        self.ctrl_optimizer = optim.Adam(self.controller.parameters(), lr=lr)
        self.baseline = 0.0
        self.entropy_weight = entropy_weight 
        
        if entropy_fct is True or entropy_fct == "default":
            self.entropy_fct = self.variable_entropy
        else:
            self.entropy_fct = entropy_fct

    def variable_entropy(self, current_weight: float, iters_without_improvement: int, patience: int = 5, base: float = 0.05, max_w: float = 0.5) -> float:
        """Dynamically scales the entropy weight to force exploration if search stagnates."""
        if iters_without_improvement == 0:
            return base
        if iters_without_improvement >= patience:
            return min(max_w, current_weight * 1.5)
        return current_weight

    def _token_to_cfg(self, token: str, is_linear_context: bool) -> Optional[Any]:
        if token == "conv_3_16" and not is_linear_context:
            return Conv2dCfg(in_channels=0, out_channels=16, kernel_size=3, padding=1, activation=nn.ReLU)
        elif token == "conv_3_32" and not is_linear_context:
            return Conv2dCfg(in_channels=0, out_channels=32, kernel_size=3, padding=1, activation=nn.ReLU)
        elif token == "conv_5_16" and not is_linear_context:
            return Conv2dCfg(in_channels=0, out_channels=16, kernel_size=5, padding=2, activation=nn.ReLU)
        elif token == "resblock_16" and not is_linear_context:
            sub_layers = [
                Conv2dCfg(in_channels=0, out_channels=16, kernel_size=3, padding=1, activation=nn.ReLU),
                BatchNorm2dCfg(num_features=16),
                Conv2dCfg(in_channels=0, out_channels=16, kernel_size=3, padding=1, activation=None)
            ]
            return ResBlockCfg(sub_layers=sub_layers, use_projection=True)
        elif token == "resblock_32" and not is_linear_context:
            sub_layers = [
                Conv2dCfg(in_channels=0, out_channels=32, kernel_size=3, padding=1, activation=nn.ReLU),
                BatchNorm2dCfg(num_features=32),
                Conv2dCfg(in_channels=0, out_channels=32, kernel_size=3, padding=1, activation=None)
            ]
            return ResBlockCfg(sub_layers=sub_layers, use_projection=True)
        elif token == "pool_2" and not is_linear_context:
            return MaxPool2dCfg(kernel_size=2, stride=2, padding=0)
        elif token == "bn2d" and not is_linear_context:
            return BatchNorm2dCfg(num_features=0)
        elif token == "avgpool" and not is_linear_context:
            return GlobalAvgPoolCfg()
        elif token == "flatten":
            return FlattenCfg()
        elif token == "linear_32":
            return LinearCfg(in_features=0, out_features=32, activation=nn.ReLU)
        elif token == "linear_64":
            return LinearCfg(in_features=0, out_features=64, activation=nn.ReLU)
        elif token == "dropout_0.2":
            return DropoutCfg(p=0.2)
        elif token == "dropout_0.5":
            return DropoutCfg(p=0.5)
        elif token == "bn1d" and is_linear_context:
            return BatchNorm1dCfg(num_features=0)
        return None

    def generate_architecture(self) -> Tuple[List[Any], torch.Tensor, torch.Tensor]:
        sequence = torch.tensor([[0]], dtype=torch.long).to(self.device)
        
        log_probs = []
        entropy_total = 0
        generated_cfg = []
        is_linear_context = False
        
        for i in range(self.max_layers):
            logits = self.controller(sequence)
            
            next_token_logits = logits[-1, 0, :]
            
            probs = F.softmax(next_token_logits, dim=-1)
            m = Categorical(probs)
            
            action = m.sample()
            log_prob = m.log_prob(action)
            
            log_probs.append(log_prob)
            entropy_total += m.entropy()
            
            token_str = self.vocab[action.item()]
            
            if token_str == "stop":
                break
                
            if token_str == "flatten" or token_str == "avgpool" or token_str.startswith("linear"):
                is_linear_context = True
                
            cfg = self._token_to_cfg(token_str, is_linear_context)
            if cfg is not None:
                generated_cfg.append(cfg)
                
            action_tensor = action.unsqueeze(0).unsqueeze(0)
            sequence = torch.cat([sequence, action_tensor], dim=0)
            
        if not any(isinstance(layer, LinearCfg) for layer in generated_cfg):
            generated_cfg.append(FlattenCfg())
        
        # Inject the mandatory final linear layer required for classification/regression
        generated_cfg.append(LinearCfg(in_features=0, out_features=self.out_features, activation=None))
        
        return generated_cfg, torch.stack(log_probs).sum(), entropy_total

    def run(self, n_iterations: int) -> Tuple[List[Any], Dict[str, Any]]:
        best_gen = -1
        
        initial_score = self.evaluate(self.layers)
        if initial_score == -float('inf'):
            initial_score = 0.0 
        self.baseline = initial_score
        
        regression = (initial_score <= 0.0)
        crash_score = -1000.0 if regression else 0.0

        batch_size = 16
        iters_without_improvement = 0

        for i in range(n_iterations):
            self.controller.train()
            self.ctrl_optimizer.zero_grad()
            
            batch_loss = 0
            improved_this_iter = False
            
            for _ in range(batch_size):
                arch_cfg, log_prob, entropy = self.generate_architecture()
                
                raw_score = self.evaluate(arch_cfg)
                
                if raw_score == -float('inf'):
                    reward = crash_score
                else:
                    length_penalty = 0.5 * len(arch_cfg) if not regression else 0.0
                    reward = raw_score - length_penalty
                
                advantage = reward - self.baseline
                
                batch_loss += (-log_prob * advantage) - (self.entropy_weight * entropy)
                
                self.baseline = 0.95 * self.baseline + 0.05 * reward

                if raw_score > self.best_score and raw_score != crash_score:
                    self.best_score = raw_score
                    self.best_arch = copy.deepcopy(arch_cfg)
                    best_gen = i
                    improved_this_iter = True
                    print(f"Transformer Iter {i}: New Best Score {self.best_score:.2f} (Depth: {len(arch_cfg)})")
            
            if improved_this_iter:
                iters_without_improvement = 0
            else:
                iters_without_improvement += 1
                
            if self.entropy_fct is not None:
                self.entropy_weight = self.entropy_fct(self.entropy_weight, iters_without_improvement)
            
            batch_loss = batch_loss / batch_size
            batch_loss.backward()
            self.ctrl_optimizer.step()

        stats = {
            "initial_score": initial_score,
            "best_score": self.best_score if self.best_score != crash_score else initial_score,
            "best_iter": best_gen,
            "gain": (self.best_score - initial_score) if self.best_score != crash_score else 0.0
        }
        return self.best_arch, stats