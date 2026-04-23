import os
import random
import copy
import json
import types
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report

from layer_classes import Conv2dCfg, MaxPool2dCfg, FlattenCfg, LinearCfg, DropoutCfg
from optimizer import ABCOptimizer, TransformerOptimizer, SAOptimizer
from model import DynamicNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def set_global_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate_cifar_proxy(self, genome, train_epochs=4):
    try:
        model = DynamicNet(genome, input_shape=(3, 32, 32))
        model.to(DEVICE)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        best_val_acc = 0.0
        patience = 2
        patience_counter = 0
        
        for epoch in range(train_epochs):
            model.train()
            for inputs, targets in self.dataset:
                inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                # On utilise un test_loader_proxy allégé si besoin, ou on garde le test_loader global
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            current_acc = 100. * correct / total if total > 0 else 0.0
            
            if current_acc > best_val_acc:
                best_val_acc = current_acc
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
                
        return best_val_acc
    except Exception:
        return -float('inf')

if __name__ == "__main__":
        
    print(DEVICE)
    transform_train_proxy = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_train_final = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    full_trainset_proxy = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_proxy)
    full_trainset_final = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train_final)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # Pour l'évaluation rapide, on peut garder num_workers=2 ou 0
    test_loader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    seeds = [42, 43, 44]
    all_accuracies = []
    all_results = {}
    
    os.makedirs("results", exist_ok=True)

    i=0
    ITER=[2, 6,13]

    for current_seed in seeds:
        print(f"\n{'='*50}\nLANCEMENT DE L'EXPÉRIENCE - SEED: {current_seed}\n{'='*50}")
        set_global_seed(current_seed)
        
        proxy_size = int(0.5 * len(full_trainset_proxy))
        indices_proxy = np.random.choice(len(full_trainset_proxy), proxy_size, replace=False)
        proxy_dataset = Subset(full_trainset_proxy, indices_proxy)
        # num_workers=0 est souvent plus rapide sous Windows pour les appels très fréquents
        train_loader_proxy = DataLoader(proxy_dataset, batch_size=512, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
        train_loader_final = DataLoader(full_trainset_final, batch_size=512, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)

        initial_arch = [
            Conv2dCfg(in_channels=0, out_channels=16, kernel_size=3, padding=1, activation=nn.ReLU),
            MaxPool2dCfg(kernel_size=2, stride=2, padding=0),
            FlattenCfg(),
            LinearCfg(in_features=0, out_features=128, activation=nn.ReLU),
            LinearCfg(in_features=0, out_features=10, activation=None)
        ]

        print("Début de la recherche Transformer sur proxy CIFAR-10...")
        start_time_trans = time.time()
        opt_trans = TransformerOptimizer(layers=initial_arch, max_layers=20, dataset=train_loader_proxy, entropy_fct="default")
        opt_trans.evaluate = types.MethodType(evaluate_cifar_proxy, opt_trans)
        best_sol_final, stats_trans = opt_trans.run(ITER[i])
        time_trans = time.time() - start_time_trans
        i+=1
        print("\nDébut de la recherche Recuit simulee sur proxy CIFAR-10...")
        start_time_abc = time.time()

        time_abc = time.time() - start_time_abc

        total_search_time = time_trans + time_abc

        print("\nEntraînement final de l'architecture optimale sur 100% de CIFAR-10...")
        start_time_train = time.time()
        
        final_model = DynamicNet(best_sol_final, input_shape=(3, 32, 32)).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(final_model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

        EPOCHS = 100

        for epoch in range(EPOCHS):
            final_model.train()
            running_loss = 0.0
            
            for inputs, targets in train_loader_final:
                inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
                optimizer.zero_grad()
                outputs = final_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
            scheduler.step()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader_final):.4f}")

        time_train = time.time() - start_time_train

        final_model.eval()
        
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.to(DEVICE, non_blocking=True)
                outputs = final_model(inputs)
                _, preds = outputs.max(1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.numpy())

        classes = ['avion', 'auto', 'oiseau', 'chat', 'cerf', 'chien', 'grenouille', 'cheval', 'navire', 'camion']
        report_dict = classification_report(all_targets, all_preds, target_names=classes, output_dict=True)
        final_accuracy = report_dict['accuracy']
        
        print(f"--> Accuracy finale pour la seed {current_seed} : {final_accuracy*100:.2f}%")
        print(f"Temps de recherche NAS : {total_search_time:.2f} s | Temps d'entraînement final : {time_train:.2f} s")
        all_accuracies.append(final_accuracy)

        torch.save(final_model.state_dict(), f"results/best_model_seed_{current_seed}.pth")
        
        all_results[f"seed_{current_seed}"] = {
            "search_times": {
                "transformer_time_s": time_trans,
                "abc_time_s": time_abc,
                "total_search_time_s": total_search_time
            },
            "final_training_time_s": time_train,
            "final_accuracy": final_accuracy,
            "report": report_dict
        }
        with open(f"results/academic_results_{current_seed}.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=4)

    mean_acc = np.mean(all_accuracies) * 100
    std_acc = np.std(all_accuracies) * 100

    print(f"\n{'='*50}\nBILAN GLOBAL DES {len(seeds)} EXÉCUTIONS\n{'='*50}")
    print(f"Accuracy Moyenne : {mean_acc:.2f}% ± {std_acc:.2f}%")

    all_results["global_summary"] = {
        "mean_accuracy": mean_acc,
        "std_accuracy": std_acc,
        "seeds_tested": seeds
    }

    with open("results/academic_results_transf.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    with open("results/academic_summary_transf.txt", "w", encoding="utf-8") as f:
        f.write("Rapport Académique - Expériences NAS Mémétique (CIFAR-10)\n")
        f.write("=========================================================\n\n")
        f.write(f"Nombre d'exécutions indépendantes : {len(seeds)}\n")
        f.write(f"Graines aléatoires utilisées : {seeds}\n\n")
        f.write(f"Résultat final : {mean_acc:.2f}% ± {std_acc:.2f}%\n\n")
        f.write("Détails par seed :\n")
        for idx, seed in enumerate(seeds):
            search_t = all_results[f"seed_{seed}"]["search_times"]["total_search_time_s"]
            train_t = all_results[f"seed_{seed}"]["final_training_time_s"]
            f.write(f" - Seed {seed} : {all_accuracies[idx]*100:.2f}% (Recherche: {search_t/60:.2f} min | Entraînement: {train_t/60:.2f} min)\n")