import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from layer_classes import FlattenCfg, LinearCfg
from optimizer import ABCOptimizer, TransformerOptimizer
from model import DynamicNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("data/creditcard.csv")

y = df['Class']
X = df.drop(columns=['Class', 'Time'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train_full, X_test, y_train_full, y_test = train_test_split(X_scaled, y.values, test_size=0.2, random_state=42, stratify=y.values)

X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=256, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=256)
test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=256)

opt_trans = TransformerOptimizer(max_layers=50, dataset=train_loader, entropy_fct="default")
best_arch_trans, stats_trans = opt_trans.run(2)

opt_abc = ABCOptimizer(layers=best_arch_trans, dataset=train_loader)
best_sol_final, optim_stats_abc = opt_abc.run(20)
#%%
num_negatives = (y_train == 0).sum()
num_positives = (y_train == 1).sum()
pos_weight_value = torch.tensor([num_negatives / num_positives], dtype=torch.float32).to(DEVICE)

final_model = DynamicNet(best_sol_final, input_shape=( 29,)).to(DEVICE)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_value)
optimizer = optim.Adam(final_model.parameters(), lr=0.001)

EPOCHS = 1

for epoch in range(EPOCHS):
    final_model.train()
    running_loss = 0.0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = final_model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    final_model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            outputs = final_model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
            
    print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {running_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
#%%
from sklearn.metrics import f1_score
final_model.eval()
all_probs = []
all_targets = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        outputs = final_model(X_batch)
        
        probs = torch.sigmoid(outputs)
        
        all_probs.extend(probs.cpu().numpy())
        all_targets.extend(y_batch.numpy())

all_probs = np.array(all_probs)
all_targets = np.array(all_targets)

best_f1 = -1
best_threshold = 0.80

for thresh in np.arange(0.80, 1, 0.01):
    preds = (all_probs > thresh).astype(float)
    f1 = f1_score(all_targets, preds)
    print(f"Threshold: {thresh:.2f} | F1-Score (Fraud): {f1:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = thresh

print(f"\n--- MEILLEUR SEUIL TROUVÉ : {best_threshold:.2f} ---")
best_preds = (all_probs > best_threshold).astype(float)
print(classification_report(all_targets, best_preds, target_names=["Normal", "Fraud"]))