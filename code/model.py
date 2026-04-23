import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import copy
from layer_classes import Conv2dCfg, DropoutCfg, FlattenCfg, LinearCfg, MaxPool2dCfg, GlobalAvgPoolCfg, BatchNorm1dCfg, BatchNorm2dCfg, ResBlockCfg

FEATURE_SIZE = 15

class ResidualWrapper(nn.Module):
    def __init__(self, sub_layers_module, use_projection=False, in_channels=0, out_channels=0):
        super().__init__()
        self.net = sub_layers_module
        self.use_projection = use_projection
        self.projection = None
        
        if use_projection and in_channels != out_channels:
            self.projection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        elif in_channels != out_channels:
             pass

    def forward(self, x):
        identity = x
        out = self.net(x)
        if self.projection is not None:
            identity = self.projection(identity)
        if identity.shape != out.shape:
            return out 
        return out + identity

class DynamicNet(nn.Module):
    def __init__(self, layers_cfg: list, input_shape: tuple = None):
        super().__init__()
        if input_shape is not None:
            self.layers_cfg = self._reconnect_layers(layers_cfg, input_shape)
        else:
            self.layers_cfg = layers_cfg

        self.net = self._build_sequential(self.layers_cfg)
        
    def _build_sequential(self, cfgs):
        layers = []
        for cfg in cfgs:
            if isinstance(cfg, LinearCfg):
                layers.append(nn.Linear(cfg.in_features, cfg.out_features))
                if cfg.activation: layers.append(cfg.activation())
            elif isinstance(cfg, Conv2dCfg):
                layers.append(nn.Conv2d(cfg.in_channels, cfg.out_channels, cfg.kernel_size, cfg.stride, cfg.padding))
                if cfg.activation: layers.append(cfg.activation())
            elif isinstance(cfg, DropoutCfg):
                layers.append(nn.Dropout(p=cfg.p))
            elif isinstance(cfg, FlattenCfg):
                layers.append(nn.Flatten(start_dim=cfg.start_dim))
            elif isinstance(cfg, MaxPool2dCfg):
                layers.append(nn.MaxPool2d(kernel_size=cfg.kernel_size, stride=cfg.stride, padding=cfg.padding,  ceil_mode=cfg.ceil_mode))
            elif isinstance(cfg, GlobalAvgPoolCfg):
                layers.append(nn.AdaptiveAvgPool2d((1, 1)))
                layers.append(nn.Flatten())
            elif isinstance(cfg, BatchNorm1dCfg):
                layers.append(nn.BatchNorm1d(cfg.num_features))
            elif isinstance(cfg, BatchNorm2dCfg):
                layers.append(nn.BatchNorm2d(cfg.num_features))
            elif isinstance(cfg, ResBlockCfg):
                inner_seq = self._build_sequential(cfg.sub_layers)
                in_ch = 0
                out_ch = 0
                if len(cfg.sub_layers) > 0:
                    first = cfg.sub_layers[0]
                    last = cfg.sub_layers[-1]
                    if hasattr(first, 'in_channels'): in_ch = first.in_channels
                    elif hasattr(first, 'in_features'): in_ch = first.in_features 
                    if hasattr(last, 'out_channels'): out_ch = last.out_channels
                    elif hasattr(last, 'out_features'): out_ch = last.out_features
                wrapper = ResidualWrapper(inner_seq, cfg.use_projection, in_ch, out_ch)
                layers.append(wrapper)
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def flatten_weights(self, to_numpy=True, device=None):
        vec = parameters_to_vector(self.parameters())
        if to_numpy: return vec.detach().cpu().numpy()
        return vec.to(device) if device is not None else vec

    def load_flattened_weights(self, flat_weights):
        if isinstance(flat_weights, np.ndarray):
            flat_weights = torch.as_tensor(flat_weights, dtype=torch.float32)
        device = next(self.parameters()).device
        flat_weights = flat_weights.to(device)
        try:
            vector_to_parameters(flat_weights, self.parameters())
        except RuntimeError:
            pass

    def evaluate_model(self, X, y, loss_fn=nn.MSELoss(), n_warmup=3, n_runs=20, verbose=False):
        model = self.net
        model.eval()
        use_cuda = torch.cuda.is_available()
        device = "cuda" if use_cuda else "cpu"
        if next(model.parameters()).device.type != device: model = model.to(device)
        X, y = X.to(device), y.to(device)

        with torch.no_grad():
            pred = model(X)
            loss_value = loss_fn(pred, y).item()
            for _ in range(n_warmup): _ = model(X)
            if use_cuda: torch.cuda.synchronize()

        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.perf_counter()
                _ = model(X)
                if use_cuda: torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
        
        inference_time = float(np.median(times))
        if verbose: print(f"Loss: {loss_value:.6f} | Inference time: {inference_time*1000:.3f} ms")
        return loss_value, inference_time
    
    def _reconnect_layers(self, layers, input_shape):
        dummy_input = torch.zeros(1, *input_shape)
        def process_recursive(cfg_list, current_tensor):
            processed = []
            x = current_tensor
            for original_cfg in cfg_list:
                cfg = copy.deepcopy(original_cfg)
                try:
                    if isinstance(cfg, Conv2dCfg):
                        cfg.in_channels = x.shape[1]
                        layer = nn.Conv2d(cfg.in_channels, cfg.out_channels, cfg.kernel_size, cfg.stride, cfg.padding)
                        x = layer(x)
                        processed.append(cfg)
                    elif isinstance(cfg, BatchNorm2dCfg):
                        cfg.num_features = x.shape[1]
                        layer = nn.BatchNorm2d(cfg.num_features)
                        x = layer(x)
                        processed.append(cfg)
                    elif isinstance(cfg, LinearCfg):
                        if len(x.shape) > 2:
                            processed.append(FlattenCfg())
                            x = torch.flatten(x, 1)
                        cfg.in_features = x.shape[1]
                        layer = nn.Linear(cfg.in_features, cfg.out_features)
                        x = layer(x)
                        processed.append(cfg)
                    elif isinstance(cfg, ResBlockCfg):
                        inner_cfgs, inner_out = process_recursive(cfg.sub_layers, x)
                        cfg.sub_layers = inner_cfgs
                        x = inner_out
                        processed.append(cfg)
                    elif isinstance(cfg, GlobalAvgPoolCfg):
                        x = nn.AdaptiveAvgPool2d((1, 1))(x)
                        x = torch.flatten(x, 1)
                        processed.append(cfg)
                    elif isinstance(cfg, FlattenCfg):
                        x = torch.flatten(x, cfg.start_dim)
                        processed.append(cfg)
                    else:
                        processed.append(cfg)
                except Exception: pass
            return processed, x
        new_layers, _ = process_recursive(layers, dummy_input)
        return new_layers

    def _encode_config_to_vector(self, cfg, sub_layer_count=0):
        vec = np.zeros(FEATURE_SIZE, dtype=np.float32)
        if isinstance(cfg, Conv2dCfg):
            vec[0] = 1
            vec[8] = cfg.kernel_size / 7.0 if hasattr(cfg, 'kernel_size') else 0
            vec[9] = cfg.stride / 4.0 if hasattr(cfg, 'stride') else 0
            vec[10] = cfg.padding / 4.0 if hasattr(cfg, 'padding') else 0
            vec[14] = 1.0 if cfg.activation is not None else 0.0
        elif isinstance(cfg, LinearCfg):
            vec[1] = 1
            vec[14] = 1.0 if cfg.activation is not None else 0.0
        elif isinstance(cfg, (MaxPool2dCfg, GlobalAvgPoolCfg)):
            vec[2] = 1
            if hasattr(cfg, 'kernel_size'): vec[8] = cfg.kernel_size / 7.0
            if hasattr(cfg, 'stride'): vec[9] = cfg.stride / 4.0
        elif isinstance(cfg, (BatchNorm1dCfg, BatchNorm2dCfg)):
            vec[3] = 1
            if hasattr(cfg, 'num_features'):
                vec[11] = cfg.num_features / 1024.0
                vec[12] = cfg.num_features / 1024.0
        elif isinstance(cfg, DropoutCfg):
            vec[4] = 1
            vec[13] = cfg.p
        elif isinstance(cfg, FlattenCfg):
            vec[5] = 1
        elif isinstance(cfg, ResBlockCfg):
            vec[6] = 1
            vec[14] = 1.0 if cfg.use_projection else 0.0
            vec[8] = float(sub_layer_count)
        else:
            vec[7] = 1

        if hasattr(cfg, 'in_channels'): vec[11] = cfg.in_channels / 1024.0
        elif hasattr(cfg, 'in_features'): vec[11] = cfg.in_features / 1024.0
        if hasattr(cfg, 'out_channels'): vec[12] = cfg.out_channels / 1024.0
        elif hasattr(cfg, 'out_features'): vec[12] = cfg.out_features / 1024.0
        return vec
    
    def get_graph(self):
        """
        Retourne la topologie sous forme de dictionnaire d'adjacence.
        Returns:
            features (np.array): Matrice [N_nodes, FEATURE_SIZE]
            adj_dict (dict): Dictionnaire {node_idx: [voisin1, voisin2, ...]}
        """
        node_counter = 0
        
        def process_block(configs, start_node_idx):
            nonlocal node_counter
            local_features = []
            local_adj = {}
            current_prev_node = start_node_idx
            
            for cfg in configs:
                node_counter += 1
                this_node_idx = node_counter
                
                if current_prev_node not in local_adj: local_adj[current_prev_node] = []
                
                if isinstance(cfg, ResBlockCfg):
                    child_feats, child_adj, last_child_idx = process_block(cfg.sub_layers, this_node_idx)
                    
                    parent_vec = self._encode_config_to_vector(cfg, sub_layer_count=len(cfg.sub_layers))
                    local_features.append(parent_vec)
                    local_features.extend(child_feats)
                    
                    local_adj.update(child_adj)
                    

                    local_adj[current_prev_node].append(this_node_idx)
                    
                    if this_node_idx not in local_adj: local_adj[this_node_idx] = []
                    local_adj[this_node_idx].append(last_child_idx)
                    
                    current_prev_node = last_child_idx
                else:
                    vec = self._encode_config_to_vector(cfg)
                    local_features.append(vec)
                    local_adj[current_prev_node].append(this_node_idx)
                    current_prev_node = this_node_idx
            
            return local_features, local_adj, current_prev_node

        input_vec = np.zeros(FEATURE_SIZE, dtype=np.float32)
        input_vec[7] = 1 
        
        feats, adj_dict, _ = process_block(self.layers_cfg, 0)
        final_features = [input_vec] + feats
        
        return np.array(final_features), adj_dict
    
    @staticmethod
    def to_gnn_format(features, adj_dict):
        """
        Transforme le format dictionnaire en format PyTorch Geometric (edge_index).
        À appeler juste avant d'envoyer au GNN.
        """
        if isinstance(features, np.ndarray):
            x = torch.from_numpy(features).float()
        else:
            x = features.float()
            
        sources = []
        targets = []
        
        for src, neighbors in adj_dict.items():
            for dst in neighbors:
                sources.append(src)
                targets.append(dst)
        
        edge_index = torch.tensor([sources, targets], dtype=torch.long)
        return x, edge_index

    def save_model(self, filename):
        features_x, adj_dict = self.get_graph() 
        weights_w = self.flatten_weights(to_numpy=True)
        

        _, edge_index = DynamicNet.to_gnn_format(features_x, adj_dict)

        np.savez_compressed(
            filename, 
            adj=edge_index.numpy(),
            features=features_x,
            weights=weights_w            
        )
        print(f"Model saved as: {filename}.npz")

    @staticmethod
    def _vector_to_single_config(vec):
        type_idx = np.argmax(vec[0:8])
        k = int(round(vec[8] * 7.0))
        s = int(round(vec[9] * 4.0))
        p = int(round(vec[10] * 4.0))
        c_in = int(round(vec[11] * 1024.0)) if vec[11] > 0 else 0
        c_out = int(round(vec[12] * 1024.0)) if vec[12] > 0 else 0
        prob = float(vec[13])
        has_activation = (vec[14] > 0.5)
        act_fn = nn.ReLU if has_activation else None
        
        if type_idx == 0:
            return Conv2dCfg(in_channels=c_in, out_channels=c_out, kernel_size=k, stride=s, padding=p, activation=act_fn)
        elif type_idx == 1:
            return LinearCfg(in_features=c_in, out_features=c_out, activation=act_fn)
        elif type_idx == 2:
            if k == 0: return GlobalAvgPoolCfg()
            return MaxPool2dCfg(kernel_size=k, stride=s, padding=p)
        elif type_idx == 3:
            return BatchNorm2dCfg(num_features=c_in)
        elif type_idx == 4:
            return DropoutCfg(p=prob)
        elif type_idx == 5:
            return FlattenCfg()
        return None

    @staticmethod
    def decode_matrix(features_x):
        """
        Décodage basé uniquement sur la matrice de features (et l'info count_children).
        Plus besoin de la matrice d'adjacence pour reconstruire l'architecture.
        """
        if isinstance(features_x, torch.Tensor): features_x = features_x.cpu().numpy()
        
        # On utilise une file (queue) pour consommer les lignes
        rows = [features_x[i] for i in range(1, features_x.shape[0])]
        
        def recursive_parser(rows_queue):
            configs = []
            while len(rows_queue) > 0:
                vec = rows_queue[0]
                type_idx = np.argmax(vec[0:8])
                
                if type_idx == 6: # ResBlock
                    rows_queue.pop(0) 
                    count_children = int(vec[8])
                    use_proj = (vec[14] > 0.5)
                    
                    sub_layers = []

                    for _ in range(count_children):

                        child_cfgs = extract_one_object(rows_queue)
                        sub_layers.extend(child_cfgs)
                    
                    configs.append(ResBlockCfg(sub_layers=sub_layers, use_projection=use_proj))
                
                else:
                    rows_queue.pop(0)
                    cfg = DynamicNet._vector_to_single_config(vec)
                    if cfg: configs.append(cfg)
                    
            return configs

        # Helper pour extraire exactement UN "Statement" (soit une couche simple, soit un bloc entier)
        def extract_one_object(rows_queue):
            if len(rows_queue) == 0: return []
            
            vec = rows_queue[0]
            type_idx = np.argmax(vec[0:8])
            
            if type_idx == 6: 
                rows_queue.pop(0)
                count_children = int(vec[8])
                use_proj = (vec[14] > 0.5)
                sub_layers = []
                for _ in range(count_children):
                    sub_layers.extend(extract_one_object(rows_queue))
                return [ResBlockCfg(sub_layers=sub_layers, use_projection=use_proj)]
            else:
                rows_queue.pop(0)
                cfg = DynamicNet._vector_to_single_config(vec)
                return [cfg] if cfg else []

        final_configs = []
        while len(rows) > 0:
            final_configs.extend(extract_one_object(rows))
            
        return final_configs

    @staticmethod
    def load_model(filename):
        data = np.load(f"{filename}.npz")
        X = data['features']
        W = data['weights']  
        
        reconstructed_cfgs = DynamicNet.decode_matrix(X)
        model = DynamicNet(reconstructed_cfgs)
        try:
            model.load_flattened_weights(W)
        except Exception: pass
        return model

        


