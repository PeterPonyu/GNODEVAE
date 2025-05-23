from .GODEVAE_env import GNODEVAE_Env_r
from anndata import AnnData  
import numpy as np  
import pandas as pd
import torch  
import tqdm  
from typing import Optional  
import time 
import psutil

class GNODEVAE_agent_r(GNODEVAE_Env_r):  
    def __init__(  
        self,  
        adata: AnnData,  
        layer: str = 'counts',
        n_var: Optional[int] = None,
        tech: str = 'PCA',
        n_neighbors: int = 15,
        batch_tech: Optional[str] = None,
        all_feat: bool = False,
        hidden_dim: int = 128,  
        latent_dim: int = 10,  
        ode_hidden_dim: int = 25,
        encoder_type: str = 'graph',  
        graph_type: str = 'GAT',  
        structure_decoder_type: str = 'mlp',  
        feature_decoder_type: str = 'linear',  
        hidden_layers: int = 2,  
        decoder_hidden_dim: int = 128,  
        dropout: float = 0.05,  
        use_residual: bool = True,  
        Cheb_k: int = 1,  
        alpha: float = 0.5,
        threshold: float = 0,
        sparse_threshold: Optional[int] = None,  
        lr: float = 1e-4,  
        beta: float = 1.0,  
        graph: float = 1.0,  
        device: torch.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),  
        num_parts: int = 10,  
        scale1: float = .5,
        scale2: float = .5,
        *args,  
        **kwargs,  
    ):  
        super().__init__(  
            adata=adata,  
            layer=layer, 
            n_var=n_var,
            tech=tech,
            n_neighbors=n_neighbors,
            batch_tech=batch_tech,
            all_feat=all_feat,
            hidden_dim=hidden_dim,  
            latent_dim=latent_dim,
            ode_hidden_dim=ode_hidden_dim,
            encoder_type=encoder_type,
            graph_type=graph_type,
            structure_decoder_type=structure_decoder_type,
            feature_decoder_type=feature_decoder_type,
            hidden_layers=hidden_layers,
            decoder_hidden_dim=decoder_hidden_dim,
            dropout=dropout,
            use_residual=use_residual,
            Cheb_k=Cheb_k,
            alpha=alpha,
            threshold=threshold,
            sparse_threshold=sparse_threshold,
            lr=lr,
            beta=beta,  
            graph=graph, 
            device=device,  
            num_parts=num_parts,   
        )  
        self.scale1 = scale1
        self.scale2 = scale2

    def fit(  
            self,  
            epochs: int = 300,  
            update_steps: int = 10,  
            silent: bool = False,  
        ) -> "agent":  
        
        self.resource = []  
        start_time = time.time()  
    
        try:  
            if not silent:  
                progress_bar = tqdm.tqdm(total=epochs, desc='Fitting', ncols=200)  
            else:  
                progress_bar = None  
    
            for i in range(epochs):  
                step_start_time = time.time()  
    
                self.step_ode()  
      
                step_end_time = time.time()  
                step_time = step_end_time - step_start_time  
                  
                process = psutil.Process()  
                cpu_memory = process.memory_info().rss / (1024 ** 2)   
    
                if torch.cuda.is_available():  
                    gpu_memory = torch.cuda.memory_allocated(self.device) / (1024 ** 2)   
                else:  
                    gpu_memory = 0.0  
    
                self.resource.append((step_time, cpu_memory, gpu_memory))  
    
                if (i + 1) % update_steps == 0:  
                    recent_losses = self.loss[-update_steps:] if len(self.loss) >= update_steps else self.loss  
                    recent_scores = self.mix_score[-update_steps:] if len(self.mix_score) >= update_steps else self.mix_score  
                    recent_resources = self.resource[-update_steps:] if len(self.resource) >= update_steps else self.resource  
    
                    loss = np.mean([sum(loss_step) for loss_step in recent_losses])  
                    ari, nmi, asw, ch, db, pc = np.mean(recent_scores, axis=0)  
                    st, cm, gm = np.mean(recent_resources, axis=0)  
    
                    if not silent and progress_bar is not None:  
        
                        progress_bar.set_postfix({  
                            'Loss': f'{loss:.2f}',  
                            'ARI': f'{ari:.2f}',  
                            'NMI': f'{nmi:.2f}',  
                            'ASW': f'{asw:.2f}',  
                            'C_H': f'{ch:.2f}',  
                            'D_B': f'{db:.2f}',  
                            'P_C': f'{pc:.2f}',  
                            'Step Time': f'{st:.2f}s',  
                            'CPU Mem': f'{cm:.0f}MB',  
                            'GPU Mem': f'{gm:.0f}MB',  
                        }, refresh=False)   
                        progress_bar.update(update_steps)
                    else:   
                        pass 
    
        except Exception as e:  
            print(f"{e}")  
            raise e   
    
        end_time = time.time()  
        self.time_all = end_time - start_time  
        return self     

    def get_mix_latent(
        self,
    ) -> np.ndarray:
        latent = self.get_latent()
        odelatent = self.get_odelatent()
        mix_latent = self.scale1 * latent + self.scale2 * odelatent
        return mix_latent
    
    def get_latent(  
        self,  
    ) -> np.ndarray:  
        ls_l = []  
        for cd in self.cdata:  
            latent = self.take_latent(cd)  
            ls_l.append(latent)  
        latent = np.vstack(ls_l)  
        lut = dict(zip(self.idx, latent))  
        latent1 = np.vstack([lut[i] for i in range(self.n_obs)])  
        return latent1

    def get_odelatent(  
        self,  
    ) -> np.ndarray:  
        ls_l = []  
        for cd in self.cdata:  
            latent = self.take_odelatent(cd)  
            ls_l.append(latent)  
        latent = np.vstack(ls_l)  
        lut = dict(zip(self.idx, latent))  
        latent1 = np.vstack([lut[i] for i in range(self.n_obs)])  
        return latent1

    def score_final(
        self,
    ) -> None:
        ls_l = []  
        for cd in self.cdata:  
            latent = self.take_latent(cd)  
            ls_l.append(latent)  
        latent = np.vstack(ls_l)  
        score = self._calc_score(latent)
        self.final_score = score
        return 
        
    def score_odefinal(
        self,
    ) -> None:
        ls_l = []  
        for cd in self.cdata:  
            latent = self.take_odelatent(cd)  
            ls_l.append(latent)  
        latent = np.vstack(ls_l)  
        score = self._calc_score(latent)
        self.ode_final_score = score
        return 

    def score_mixfinal(
        self,
    ) -> None:
        ls_l = []  
        for cd in self.cdata:  
            latent1 = self.take_latent(cd)  
            latent2 = self.take_odelatent(cd) 
            latent = self.scale1 * latent1 + self.scale2 * latent2
            ls_l.append(latent)
        latent = np.vstack(ls_l)  
        score = self._calc_score(latent)
        self.mix_final_score = score
        return 

    def partition_time(
        self,
    ) -> pd.DataFrame:
        idx_ls = []
        t_ls = []
        for cd in self.cdata:
            states = cd.x  
            edge_index = cd.edge_index  
            edge_weight = cd.edge_attr  
            _, _, _, t = self.godevae.encoder(states, edge_index, edge_weight)
            t_ls.append(t)
            idx_ls.append(cd.y.cpu().numpy())
        
        df = pd.DataFrame(index=np.concatenate(idx_ls))
        
        for i, idx in enumerate(idx_ls):
            df.loc[idx, 'c'] = str(i)
            df.loc[idx, 't'] = t_ls[i].detach().cpu().numpy()
        
        df1 = df.loc[df.index.sort_values()]
        return df1

        