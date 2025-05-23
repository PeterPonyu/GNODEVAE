import numpy as np
import pandas as pd
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from torchdiffeq import odeint  
from .mixin import scviMixin

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

class VAE(nn.Module, scviMixin):  
    def __init__(self, input_dim, hidden_dim, latent_dim):  
        super(VAE, self).__init__()  
        # Encoder  
        self.fc1 = nn.Linear(input_dim, hidden_dim)  
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # Mean of latent space  
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # Log variance of latent space  

        # Decoder  
        self.fc2 = nn.Linear(latent_dim, hidden_dim)  
        self.fc3 = nn.Linear(hidden_dim, input_dim)  
        
        self.time_predictor = TimePredictor(latent_dim)

        self.disp = nn.Parameter(torch.randn(input_dim))
        
    def encode(self, x):  
        h = F.relu(self.fc1(x))  
        mu = self.fc_mu(h)  
        logvar = self.fc_logvar(h)  
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar  

    def reparameterize(self, mu, logvar):  
        std = torch.exp(0.5 * logvar)  # Standard deviation  
        eps = torch.randn_like(std)  # Random noise  
        return mu + eps * std  # Reparameterization trick  

    def decode(self, z):  
        h = F.relu(self.fc2(z))  
        return torch.softmax(self.fc3(h), dim=-1)  # Output is in [0, 1]  

    def forward(self, x):  
        z, mu, logvar = self.encode(x)  
        t = self.time_predictor(z)
        recon_x = self.decode(z)  
        return recon_x, z , mu, logvar, t     

    def vae_loss(self, q_m, q_s, x, pred_x):
        p_m = torch.zeros_like(q_m)  
        p_s = torch.zeros_like(q_s)  
        kl_loss = self._normal_kl(q_m, q_s, p_m, p_s).sum(-1).mean()
        # Compute losses  
        l = x.sum(-1).view(-1, 1)  
        pred_x = pred_x * l  
        disp = torch.exp(self.disp)  
        recon_loss = -self._log_nb(x, pred_x, disp).sum(-1).mean() 
        return kl_loss, recon_loss

# Define the Time Predictor model  
class TimePredictor(nn.Module):  
    def __init__(self, latent_dim):  
        super(TimePredictor, self).__init__()  
        self.net = nn.Sequential(  
            nn.Linear(latent_dim, 128),  
            nn.ReLU(),  
            nn.Linear(128, 1),    
            nn.Sigmoid()  # Ensures time values are between 0 and 1  
        )  
    
    def forward(self, z):  
        return self.net(z)  

# Define the ODE function (dynamics model)  
class LatentDynamics(nn.Module):  
    def __init__(self, latent_dim):  
        super(LatentDynamics, self).__init__()  
        self.net = nn.Sequential(  
            nn.Linear(latent_dim, 128),   
            nn.ELU(),  
            nn.Linear(128, latent_dim)  
        )  
    
    def forward(self, t, z):  
        # Note: torchdiffeq expects the function signature to be (t, z)  
        return self.net(z)  


class PartitionHandler:
    def __init__(self, agent, batch, device):
        self.fetch_data(agent, batch)

        # Initialize models  
        self.dynamics_model = LatentDynamics(10).to(device)  
         
        self.vae = VAE(self.df.shape[1]-2, 128, 10).to(device) 
        
        # Initialize optimizer  
        self.optimizer = torch.optim.Adam(  
            self.vae.parameters(),  
            lr=1e-3 
        )
        
        self.device = device

    def fit(self, num_epochs):
        
        x = torch.tensor(self.df.iloc[:,:-2].values, device=self.device)
        
        for epoch in range(num_epochs):  

            recon_x, z, q_m, q_s, t = self.vae(x)  
            kl_loss, recon_loss = self.vae.vae_loss(q_m, q_s, x, recon_x) 
            
            t = t.ravel()
            idx = torch.argsort(t)   
            t_sort, z_sort = t[idx], z[idx]    

            # Keep only unique times (remove duplicates)  
            mask = torch.cat([  
                (t_sort[:-1] != t_sort[1:]),  # Find where adjacent times differ  
                torch.tensor([True], device=t.device)  # Always keep last point  
            ])  
            t_uniq, z_uniq = t_sort[mask], z_sort[mask]
            
            try:  
                pred_z = odeint(  
                    self.dynamics_model,   
                    z_uniq[0],   
                    t_uniq,  
                    method='dopri5',  
                    options={'atol': 1e-7, 'rtol': 1e-5}  
                )  
            except:  
                # If adaptive solver fails, try with a simpler fixed-step method  
                pred_z = odeint(  
                    self.dynamics_model,   
                    z_uniq[0],   
                    t_uniq,  
                    method='rk4'  
                )  
            pred_x_ode = self.vae.decode(pred_z)
            
            l = x.sum(-1).view(-1, 1)[idx][mask]  
            pred_x_ode = pred_x_ode * l  
            disp = torch.exp(self.vae.disp)  
            
            recon_loss_ode = -self.vae._log_nb(x[idx][mask], pred_x_ode, disp).sum(-1).mean()
            z_div = F.mse_loss(z_uniq, pred_z, reduction="none").sum(-1).mean()

            loss = recon_loss + recon_loss_ode + kl_loss + z_div
            
            self.optimizer.zero_grad()  
            loss.backward()  
            self.optimizer.step()
            if (epoch+1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")
        return self

    def predict(self, ):
        with torch.no_grad():
            res = self.vae(torch.tensor(self.df.iloc[:,:-2].values, device=self.device))
            self.df['time'] = res[-1].cpu().numpy()
            
    def fetch_data(self, agent, batch=256):
        size = int(batch / len(agent.cdata))
        df_ls = []
        idx_dict = dict()
        for i, cd in enumerate(agent.cdata):
            idx_dict.update(dict(zip(cd.y.cpu().numpy(), np.repeat(i, len(cd.y)))))
            idx = np.random.choice(len(cd.y), size=size, replace=False)
            df = pd.DataFrame(cd.x.cpu().numpy()[idx])
            df['idx'] = cd.y.cpu().numpy()[idx]
            df['cat'] = i
            df_ls.append(df)
        self.idx_dict = idx_dict    
        self.df = pd.concat(df_ls)







        