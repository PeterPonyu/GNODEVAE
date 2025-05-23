import math, os
from time import time
import numpy as np
import pandas as pd
import collections
import joblib
import h5py
import scanpy as sc
import scanpy.external as sce
import scipy.sparse as sp

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import PCA, FastICA, NMF, FactorAnalysis, TruncatedSVD
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, silhouette_samples
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import minmax_scale
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel

import scvi
from vgae import GNODEVAE_agent_r, agent_r
from iVAE.agent import agent
from scDHMap.scDHMap import scDHMap
from scDHMap.preprocess import read_dataset, normalize, pearson_residuals
from scDHMap.single_cell_tools import *
from scDHMap.embedding_quality_score import get_quality_metrics, get_scalars
import sctour as sct

# idxs = ['PCA', 'ICA', 'FA', 'SVD', 'NMF', 'Diffmap', 'Palantir', 'Phate', 'beta-VAE', 'DIP-VAE', 'TC-VAE', 'Info-VAE', 'scVI', 'scTour', 'scDHMAP', 'GNODEVAE']
names = ["lsklk", "irall"]
# "lsklk", "irall", "muscle", "dentate", "endo", "paul", "pbmc", "setty", "spinoids", "lung", "neurons", "hemato", "endo_GSE84133", "bm_GSE120446", "ifnHSPC_GSE226824"
nc = 10
for r in range(4,5):
    for n in names:
        adata = sc.read_h5ad(f'data/{n}.h5ad')
        if not sp.issparse(adata.X):  
            adata.X = sp.csr_matrix(adata.X)
        adata.X = adata.X.astype(np.float32)
        adata.layers['counts'] = adata.X.copy()
        adata.raw = adata.copy()
        sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata)
        sc.pp.pca(adata, mask_var='highly_variable')
        adata1 = adata[:, adata.var['highly_variable']].copy()
        sc.pp.neighbors(adata1)
        sc.tl.leiden(adata1, resolution=.6)
        n_parts = len(adata1.obs['leiden'].cat.categories)
        X = adata1.to_df().values
        
        X_pca = adata.obsm['X_pca']
        
        # ica = FastICA(n_components=nc, random_state=42)
        # X_ica = ica.fit_transform(X)
        
        # nmf = NMF(n_components=nc, random_state=42)
        # X_nmf = nmf.fit_transform(X)
        
        # fa = FactorAnalysis(n_components=nc, random_state=42)
        # X_fa = fa.fit_transform(X)
        
        # svd = TruncatedSVD(n_components=nc, random_state=42)
        # X_svd = svd.fit_transform(X)
        
        # sc.tl.diffmap(adata1)
        # X_diffmap = adata1.obsm['X_diffmap']
        
        # sce.tl.palantir(adata1, n_components=nc, impute_data=False)
        
        # X_palantir = adata1.obsm['X_palantir_diff_comp']
        
        # sce.tl.phate(adata1, n_components=nc, n_pca=30)
        # X_phate = adata1.obsm['X_phate']
    
        a_beta = agent(adata1, irecon=False, beta=.1).fit(4000)
        X_betavae = a_beta.get_latent()
        
        a_dip = agent(adata1, irecon=False, beta=.1, dip=1.0).fit(4000)
        X_dip = a_dip.get_latent()
        
        a_tc = agent(adata1, irecon=False, beta=.1, tc=1.0).fit(4000)
        X_tc = a_tc.get_latent()
        
        a_info = agent(adata1, irecon=False, beta=.1, info=1.0).fit(4000)
        X_info = a_info.get_latent()
    
        scvi.model.SCVI.setup_anndata(adata1, layer='counts')
        model = scvi.model.SCVI(adata1, n_layers=2)
        model.train(batch_size=256)
        X_scvi = model.get_latent_representation()
        
        adata1.X = adata1.layers['counts']
        tnode = sct.train.Trainer(adata1, loss_mode='nb', n_latent=10, alpha_recon_lec=0.5, alpha_recon_lode=0.5)
        tnode.train()
        X_sctour, zs, pred_zs = tnode.get_latentsp(alpha_z=0.5, alpha_predz=0.5)
        
        # a_gnodevae = GNODEVAE_agent_r(adata1, num_parts=n_parts).fit(500)
        # X_gnodevae = a_gnodevae.get_mix_latent()
    
        # class Args(object):
        #     def __init__(self):
        #         self.batch_size = 128
        #         self.pretrain_iter = 400
        #         self.maxiter = 50
        #         self.minimum_iter = 0
        #         self.patience = 150
        #         self.lr = 1e-3
        #         self.alpha = 1000.
        #         self.beta = 10.
        #         self.prob = 0
        #         self.perplexity = [30.]
        #         self.save_dir = 'ES_model/'
        #         self.device = 'cuda'     
        # args = Args()
        # adata1.X = adata1.layers['counts']
        # adata1 = normalize(adata1,
        #                   size_factors=True,
        #                   normalize_input=True,
        #                   logtrans_input=True)
        # model = scDHMap(input_dim=adata1.shape[1], encodeLayer=[128, 64, 32, 16], decodeLayer=[16, 32, 64, 128], 
        #         batch_size=args.batch_size, activation="elu", z_dim=10, alpha=args.alpha, beta=args.beta, 
        #         perplexity=args.perplexity, prob=args.prob, device=args.device).to(args.device)
        # model.pretrain_autoencoder(adata1.X.astype(np.float32), adata1.raw.X.A.astype(np.float32), adata1.obs.size_factors.astype(np.float32), 
        #     lr=args.lr, pretrain_iter=args.pretrain_iter,)
        # model.train_model(adata1.X.astype(np.float32), adata1.raw.X.A.astype(np.float32), adata1.obs.size_factors.astype(np.float32), X_pca.astype(np.float32),
        #                 lr=args.lr, maxiter=args.maxiter, minimum_iter=args.minimum_iter,patience=args.patience,save_dir=args.save_dir)
        # X_scdhmap = model.encodeBatch(torch.tensor(adata1.X).float().to(args.device))
        # X_pca,X_ica,X_nmf,X_fa,X_svd,X_diffmap,X_palantir,X_phate,
        arrays = [X_betavae, X_dip, X_tc, X_info, X_scvi, X_sctour]
        joblib.dump(arrays, f'compares/repeat{r}/{n}_arrays')
    
        # ls = []
        # for x in arrays:
        #     x = x[:,:nc]
        #     lab = KMeans(nc).fit_predict(x)
        #     ls.append([normalized_mutual_info_score(adata1.obs['leiden'], lab),
        #                adjusted_rand_score(adata1.obs['leiden'], lab),
        #                silhouette_score(x, lab),
        #                calinski_harabasz_score(x, lab),
        #                davies_bouldin_score(x, lab)])
        # df = pd.DataFrame(ls, columns=['nmi','ari','sil','cal','dav'], index=idxs)
        # tmp = df.values.copy()
        # tmp[: ,-1] *= -1
        # df['overall'] = minmax_scale(tmp, axis=0).mean(axis=1)
        # df.to_csv(f'compares/repeat{r}/{n}_cluster.csv')
    
        # ls = []
        # for x in arrays:
        #     x = x[:,:nc]
        #     df_score = get_quality_metrics(X_pca, x, distance='P')
        #     res = get_scalars(df_score['Qnx'].values)
        #     x_cs = cosine_similarity(x)
        #     x_rbf = rbf_kernel(x)
        #     tmp = []
        #     for L in adata1.obs['leiden'].cat.categories:
        #         mask = adata1.obs['leiden'].values == L
        #         tmp.append((x_cs[mask, :][:, mask].mean(),x_rbf[mask, :][:, mask].mean()))
        #     ls.append(list(res)+list(np.array(tmp).mean(axis=0)))
        # df = pd.DataFrame(ls, columns=['Qlocal','Qglobal','Kmax','Cosine','RBF'], index=idxs)
        # tmp = df.values.copy()
        # df['overall'] = minmax_scale(tmp[:,[0,1,3,4]], axis=0).mean(axis=1)
        # df.to_csv(f'compares/repeat{r}/{n}_embed.csv')










