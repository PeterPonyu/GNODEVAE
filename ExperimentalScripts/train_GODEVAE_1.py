import argparse  
from anndata import AnnData  
import torch  
import sys  
import scanpy as sc  
import joblib  
import os  
from typing import Optional  
from vgae import GNODEVAE_agent_r

def str2bool(v):  
    if isinstance(v, bool):  
        return v  
    if v.lower() in ('yes', 'true', 't', 'y', '1'):  
        return True  
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):  
        return False  
    else:  
        raise argparse.ArgumentTypeError('Boolean value expected.')  

def main():  
    parser = argparse.ArgumentParser(  
        description="Run the GNODEVAE agent with specified parameters."  
    )  

    # 数据参数  
    parser.add_argument("--data_path", type=str, required=True,   
                      help="Path to the AnnData object (.h5ad file).")  
    parser.add_argument("--layer", type=str, default="counts",  
                      help="Data layer to use (default: counts).")  
    parser.add_argument("--n_var", type=int, default=None,  
                      help="Number of highly variable genes to select.")  
    parser.add_argument("--tech", type=str, default="PCA",  
                      help="Dimensionality reduction technique (default: PCA).")  
    
    # 图网络参数  
    parser.add_argument("--n_neighbors", type=int, default=15,  
                      help="Number of neighbors for graph construction.")  
    parser.add_argument("--batch_tech", type=str, default=None,  
                      help="Batch correction method (None for no correction).")  
    parser.add_argument("--all_feat", type=str2bool, default=False,  
                      help="Use all features (True) or HVGs only (False).")  
    
    # 模型架构参数  
    parser.add_argument("--hidden_dim", type=int, default=128,  
                      help="Hidden dimension size (default: 128).")  
    parser.add_argument("--latent_dim", type=int, default=10,  
                      help="Latent space dimension (default: 10).")  
    parser.add_argument("--ode_hidden_dim", type=int, default=25,  
                      help="ODE network hidden dimension (default: 25).")  
    parser.add_argument("--encoder_type", type=str, default="graph",  
                      choices=['graph', 'linear'],   
                      help="Encoder type (default: graph).")  
    parser.add_argument("--graph_type", type=str, default="GAT",  
                      help="GNN layer type (default: GAT).")  
    parser.add_argument("--structure_decoder_type", type=str, default="mlp",  
                      help="Graph decoder type (default: mlp).")  
    parser.add_argument("--feature_decoder_type", type=str, default="linear",  
                      help="Feature decoder type (default: linear).")  
    
    # 训练参数  
    parser.add_argument("--hidden_layers", type=int, default=2,  
                      help="Number of hidden layers (default: 2).")  
    parser.add_argument("--decoder_hidden_dim", type=int, default=128,  
                      help="Decoder hidden dimension (default: 128).")  
    parser.add_argument("--dropout", type=float, default=0.05,  
                      help="Dropout rate (default: 0.05).")  
    parser.add_argument("--use_residual", type=str2bool, default=True,  
                      help="Use residual connections (default: True).")  
    parser.add_argument("--Cheb_k", type=int, default=1,  
                      help="Chebyshev polynomial order (default: 1).")  
    parser.add_argument("--alpha", type=float, default=0.5,  
                      help="Teleport probability (default: 0.5).")  
    parser.add_argument("--threshold", type=float, default=0,  
                      help="Sparsity threshold (default: 0).")  
    parser.add_argument("--sparse_threshold", type=int, default=None,  
                      help="Edge sparsity threshold (default: None).")  
    
    # 优化参数  
    parser.add_argument("--lr", type=float, default=1e-4,  
                      help="Learning rate (default: 1e-4).")  
    parser.add_argument("--beta", type=float, default=1.0,  
                      help="KL divergence weight (default: 1.0).")  
    parser.add_argument("--graph", type=float, default=1.0,  
                      help="Graph reconstruction weight (default: 1.0).") 
    parser.add_argument("--scale1", type=float, default=0.5,  
                      help="Space scale weight (default: 0.5).")
    parser.add_argument("--scale2", type=float, default=0.5,  
                      help="Space scale weight (default: 0.5).")
    
    # 系统参数  
    parser.add_argument("--device", type=str,   
                      default="cuda" if torch.cuda.is_available() else "cpu",  
                      help="Compute device (default: cuda if available).")  
    parser.add_argument("--num_parts", type=int, default=10,  
                      help="Number of graph partitions (default: 10).")  
    
    # 输出参数  
    parser.add_argument("--output_dir", type=str, default="outputs",  
                      help="Output directory (default: outputs).")  
    parser.add_argument("--run_number", type=int, default=1,  
                      help="Experiment run number (default: 1).")  
    parser.add_argument("--epochs", type=int, default=300,  
                      help="Training epochs (default: 300).")  
    parser.add_argument("--update_steps", type=int, default=10,  
                      help="Progress update interval (default: 10).")  
    parser.add_argument("--silent", type=str2bool, default=False,  
                      help="Disable progress bar (default: False).")  

    args = parser.parse_args()  

    # 加载数据  
    try:  
        adata = sc.read_h5ad(args.data_path)  
        print(f"Loaded data with {adata.n_obs} cells and {adata.n_vars} genes")  
    except Exception as e:  
        print(f"Error loading data: {e}")  
        sys.exit(1)  

    # 初始化模型  
    try:  
        model = GNODEVAE_agent_r(  
            adata=adata,  
            layer=args.layer,  
            n_var=args.n_var,  
            tech=args.tech,  
            n_neighbors=args.n_neighbors,  
            batch_tech=args.batch_tech,  
            all_feat=args.all_feat,  
            hidden_dim=args.hidden_dim,  
            latent_dim=args.latent_dim,  
            ode_hidden_dim=args.ode_hidden_dim,  
            encoder_type=args.encoder_type,  
            graph_type=args.graph_type,  
            structure_decoder_type=args.structure_decoder_type,  
            feature_decoder_type=args.feature_decoder_type,  
            hidden_layers=args.hidden_layers,  
            decoder_hidden_dim=args.decoder_hidden_dim,  
            dropout=args.dropout,  
            use_residual=args.use_residual,  
            Cheb_k=args.Cheb_k,  
            alpha=args.alpha,  
            threshold=args.threshold,  
            sparse_threshold=args.sparse_threshold,  
            lr=args.lr,  
            beta=args.beta,  
            graph=args.graph,  
            device=torch.device(args.device),  
            num_parts=args.num_parts,  
            scale1=args.scale1,
            scale2=args.scale2
        )  
        print("Model initialized successfully")  
    except Exception as e:  
        print(f"Error initializing model: {e}")  
        sys.exit(1)  

    # 创建输出目录  
    os.makedirs(args.output_dir, exist_ok=True)  

    # 训练模型  
    try:  
        print(f"Starting training for {args.epochs} epochs")  
        model.fit(  
            epochs=args.epochs,  
            update_steps=args.update_steps,  
            silent=args.silent  
        )  
    except Exception as e:  
        print(f"Training failed: {e}")  
        sys.exit(1)  

    # 评估模型  
    try:  
        print("Evaluating final model...")  
        model.score_final()  
        model.score_odefinal() 
        model.score_mixfinal()
    except Exception as e:  
        print(f"Evaluation failed: {e}")  
        sys.exit(1)  

    # 保存结果  
    try:  
        filename = f"GNODEVAE_{args.graph_type}_{args.encoder_type}_run{args.run_number}.pkl"  
        filepath = os.path.join(args.output_dir, filename)  
        res = [  
            model.loss, 
            model.mix_score,
            model.resource,  
            model.time_all,  
            model.final_score,  
            model.ode_final_score,
            model.mix_final_score,  
            model.get_latent(),  
            model.get_odelatent(),
            model.get_mix_latent()
        ]  
        joblib.dump(res, filepath)  
        print(f"Results saved to {filepath}")  
    except Exception as e:  
        print(f"Error saving results: {e}")  
        sys.exit(1)  

if __name__ == "__main__":  
    main()