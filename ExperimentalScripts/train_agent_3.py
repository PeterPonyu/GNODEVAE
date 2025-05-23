import argparse  
from anndata import AnnData  
import torch  
import sys  
from vgae import agent  
import scanpy as sc
import joblib

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
        description="Run the GraphVAE agent with specified parameters."  
    )  

    # Add arguments corresponding to the parameters of the agent class  
    parser.add_argument("--data_path", type=str, required=True, help="Path to the AnnData object (e.g., .h5ad file).")  
    parser.add_argument("--layer", type=str, default="counts", help="Layer of the AnnData object to use for input features.")  
    parser.add_argument("--n_var", type=int, default=None, help="Number of highly variable genes to select.")  
    parser.add_argument("--tech", type=str, default="PCA", help="Decomposition method to use.")  
    parser.add_argument("--n_neighbors", type=int, default=15, help="Number of neighbors for graph construction.")  
    parser.add_argument("--batch_tech", type=str, default=None, help="Method to correct batch effects ('harmony' or 'scvi').")  
    parser.add_argument("--all_feat", type=str2bool, nargs='?', const=True, default=True, help="Use all features or only highly variable ones.")  
    parser.add_argument("--interpretable", type=str2bool, nargs='?', const=True, default=False, help="Make the model interpretable.")  
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden layer dimension for the encoder.")  
    parser.add_argument("--latent_dim", type=int, default=10, help="Latent space dimension.")  
    parser.add_argument("--idim", type=int, default=2, help="Input dimension for latent representation.")  
    parser.add_argument("--encoder_type", type=str, default="GAT", help="Type of graph convolutional layer.")  
    parser.add_argument("--encoder_hidden_layers", type=int, default=2, help="Number of hidden layers in the graph encoder.")  
    parser.add_argument("--decoder_type", type=str, default="MLP", help="Type of graph decoder.")  
    parser.add_argument("--decoder_hidden_dim", type=int, default=128, help="Hidden dimension for the MLPDecoder.")  
    parser.add_argument("--feature_decoder_hidden_layers", type=int, default=2, help="Number of hidden layers in the feature decoder.")  
    parser.add_argument("--dropout", type=float, default=5e-3, help="Dropout rate.")  
    parser.add_argument('--use_residual', type=str2bool, nargs='?', const=True, default=True, help='Whether to use residual connections (default: True).')  
    parser.add_argument("--Cheb_k", type=int, default=1, help="Order of Chebyshev polynomials for ChebConv.")  
    parser.add_argument("--alpha", type=float, default=0.5, help="Teleport probability.")  
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for the optimizer.")  
    parser.add_argument("--beta", type=float, default=1.0, help="Weight for the KL divergence term in the loss function.")  
    parser.add_argument("--graph", type=float, default=1.0, help="Weight for the graph reconstruction loss.")  
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run the model on ('cpu' or 'cuda').")  
    parser.add_argument("--num_parts", type=int, default=10, help="Number of partitions for clustering the graph data.")  
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save the trained models.')  
    parser.add_argument('--run_number', type=int, default=1, help='Run number for repeated experiments.')
    parser.add_argument('--update_steps', type=int, default=10, help='Frequency of updating the progress postfix.')
    parser.add_argument('--silent', type=str2bool, nargs='?', const=True, default=False, help='If set to True, disables the tqdm progress bar (default: False).')
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs.')
    
    # Parse the arguments  
    args = parser.parse_args()  

    # Load the AnnData object from the specified path  
    try:  
        adata = sc.read_h5ad(args.data_path)  
    except Exception as e:  
        print(f"Error loading AnnData file: {e}")  
        sys.exit(1)  

    # Instantiate the agent with the parsed arguments  
    model = agent(  
        adata=adata,  
        layer=args.layer,  
        n_var=args.n_var,  
        tech=args.tech,  
        n_neighbors=args.n_neighbors,  
        batch_tech=args.batch_tech,  
        all_feat=args.all_feat,  
        interpretable=args.interpretable,  
        hidden_dim=args.hidden_dim,  
        latent_dim=args.latent_dim,  
        idim=args.idim,  
        encoder_type=args.encoder_type,  
        encoder_hidden_layers=args.encoder_hidden_layers,  
        decoder_type=args.decoder_type,  
        decoder_hidden_dim=args.decoder_hidden_dim,  
        feature_decoder_hidden_layers=args.feature_decoder_hidden_layers,  
        dropout=args.dropout,  
        use_residual=args.use_residual,  
        Cheb_k=args.Cheb_k,  
        alpha=args.alpha,  
        lr=args.lr,  
        beta=args.beta,  
        graph=args.graph,  
        device=torch.device(args.device),  
        num_parts=args.num_parts,  
    )  

    import os  
    if not os.path.exists(args.output_dir):  
        os.makedirs(args.output_dir)
    model.fit(epochs=args.epochs, update_steps=args.update_steps, silent=args.silent)  
    model.score_final()  
    RUN_NUMBER = args.run_number  
    
    filename = f"agent_{args.encoder_type}_{args.decoder_type}_{args.tech}_run{RUN_NUMBER}.pkl"  
    filepath = os.path.join(args.output_dir, filename)  
    res = [model.loss, model.score, model.resource, model.time_all, model.final_score]
    joblib.dump(res, filepath) 
    
    print(f"Saved into {filepath}")
if __name__ == "__main__":  
    main()



    