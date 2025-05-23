import argparse  
import torch  
import scanpy as sc
from vgae import agent  
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
    parser = argparse.ArgumentParser(description='Train agent with specified encoder and decoder types.')  
      
    parser.add_argument('--encoder_type', type=str, default='GAT', help='Encoder type to train.')  
    parser.add_argument('--decoder_type', type=str, default='MLP', help='Decoder type to train.')  
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs.')  
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the training.')  
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input AnnData file.')  
    parser.add_argument('--layer', type=str, default='counts', help='Layer of the AnnData object to use for input features.')  
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden layer dimension for the encoder.')  
    parser.add_argument('--latent_dim', type=int, default=10, help='Latent space dimension.')  
    parser.add_argument('--encoder_hidden_layers', type=int, default=2, help='Number of hidden layers in the graph encoder.')  
    parser.add_argument('--decoder_hidden_dim', type=int, default=128, help='Hidden dimension for the MLPDecoder (if used).')  
    parser.add_argument('--feature_decoder_hidden_layers', type=int, default=2, help='Number of hidden layers in the feature decoder.')  
    parser.add_argument('--dropout', type=float, default=5e-3, help='Dropout rate.')  
    parser.add_argument('--use_residual', type=str2bool, nargs='?', const=True, default=True, help='Whether to use residual connections (default: True).')  
    parser.add_argument('--Cheb_k', type=int, default=1, help='The order of Chebyshev polynomials for ChebConv.')  
    parser.add_argument('--alpha', type=float, default=0.5, help='Teleport probability.')  
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate.')  
    parser.add_argument('--beta', type=float, default=1.0, help='Weight for the KL divergence term in the loss function.')  
    parser.add_argument('--graph_weight', type=float, default=1.0, help='Weight for the graph reconstruction loss.')  
    parser.add_argument('--num_parts', type=int, default=10, help='Number of partitions for clustering the graph data.')  
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory to save the trained models.')  
    parser.add_argument('--run_number', type=int, default=1, help='Run number for repeated experiments.')
    parser.add_argument('--update_steps', type=int, default=10, help='Frequency of updating the progress postfix.')
    parser.add_argument('--silent', type=str2bool, nargs='?', const=True, default=False, help='If set to True, disables the tqdm progress bar (default: False).')
    
    args = parser.parse_args()  
 
    adata = sc.read_h5ad(args.data_path)  
     
    import os  
    if not os.path.exists(args.output_dir):  
        os.makedirs(args.output_dir)  

    model = agent(  
        adata=adata,  
        layer=args.layer,  
        hidden_dim=args.hidden_dim,  
        latent_dim=args.latent_dim,  
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
        graph=args.graph_weight,  
        device=torch.device(args.device),  
        num_parts=args.num_parts,  
    )  
 
    model.fit(epochs=args.epochs, update_steps=args.update_steps, silent=args.silent)  
    model.score_final()  
    RUN_NUMBER = args.run_number  
    
    filename = f"agent_{args.encoder_type}_{args.decoder_type}_run{RUN_NUMBER}.pkl"  
    filepath = os.path.join(args.output_dir, filename)  
    res = [model.loss, model.score, model.resource, model.time_all, model.final_score]
    joblib.dump(res, filepath) 
    
    print(f"Saved into {filepath}")  

if __name__ == "__main__":  
    main()


    
