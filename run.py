import torch
from torch import nn
from torch.utils.data import DataLoader
import argparse

import os
import ipdb
import torch.fft
import lightning as L
import torch.optim as optim
from Transfomer_utils import Trajformer
from transformer_dataloader import TrajectoryDataset, TrajDatasetSplit, collate_fn, train, validate, topk_hits


def run(args):
    # Setting the seed
    L.seed_everything(args.random_seed)

    data_path = f"./TrajectoryDistiallation/datasets/{args.dataset}/data_info.pkl"
    dataset, freq = TrajDatasetSplit(data_path, args.abnormal_samples, args.normal_samples)
    abnormal_ids = torch.load(f"./TrajectoryDistiallation/datasets/{args.dataset}/abnormal_ids.pt")

    train_data, test_data = TrajectoryDataset(dataset["train"], freq), TrajectoryDataset(dataset["test"], freq)
    train_loader = DataLoader(train_data, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(train_data, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_data, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True, drop_last=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Trajformer(num_classes = args.num_classes,
                       num_layers = args.num_layers,
                       dim_hidden = args.dim_hidden,
                       freq = freq,
                       num_kernels = args.num_kernels,
                       input_dim_st = args.input_dim_st,
                       input_dim_text = args.input_dim_text,
                       alpha = args.alpha,
                       scaler = args.scaler,
                       num_heads = args.num_heads,
                       dropout = args.dropout,
                       attn_threhold = args.attn_threhold,
                       return_attention = args.return_attention).to(device)

    label_criterion, attn_criterion = nn.CrossEntropyLoss(), nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr,
    )

    model.train()

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    least_loss = float('inf')
    # Start the training.
    for epoch in range(args.epochs):
        print(f"[INFO]: Epoch {epoch+1} of {args.epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader, 
                                                optimizer, label_criterion, attn_criterion, args.return_attention)
        valid_epoch_loss, valid_epoch_acc, _ = validate(model, valid_loader,  
                                                    label_criterion, attn_criterion, args.return_attention)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss}, training acc: {train_epoch_acc}")
        print(f"Validation loss: {valid_epoch_loss}, validation acc: {valid_epoch_acc}")
        # Save model.
        if valid_epoch_loss < least_loss:
            least_loss = valid_epoch_loss
            print(f"Saving best model till now... LEAST LOSS {valid_epoch_loss:.3f}")
            torch.save(
                model, os.path.join(args.outputs_dir, 'model.pth')
            )
        print('-'*50)

    trained_model = torch.load(
        os.path.join(args.outputs_dir, 'model.pth')
    )

    _, _, probs = validate(
        trained_model, 
        test_loader,  
        label_criterion, 
        attn_criterion, 
        args.return_attention
    )

    test_ids = list(probs.keys())
    scores = [float(probs[test_ids[i]][args.abnormal_index].cpu()) for i in range(len(test_ids))]
    test_ids = [id.cpu() for id in test_ids]

    print("Top k hits for abnormal trajectors when k = 10, 25, 50, 100:",
          topk_hits(10, scores, test_ids, abnormal_ids), 
          topk_hits(25, scores, test_ids, abnormal_ids), 
          topk_hits(50, scores, test_ids, abnormal_ids), 
          topk_hits(100, scores, test_ids, abnormal_ids)
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #Model Hyperparameter
    parser.add_argument('--outputs_dir', type=str, default=f"./TrajectoryDistiallation/model_outputs")
    parser.add_argument('--dataset', type=str, default="hunger")
    parser.add_argument('--abnormal_samples', type=int, default=5)
    parser.add_argument('--normal_samples', type=int, default=45)
    parser.add_argument('--abnormal_index', type=int, default=0)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=10e-5)
    #Parameter to Initialize Trajectory Transformer
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dim_hidden', type=int, default=64)
    parser.add_argument('--num_kernels', type=int, default=8)
    parser.add_argument('--input_dim_st', type=int, default=4) 
    parser.add_argument('--input_dim_text', type=int, default=768)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--scaler', type=float, default=10e-5)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--attn_threhold', type=float, default=0.1)
    parser.add_argument('--return_attention', type=bool, default=True)
    args = parser.parse_args()

    run(args)
