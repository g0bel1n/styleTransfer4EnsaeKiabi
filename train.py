import argparse

import torch
import torchvision.models as models
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                           SpinnerColumn, TextColumn, TimeElapsedColumn,
                           TimeRemainingColumn)
from torch.optim import Adam
from torchvision.utils import save_image
import torchvision.transforms as transform

from src import VGG, Loss, loader, Denormalize


def train(content_img_path, style_img_path, n_epoch, lr, alpha, beta):

    device=torch.device("cuda" if (torch.cuda.is_available()) else 'cpu')

    style = loader(path = style_img_path, device=device)
    content =  loader(path = content_img_path, device=device)

    generated_image=content.clone().requires_grad_(True).to(device)

    model=VGG().to(device)

    dn =Denormalize()

    optimizer=Adam([generated_image], lr=lr)
    total_loss = 0
    loss = Loss(alpha=alpha, beta=beta)
    with Progress(TextColumn("[bold blue] Transfering",), SpinnerColumn(spinner_name='growHorizontal'), BarColumn(), MofNCompleteColumn(), TextColumn('[ elapsed'), TimeElapsedColumn(), TextColumn('| eta'), TimeRemainingColumn(), TextColumn("]{task.description}")) as progress:
        pbar = progress.add_task(f"- Loss {total_loss:.2f}", total=n_epoch+1)
        for e in range(n_epoch+1):
            #extracting the features of generated, content and the original required for calculating the loss
            gen_features=model(generated_image)
            orig_feautes=model(content)
            style_featues=model(style)

            #iterating over the activation of each layer and calculate the loss and add it to the content and the style loss
            total_loss=loss(gen_features, style_featues, orig_feautes)
            #optimize the pixel values of the generated image and backpropagate the loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            progress.update(pbar, advance=1, description=f"- Loss {total_loss:.2f}" )

            #print the image and save it after each 100 epoch
            if(e%500==0): 
                #dn_gen = dn(generated_image).clone()           
                save_image(generated_image.clone(),f"data/output/gen_{e}.png")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", type=str, default="data/input/style_leo.jpg")
    parser.add_argument("-c", type=str, default="data/input/ensae_1.jpg")
    parser.add_argument("--alpha", type=int, default=10)
    parser.add_argument("--beta", type=int, default=1e7)
    parser.add_argument("--n_epoch", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.004)
    args = parser.parse_args()

    train(content_img_path=args.c, style_img_path=args.s, n_epoch=args.n_epoch, lr=args.lr, alpha=args.alpha, beta=args.beta)
