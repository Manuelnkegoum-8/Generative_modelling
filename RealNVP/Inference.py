from Normflow import Real_nvp
import torch
import pandas as pd
import argparse
import os
import numpy as np

#MODEL PARAMETERS
DIM = 4
NUM_LAYERS = 4
HIDDEN_DIM = 32
parser = argparse.ArgumentParser()
parser.add_argument("--samples", type = int,default=410)

def init_model(path,device):
    """

    """
    #INIT THE MASKS
    init_mask = torch.zeros(DIM)
    for i in range(DIM//2,DIM):
        init_mask[i] = 1.
    masks = [init_mask]

    for i in range(1,NUM_LAYERS):
        masks.append(1-masks[i-1])
    
    #load
    model = Real_nvp(dim=DIM,hidden_dim=HIDDEN_DIM,num_coupling=NUM_LAYERS,masks=masks)
    model.load_state_dict(torch.load(PATH,map_location = device))
    model = model.to(device)
    return model



def inference(model,num_samples):
    """
    """
    samples,x = model.sample(num_samples,device)
    samples  = samples.detach().cpu().numpy()
    x = x.detach().cpu().numpy()
    return samples,x



if __name__== "__main__":

    args = parser.parse_args()
    num_samples = args.samples

    #LOAD THE MODEL
    current_path = os.path.dirname(os.path.abspath(__file__))
    
    PATH = os.path.join(current_path,'RealNVP_model.pt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] : Loading real nvp model")
    model = init_model(PATH,device)
    
    #GENERATE SAMPLES
    print("[INFO] : Generating samples")
    latent,generated = inference(model,num_samples)

    #SAVE
    print("[INFO] : Succesfully generated samples")
    latent_space = pd.DataFrame()
    observed = pd.DataFrame()
    latent_space[[0,1,2,3]] = latent
    observed[[0,1,2,3]] = generated
    latent_space.to_csv(os.path.join(current_path,'latent.csv'),index=False)
    observed.to_csv(os.path.join(current_path,'generated_samples.csv'),index=False)
