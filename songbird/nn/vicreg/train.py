#%%
from songbird.dataset.dataset_info import DatasetInfo, SampleRecordingType
from songbird.dataset.audio_dataset import AudioFileDataset as Dataset, ToTensor, PitchShift
from songbird.nn.vicreg.loss import invariance_loss, variance_loss, covariance_loss, combine_losses #loss_function, 

import songbird.nn.vicreg.models.resnet_1d as resnet

#from songbird.nn.vae.models.res2d_vae import VariationalEncoder, VariationalDecoder, VariationalAutoEncoder
from torchvision.transforms import Compose

import os
import time
import datetime
import random
import json
import numpy as np
import matplotlib
from dataclasses import asdict, dataclass


# matplotlib.use('Agg')
# import matplotlib.pyplot as plt

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

@dataclass
class TrainerParameters:
    save_epoch_interval: int = 20
    random_seed: int = 44

@dataclass    
class TrainRunParameters:
    continue_training: bool = False
    epochs: int = 10
    use_amp = False

@dataclass
class ModelParameters:
    in_channels: int = 1
    out_channels: int = 1
    dilation_growth_rate: int = 1
    dilation_depth : int = 1
    repeat_num : int = 1
    kernel_size : int = 3

@dataclass
class DatasetParameters:
    dataset_name = f'data/processed/audio/2022-06-04_13'
    batch_size: int = 512
    num_workers: int = 12


@dataclass    
class OptimizerParameters:
    learning_rate: float = 1e-4
    
def get_parent_dir(path, num_levels = 1):
    for _ in range(num_levels):
        path = os.path.dirname(path)        
    return path

def get_run_dir(model_runs_dir, prefix = 'run'):
    run_index = 0
    run_dir = os.path.join(model_runs_dir, f'{prefix}_{run_index}')
    while os.path.exists(run_dir):
        run_index += 1
        run_dir = os.path.join(model_runs_dir, f'{prefix}_{run_index}')

    return run_dir

def set_random_seed(random_seed):
    # set random seeds
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    
#%%
def create_run_dir(project_dir, model_name):
    model_dir = os.path.join(project_dir, 'models', model_name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    report_dir = os.path.join(project_dir, "reports", "vae", model_name)
    if not os.path.exists(report_dir):
        os.mkdir(report_dir)

    model_runs_dir = os.path.join(project_dir, 'runs', model_name)
    run_dir = get_run_dir(model_runs_dir)
    
    return model_dir, report_dir, run_dir

def setup_writer(run_dir):
    writer = SummaryWriter(log_dir=run_dir)
    print(f"Saving run data to dir: {run_dir}")
    
    return writer
    
def get_train_parameters():
    trainer_params = TrainerParameters()
    model_params = ModelParameters()
    dataset_params = DatasetParameters()
    train_run_params = TrainRunParameters()
    optimizer_params = OptimizerParameters()
    
    return trainer_params, model_params, dataset_params, train_run_params, optimizer_params
    
def load_data(trainset_path, testset_path):
    if not os.path.exists(trainset_path):
        print("Dataset not found. Check if path is correct")
        exit()
    else:
        print(f"Dataset found. Loading dataset from {trainset_path}")

    with open(os.path.join(trainset_path, "dataset_attributes.json"), 'r') as f:
        dataset_attributes = json.load(f)['build_parameters']

    sample_rate = dataset_attributes['sample_rate']

    composed_transforms = Compose([
        ToTensor(),
    #    PitchShift(sample_rate)
    ])

    train_set = Dataset(trainset_path, transform=composed_transforms)
    test_set = Dataset(testset_path, transform=composed_transforms)

    print(f"{'#'*3} {'Dataset info' + ' ':{'#'}<{24}}")
    print(f"Total dataset length: {len(train_set) + len(test_set)}")
    
    return train_set, test_set

def create_data_loaders(train_set, test_set, batch_size):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=12)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=12)

    print(f"Train length: {len(train_set)} Test length: {len(test_set)}")
    print(f"Train batch num: {len(train_loader)} Test batch num: {len(test_loader)}")
    
    return train_loader, test_loader
    
def add_hparams_to_writer(writer, hparam_dict, metric_dict = None):
    writer.add_hparams(hparam_dict,metric_dict)

def get_model(model_params):
    model = resnet.Resnet1D(**asdict(model_params))
    return model

def get_optimizer(model, learning_rate, use_amp):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    return optimizer, scaler

def load_state(model, optimizer, model_dir):
    checkpoint_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')] #parse checkpoint files
    checkpoint_epochs = [int(f.split('_e')[-1].split('.')[0]) for f in checkpoint_files] #get the epoch number from the checkpoint file name
    newest_checkpoint_file = checkpoint_files[np.argmax(checkpoint_epochs)] #get the newest checkpoint file

    checkpoint = torch.load(os.path.join(model_dir, newest_checkpoint_file))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
        
    return start_epoch

def train(model, optimizer, scaler, train_loader, use_amp, device):
    model.train()
    
    start_time = time.time()
    batch_count = 0
    samples_count = 0
    train_loss = 0
    
    for batch_x, batch_y, _, _ in train_loader:
        with torch.cuda.amp.autocast(enabled=use_amp):

            emb_x = model(batch_x.to(device))
            emb_y = model(batch_y.to(device))

            loss = combine_losses(
                invariance_loss(emb_x, emb_y),
                variance_loss(emb_x, emb_y),
                covariance_loss(emb_x, emb_y),
            )
        
        train_loss += loss.item()
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # loss.backward()
        # optimizer.step()

        samples_count += len(batch_x)
        batch_count += 1
        
        print(f"batch: {int(batch_count/len(train_loader) * 100):3}% | train loss: {train_loss / samples_count:16.6f} | train time: {time.strftime('%H:%M:%S',time.time() - start_time)}", end = "\r")
        
    print()
    return train_loss
  
@torch.no_grad()
def validate(model, device, use_amp, test_loader):
    model.eval()
    
    start_time = time.time()
    batch_count = 0
    samples_count = 0
    validation_loss = 0
    
    for batch in test_loader:
        with torch.cuda.amp.autocast(enabled=use_amp):
            batch = batch.to(device)

            emb_x = model(batch_x)
            emb_y = model(batch_y)

            loss = combine_losses(
                invariance_loss(emb_x, emb_y),
                variance_loss(emb_x, emb_y),
                covariance_loss(emb_x, emb_y),
            )

        batch_count += 1
        print(f"batch: {int(batch_count/len(test_loader) * 100):3}% | valid loss: {validation_loss / samples_count:16.6f} | valid time: {time.strftime('%H:%M:%S',time.time() - start_time)}", end = "\r")
        
    print()
    return validation_loss  
    
#%%
def train_run(model, optimizer, scaler, train_loader, test_loader, device, writer, use_amp, start_epoch, epochs):
    print(f"{'#'*3} {'Training' + ' ':{'#'}<{24}}")
    for epoch in range(start_epoch, epochs):
        print(f"Epoch: {epoch}")
        train_loss = train(model, optimizer, scaler, train_loader, use_amp, device)
        valid_loss = validate(model, device, use_amp, test_loader)       
        
        writer.add_scalar("Loss/train", train_loss / len(train_loader.dataset), epoch)
        writer.add_scalar("Loss/valid", valid_loss / len(test_loader.dataset), epoch)
        
    return train_loss, valid_loss
        
def main():
    project_dir = get_parent_dir(os.path.abspath(__file__), num_levels = 4)
    model_name = os.path.basename(resnet.__file__).split('.')[0]
    
    model_dir, report_dir, run_dir = create_run_dir(project_dir, model_name)
    writer = setup_writer(run_dir)
    
    trainer_params, model_params, dataset_params, train_run_params, optimizer_params = get_train_parameters()
    
    set_random_seed(trainer_params.random_seed) # for reproducibility set random seed before loading data
    trainset_path = os.path.join(project_dir, dataset_params.dataset_name, 'train')
    testset_path = os.path.join(project_dir, dataset_params.dataset_name, 'test')
    train_set, test_set = load_data(trainset_path, testset_path)
    
    train_loader, test_loader = create_data_loaders(train_set, test_set, batch_size = dataset_params.batch_size)
    
    hparam_dict = asdict(trainer_params)
    add_hparams_to_writer(writer, hparam_dict, {})
    
    model = get_model(model_params)
    optimizer, scaler = get_optimizer(model, optimizer_params.learning_rate, train_run_params.use_amp)
    
    if train_run_params.continue_training:
        start_epoch = load_state(model, optimizer, model_dir)
    else:
        start_epoch = 0
    
    writer.add_graph(model, next(iter(train_loader))[0]) # add model to tensorboard and
    set_random_seed(trainer_params.random_seed) # for reproducibility set random seed after the model is initialized
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"{'#'*3} {'Training info' + ' ':{'#'}<{24}}")
    print(f"Using {device} device for training")
    print(f"Using mixed precision: {train_run_params.use_amp}")
    print(f"Epochs: {train_run_params.epochs}")
    print(f"Batch size: {dataset_params.batch_size}")


    model.to(device)
    
    set_random_seed(trainer_params.random_seed) # for reproducibility set random seed after the optimizer is initialized (may be not needed)
    train_run(model, optimizer, scaler, train_loader, test_loader, device, writer, train_run_params.use_amp, start_epoch, train_run_params.epochs)
    
if __name__ == "__main__":
    main()
#%%
# time_mean_width =  np.mean([event.end_time_index - event.start_time_index for event in all_sample_events])
# time_std_with = np.std([event.end_time_index - event.start_time_index for event in all_sample_events])
# time_mean_width, time_std_with