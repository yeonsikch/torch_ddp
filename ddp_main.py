#!/usr/bin/env python
# -*- coding: utf-8 -*-
from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

SEED = 42
BATCH_SIZE = 9
NUM_EPOCHS = 3


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(10, 100),
            nn.Linear(100, 100),
            nn.Linear(100, 100),
            nn.Linear(100, 100),
            nn.Linear(100, 1000),
            nn.Linear(1000, 1000),
            nn.Linear(1000, 100),
            nn.Linear(100, 100),
            nn.Linear(100, 100),
            nn.Linear(100, 100),
            nn.Linear(100, 10),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(10, 2),
            nn.Linear(2, 1),
        )

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(out)
        return out


class CustomDataset(Dataset):

    def __init__(self):
        self.data = torch.randn((100,10))
        #self.label = torch.ones(100)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
        #return {'data' : self.data[idx], 'label' : self.label[idx]}

def main():
    parser = ArgumentParser('DDP usage example')
    parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank.')  # you need this argument in your scripts for DDP to work
    args = parser.parse_args()

    # keep track of whether the current process is the `master` process (totally optional, but I find it useful for data laoding, logging, etc.)
    args.is_master = args.local_rank == 0

    # set the device
    args.device = torch.cuda.device(args.local_rank)

    # initialize PyTorch distributed using environment variables (you could also do this more explicitly by specifying `rank` and `world_size`, but I find using environment variables makes it so that you can easily use the same script on different machines)
    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)

    # set the seed for all GPUs (also make sure to set the seed for random, numpy, etc.)
    torch.cuda.manual_seed_all(SEED)

    # initialize your model (BERT in this example)

    model = ToyModel()
    #model = BertForMaskedLM.from_pretrained('bert-base-uncased')

    # send your model to GPU
    model = model.to(args.local_rank)

    # initialize distributed data parallel (DDP)
    model = DDP(
        model,
        device_ids=[args.local_rank],
        output_device=args.local_rank
    )

    # initialize your dataset
    dataset = CustomDataset()

    # initialize the DistributedSampler
    sampler = DistributedSampler(dataset)

    # initialize the dataloader
    dataloader = DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=BATCH_SIZE
    )

    # start your training!
    for epoch in range(NUM_EPOCHS):
        # put model in train mode
        model.train()

        # let all processes sync up before starting with a new epoch of training
        dist.barrier()

        for step, batch in enumerate(dataloader):
            # send batch to device
            # print('step :', step)
            # print('batch :', batch.shape)
            # print('args.device :', args.device)
            # print('args.local_rank :', args.local_rank)
            # print('type(batch) :', type(batch))
            #batch = torch.Tensor(t.to(args.local_rank) for t in batch)
            batch = batch.to(args.local_rank)
            
            # forward pass
            outputs = model(batch)
            
            # compute loss
            loss = outputs[0]
            print(loss)
            # etc.


if __name__ == '__main__':
    main()