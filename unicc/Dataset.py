import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Importa la classe BigEarthNet e il DataModule di base da torchgeo
from torchgeo.datasets import BigEarthNet
from torchgeo.datamodules import NonGeoDataModule
from torchgeo.transforms.transforms import AugmentationSequential, _RandomNCrop, _Clamp
from torchgeo.transforms.color import RandomGrayscale
import kornia.augmentation as K
import torch.nn.functional as F

import itertools
from torchgeo.datasets import BigEarthNet
import torch
from kornia.augmentation import AugmentationBase2D


class CustomLambda(AugmentationBase2D):
    def __init__(self, fn):
        super().__init__(p=1.0, same_on_batch=False, keepdim=True)
        self.fn = fn

    def apply_transform(self, input, params, flags, transform):
        return self.fn(input)


class CustomBigEarthNet(BigEarthNet):
    def __init__(self, subset: int = None, *args, **kwargs):
        self._subset = subset  # Salva il valore del subset prima di inizializzare la classe base
        super().__init__(*args, **kwargs)

    def _load_folders(self) -> list[dict[str, str]]:
        
        filename = self.splits_metadata[self.split]['filename']
        print(filename)
        dir_s1 = self.metadata['s1']['directory']
        dir_s2 = self.metadata['s2']['directory']

        with open(os.path.join(self.root, filename)) as f:
            lines = f.read().strip().splitlines()
            


        # Applica subito il subset (se richiesto)
        if self._subset is not None:
            lines = lines[500:500+self._subset]

        pairs = [line.split(',') for line in lines]

        folders = [
            {
                's1': os.path.join(self.root, dir_s1, pair[1]),
                's2': os.path.join(self.root, dir_s2, pair[0]),
            }
            for pair in pairs
        ]
        return folders


class CustomBigEarthNetDataModule(NonGeoDataModule):
    def __init__(self, subset: int = None, transform=None, batch_size: int = 64, num_workers: int = 0, **kwargs):
        """
        Args:
            subset (int, optional): Numero di campioni da usare.
            transform: Pipeline di trasformazioni da applicare ai dati.
            batch_size (int): Dimensione del batch.
            num_workers (int): Numero di processi per il DataLoader.
            **kwargs: Altri parametri da passare al dataset.
        """
        self.subset = subset
        self.transform = transform
        self.kwargs = kwargs  
        super().__init__(CustomBigEarthNet, batch_size, num_workers, **kwargs)

    def setup(self, stage: str = None):
        if stage in ["fit", None]:
            self.train_dataset = CustomBigEarthNet(
                split="train", 
                subset=self.subset,
                **self.kwargs
            )
            self.train_dataset.transform = self.transform
        if stage in ["fit", "validate", None]:
            self.val_dataset = CustomBigEarthNet(
                split="val", 
                subset=self.subset, 
                **self.kwargs
            )
        if stage in ["test", None]:
            self.test_dataset = CustomBigEarthNet(
                split="test", 
                subset=self.subset, 
                **self.kwargs
            )

def min_max_fn(x):
    min_val = x.amin(dim=(2, 3), keepdim=True)  # per ogni immagine
    max_val = x.amax(dim=(2, 3), keepdim=True)
    return (x - min_val) / (max_val - min_val + 1e-8)


def carica_dati(args, setup = "fit"):

    print(args)

    if args.transform:
        transforms = [
            CustomLambda(min_max_fn),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation(degrees=90.0, p=0.5)
        ]
        print('faccio aug')

        if args.fintuning_bands == "rgb":
            transforms.append(K.RandomGrayscale(p=0.05))
        train_transform = AugmentationSequential(*transforms, data_keys=["image"])

    else: 
        train_transform = None
        print('no aug ')
    

    dm = CustomBigEarthNetDataModule(
            root=args.data_dir,
            download=True,  
            #subset=10,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            bands=args.bands,     # 's1', 's2' oppure 'all'
            num_classes=args.num_classes,
            transform=train_transform   # 19 o 43
        )
    print('dm', dm)
    
    dm.setup(setup)
    print('dm setup')

    if setup == "fit":
        train = dm.train_dataset
        train_loader = dm.train_dataloader()
        print(train)
        print('--creato train loader')

        validation = dm.val_dataset
        print(validation)
        validation_loader = dm.val_dataloader()
        print('--creato validation loader')
        
        return train, train_loader, validation, validation_loader

    else:
        test = dm.test_dataset
        test_loader = dm.test_dataloader()
        print('--creato test loader')

        return test, test_loader


