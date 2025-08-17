# -*- coding: utf-8 -*-

import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from torch import tensor, float32

class PairedImageDataset(Dataset):
    """
    Dataset that returns pairs of images (clean, new) or (clean, ref) along with their score.
    Supports optional preloading of the entire dataset to a specified device (e.g., GPU).

    Args:
        root_dir (str): Directory with subfolders 'clean', 'new', 'ref'.
        metadata_json_path (str): Path to JSON file of the form:
            [
                {
                    "unique_sample_id": "new_Fuziki_case10_2789_2650",
                    "clean_image": 4547604,
                    "distorted_image": 4547601,
                    "score": 0,
                    "metadata": {
                        "method": "new" or "ref
                        ...
                    }
                },
                ...
            ]
    
    KwArgs:
        transform (callable, optional): Default: transforms.ToTensor(). Transform to apply to both images.
        preload (bool, optional): Default: False. If True, preload the entire dataset to the specified device. 
        device (torch.device, optional): Default: None. The device to preload the data to. Required if preload is True.
    """
    def __init__(self, root_dir, metadata_json_path, transform=transforms.ToTensor(), preload=False, device=None):
        if preload and device is None:
            raise ValueError("Device must be specified if preload is True.")

        self.root_dir = root_dir
        self.transform = transform
        self.preload = preload
        self.device = device

        # Load metadata
        self.meta = pd.read_json(metadata_json_path)

        # Build list of samples: (clean_path, other_path, score)
        self.samples = []
        for _, row in self.meta.iterrows():
            clean_id = str(row['clean_image'])
            distorted_id = str(row['distorted_image'])
            method = row['metadata']['method']
            score = row['score']

            # make triples (clean path, distorted path, score)
            clean_path = os.path.join(root_dir, 'clean', f'{clean_id}.png')
            distorted_path = os.path.join(root_dir, method, f'{distorted_id}.png')

            if os.path.exists(distorted_path):
                self.samples.append((clean_path, distorted_path, score))

        # Preload data if requested
        if self.preload:
            print(f"Preloading dataset to {self.device}...")
            self.preloaded_samples = []

            for sample, (clean_path, other_path, score) in enumerate(self.samples, start=1):
                print(f"\rsample {sample}/{self.__len__()}", end='') # Print how many batchs are done
                img_clean = Image.open(clean_path)
                img_other = Image.open(other_path)

                if self.transform:
                    img_clean = self.transform(img_clean)
                    img_other = self.transform(img_other)

                # Store as a tuple of tensors
                self.preloaded_samples.append((img_clean.to(self.device), img_other.to(self.device), tensor(score, dtype=float32).to(self.device)))

            print() # Newline after same-line print in the loop
            print("Preloading complete.")


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        if self.preload:
            # Return preloaded tensors
            return self.preloaded_samples[idx]
        else:
            # Original logic: load and transform on demand
            clean_path, other_path, score = self.samples[idx]
            img_clean = Image.open(clean_path)
            img_other = Image.open(other_path)

            if self.transform:
                img_clean = self.transform(img_clean)
                img_other = self.transform(img_other)

            return img_clean, img_other, score

            