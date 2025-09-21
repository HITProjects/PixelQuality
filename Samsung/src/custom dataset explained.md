How to load the test partition of Samsung Dataset:

1. Download the Samsung dataset and test_metadata.json from the drive. Expected structure:
    ```
    dataset/
    ├── clean/
    │   └── PNGs...
    ├── new
    │   └── PNGs...
    ├── ref
    │    └── PNGs...
    └── test_metadata.json
    ```

2. Add the PairedImageDataset class to your code.
3. Define the paths to the dataset and to the test metadata file.
4. Run:
   ```
   test_dataset = PairedImageDataset(dataset_path, test_metadata_path)

   test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
   ```
5. The class accepts optional arguments:
   * transform (callable, optional):
     * Transform to apply to both images.
     * Default: transforms.ToTensor().
   * preload (bool, optional):
     * If True, preloads the entire dataset to the specified device.
     * Default: False.
   * device (torch.device, optional):
     * The device to preload the data to. Required if preload is True.
     * Default: None.


