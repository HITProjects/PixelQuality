Samsung dataset placeholder

Put the dataset here so the code can run out-of-the-box:
c:\Users\YehonatanR\OneDrive\Documents\HIT Computer Science\3rd\PixelQuality\PythonProject\notebooks\data_local\samsung\dataset

Required structure:
- clean/
- new/
- ref/
- test_metadata.json

Shared OneDrive link (download and extract here):
https://hitacil-my.sharepoint.com/:f:/g/personal/alexanderap_hit_ac_il/EkF8bt6BCltOlk9PWV0-BAoBTyNtYEbcYwVpwMzNhkbSbA?e=ZZY8Y1

Advanced: you can also set a custom path without moving files:
- Temporary (current session):
  Windows PowerShell:
      $env:PIXELQ_DATASET_DIR="C:\\path\\to\\dataset"
  CMD:
      set PIXELQ_DATASET_DIR=C:\\path\\to\\dataset

- Or create data/metadata/dataset_path.json with:
      { "dataset_dir": "C:/path/to/dataset" }