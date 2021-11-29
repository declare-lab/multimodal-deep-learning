# Visual modality

This folder tackles the visual modality. It extracts features and provides some baselines.

## Feature extraction

1. Download the videos from Google Drive to `data/videos`, placing the files there without subdirectories.
2. Move to this directory:

    ```bash
    cd visual
    ```

3. Run `save_frames.sh` to extract the frames in the video files:

    ```bash
    ./save_frames.sh
    ```

4. To extract the features and save them in heavy H5 files:

    ```bash
    ./extract_features.py resnet
    ``` 
