import json
import os
from typing import Callable, Dict

import PIL.Image
import torch
import torch.utils.data


class SarcasmDataset(torch.utils.data.Dataset):
    """Dataset of Sarcasm videos."""
    FRAMES_DIR_PATH = '../data/frames/utterances_final'

    def __init__(self, transform: Callable = None, videos_data_path: str = '../data/sarcasm_data.json',
                 check_missing_videos: bool = True) -> None:
        self.transform = transform

        with open(videos_data_path) as file:
            videos_data_dict = json.load(file)

        for video_id in list(videos_data_dict.keys()):  # Convert to list to possibly remove items.
            video_folder_path = self._video_folder_path(video_id)
            if not os.path.exists(video_folder_path):
                if check_missing_videos:
                    raise FileNotFoundError(f"Directory {video_folder_path} not found, which was referenced in"
                                            f" {videos_data_path}")
                else:
                    del videos_data_dict[video_id]

        self.video_ids = list(videos_data_dict.keys())

        self.frame_count_by_video_id = {video_id: len(os.listdir(self._video_folder_path(video_id)))
                                        for video_id in self.video_ids}

    @staticmethod
    def _video_folder_path(video_id: str) -> str:
        return os.path.join(SarcasmDataset.FRAMES_DIR_PATH, video_id)

    @staticmethod
    def features_file_path(model_name: str, layer_name: str) -> str:
        return f'../data/features/utterances_final/{model_name}_{layer_name}.hdf5'

    def __getitem__(self, index) -> Dict[str, object]:
        video_id = self.video_ids[index]

        frames = None

        video_folder_path = self._video_folder_path(video_id)
        for i, frame_file_name in enumerate(os.listdir(video_folder_path)):
            frame = PIL.Image.open(os.path.join(video_folder_path, frame_file_name))
            if self.transform:
                frame = self.transform(frame)

            if frames is None:
                # noinspection PyUnresolvedReferences
                frames = torch.empty((self.frame_count_by_video_id[video_id], *frame.size()))

            frames[i] = frame

        return {'id': video_id, 'frames': frames}

    def __len__(self) -> int:
        return len(self.video_ids)
