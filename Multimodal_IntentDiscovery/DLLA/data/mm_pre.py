from torch.utils.data import Dataset
import torch
import numpy as np

__all__ = ['MMDataset']


    
class MMDataset(Dataset):
        
    def __init__(self, label_ids, text_data, text_keyword, text_keyframe, video_data, audio_data):
        
        self.label_ids = label_ids
        self.text_data = text_data
        self.text_keyword = text_keyword
        self.text_keyframe = text_keyframe
        self.video_data = video_data
        self.audio_data = audio_data
        self.size = len(self.text_data)

    def __len__(self):
        return self.size

    def __getitem__(self, index):

       
        sample = {
            'text_feats': torch.tensor(self.text_data[index]),
            'text_keyword': torch.tensor(self.text_keyword[index]),
            'text_keyframe': torch.tensor(self.text_keyframe[index]),
            'video_feats': torch.tensor(np.array(self.video_data['feats'][index])),
            'video_lengths': torch.tensor(np.array(self.video_data['lengths'][index])),
            'audio_feats': torch.tensor(np.array(self.audio_data['feats'][index])),
            'audio_lengths': torch.tensor(np.array(self.audio_data['lengths'][index]))
        } 
        if self.label_ids is not None:
            sample['label_ids'] = torch.tensor(self.label_ids[index])

        return sample
