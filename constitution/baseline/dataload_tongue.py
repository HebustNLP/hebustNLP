from PIL.Image import open
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from pandas import read_csv
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import os
from transformers import ViTFeatureExtractor


class ClassifyDataset(Dataset):
    def __init__(self, image_dir, csv_path, resize,class_label):
        super(ClassifyDataset, self).__init__()
        self.image_dir = image_dir
        self.csv_path = csv_path
        self.resize = resize
        feature_extractor = ViTFeatureExtractor.from_pretrained('/home/zhangxiaohan/constitution/baseline/vit-base-patch32-224-in21k')
        self.transformer = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
        ])
        self.df = read_csv(self.csv_path, encoding='utf-8')
        self.class_label = class_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir,str(self.df['img_name'][idx])+'.jpg')
        if os.path.exists(img_path):
            image = open(os.path.join(self.image_dir,str(self.df['img_name'][idx])+'.jpg')).convert("RGB")
            data = self.transformer(image)
            label = self.df[self.class_label][idx]
        else:
            print(str(self.df['img_name'][idx])+'.jpg')

        return data, label


def get_dataloader(image_dir, csv_path, resize,class_label, batch_size, train_percent=0.8):
    dataset = ClassifyDataset(image_dir, csv_path, resize, class_label)
    valid_percent = (1-train_percent)/2
    test_percent = (1-train_percent)/2
    num_sample = len(dataset)
    num_train = int(train_percent * num_sample)
    num_valid = int(valid_percent * num_sample)
    num_test = num_sample-num_train-num_valid
    train_ds, valid_ds, test_ds = random_split(dataset, (num_train, num_valid, num_test))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True,
                          persistent_workers=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True,
                          persistent_workers=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=True,
                          persistent_workers=True)
    return train_dl, valid_dl, test_dl, len(dataset), len(train_ds), len(valid_ds), len(test_ds)
