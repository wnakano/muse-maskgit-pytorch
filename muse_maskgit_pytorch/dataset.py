from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import ImageFile
from pathlib import Path
from muse_maskgit_pytorch.t5 import MAX_LENGTH
import datasets
from datasets import Image, load_from_disk
import random
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageDataset(Dataset):
    def __init__(
        self, dataset, image_size, image_column="image", flip=True, center_crop=True
    ):
        super().__init__()
        self.dataset = dataset
        self.image_column = image_column
        transform_list = [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize(image_size),
        ]
        if flip:
            transform_list.append(T.RandomHorizontalFlip())
        if center_crop:
            transform_list.append(T.CenterCrop(image_size))
        transform_list.append(T.ToTensor())
        self.transform = T.Compose(transform_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image= self.dataset[index][self.image_column]
        return self.transform(image)
    
class ImageTextDataset(ImageDataset):
    def __init__(
        self,
        dataset,
        image_size,
        tokenizer,
        image_column="image",
        caption_column=None,
        flip=True,
        center_crop=True,
    ):
        super().__init__(
            dataset,
            image_size=image_size,
            image_column=image_column,
            flip=flip,
            center_crop=center_crop,
        )
        self.caption_column = caption_column
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        image_path = self.dataset[index][self.image_column]
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        if self.caption_column == None:
            text = ""
        elif isinstance(descriptions, list):
            if len(descriptions) == 0:
                text = ""
            else:
                text = random.choice(descriptions)
        else:
            caption_file = self.dataset[index][self.caption_column]
            descriptions = Path(caption_file).read_text().split('\n')
            descriptions = list(filter(lambda t: len(t) > 0, descriptions))
            # rn only working with 1st caption
            text = descriptions[0]
            
        encoded = self.tokenizer.batch_encode_plus(
            [str(text)],
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
        )

        input_ids = encoded.input_ids
        attn_mask = encoded.attention_mask
        
        # dirty way to fix shape issue
        input_ids = input_ids.squeeze(0)
        attn_mask = attn_mask.squeeze(0)
        
        return self.transform(image), input_ids, attn_mask


def get_dataset_from_dataroot(
    data_root, image_column="image", caption_column="caption", save_path="dataset"
):
    if os.path.exists(save_path):
        return load_from_disk(save_path)
    image_paths = list(Path(data_root).rglob("*.[jJ][pP][gG]"))
    random.shuffle(image_paths)
    data_dict = {args.image_column: [], args.caption_column: []}
    image_paths = image_paths[:1000]
    print(f"Found {len(image_paths)} images")
    for image_path in image_paths:
        image_path = str(image_path)
        text_file = image_path.replace("jpg", "txt") 
        data_dict[args.image_column].append(image_path)
        data_dict[args.caption_column].append(text_file)

    return datasets.Dataset.from_dict(data_dict)

def split_dataset_into_dataloaders(dataset, valid_frac=0.05, seed=42, batch_size=1):
    if valid_frac > 0:
        train_size = int((1 - valid_frac) * len(dataset))
        valid_size = len(dataset) - train_size
        dataset, validation_dataset = random_split(
            dataset,
            [train_size, valid_size],
            generator=torch.Generator().manual_seed(seed),
        )
        print(
            f"training with dataset of {len(dataset)} samples and validating with randomly splitted {len(validation_dataset)} samples"
        )
    else:
        validation_dataset = dataset
        print(
            f"training with shared training and valid dataset of {len(dataset)} samples"
        )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    validation_dataloader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=True
    )
    return dataloader, validation_dataloader
