import pandas as pd
import random
import os
import string
from skimage import io

from PIL import Image
import torch
import torch.utils.data as data

from sklearn.preprocessing import LabelEncoder

torch.set_printoptions(precision=8)
random.seed(1234)
torch.manual_seed(1234)


class ShoppingDataset(data.Dataset):
    """
        Shopping Dataset.
        Args:
            csv_file (string): csv file path.
            root_dir (string): image directory.
            transform (callable, optional): image transformer.
            max_len (int): maximum length of tokens.
            vocab (Vocab): vocabulary wrapper.
        """

    def __init__(self, csv_file, image_dir, transform=None, max_len=None, vocab=None):
        self.data_frame = pd.read_csv(csv_file, sep='\001', dtype=str, keep_default_na=True)
        self.image_dir = image_dir
        self.transform = transform
        self.label_codec = LabelEncoder()
        self.max_len = max_len
        self.vocab = vocab

        self._init_dataset()

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        """Returns one data pair (image, text and label)."""
        # image, text, label = self.samples[idx]
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir, self.data_frame.iloc[idx].path)
        image = io.imread(img_name)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)

        prod_nm = self._preprocess_text(self.data_frame.iloc[idx].prod_nm, self.max_len)
        tokens = prod_nm.split()
        n_pad = self.max_len - len(tokens)
        token_to_id = [self.vocab[t] for t in tokens]
        token_to_id.extend([self.vocab['<pad>']] * n_pad)
        text = torch.LongTensor(token_to_id)

        label = self.data_frame.iloc[idx].cat_id
        label = self.label_codec.transform([label])

        sample = {'image': image, 'text': text, 'label': label}
        return sample

    def _init_dataset(self):
        labels = list(set(list(self.data_frame.cat_id)))
        self.label_codec.fit(labels)

    def _preprocess_text(self, text, max_len):
        text = text.split('\x02')[:max_len]
        text = [''.join(c for c in s if c not in string.punctuation) for s in
                text]  # remove punctuation. e.g.!"#$%&\'()*+,-./:;
        text = ' '.join(text)
        return text


def get_loader(csv_file, image_dir, batch_size, transform, vocab, max_len, num_workers, shuffle, test_split=0.1):
    """Returns torch.utils.data.DataLoader for custom shopping dataset."""
    # Shopping dataset
    dataset = ShoppingDataset(csv_file=csv_file,
                              image_dir=image_dir,
                              transform=transform,
                              max_len=max_len,
                              vocab=vocab)
    label_codec = dataset.label_codec

    # split dataset
    test_size = int(test_split * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = data.random_split(dataset, [train_size, test_size])

    # Data loader for Shopping dataset
    # This will return (images, texts, labels) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # labels: a tensor of shape (batch_size).
    tr_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)

    data_size = {'train': train_size, 'test': test_size}

    return tr_loader, test_loader, label_codec, data_size