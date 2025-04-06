from torch.utils.data import Dataset


class TextBinaryDataset(Dataset):
    def __init__(self, texts, labels):
        """
        Args:
            texts (list of str): List of text samples.
            labels (list of int): List of binary labels (0 or 1) corresponding to the text samples.
        """
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        return text, label
