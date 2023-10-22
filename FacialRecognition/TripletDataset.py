from numpy import array
from torch import from_numpy, Tensor, cuda
from torch.utils.data import Dataset
from typing import List
from PIL import Image
from os import listdir
from random import choice, choices


class TripletDataset(Dataset):

    def __init__(self,
                 posPathList: List[str],
                 negPath: str,
                 size: int = 500) -> None:
        self.pos = []
        self.neg = [
            from_numpy(array(Image.open(f"{negPath}{f}"))).permute(
                2, 0, 1).float() / 255.0
            for f in choices(listdir(negPath), k=max(size, 500))
        ]
        for d in posPathList:
            self.pos.append([
                from_numpy(array(Image.open(f"{d}/{f}"))).permute(
                    2, 0, 1).float() / 255.0
                for f in choices(listdir(d), k=size)
            ])
        self.__len = size

    def __getitem__(self, index):
        if len(self.pos) > 1:
            posClass: int = choice([i for i in range(len(self.pos))])
            negClass: List[List[Tensor]] = choice(
                [self.neg] +
                [self.pos[c] for c in range(len(self.pos)) if c != posClass])
            if cuda.is_available():
                return choice(self.pos[posClass]).cuda(), choice(
                    self.pos[posClass]).cuda(), choice(negClass).cuda()
            return choice(self.pos[posClass]).cpu(), choice(
                self.pos[posClass]).cpu(), choice(negClass).cpu()
        else:
            if cuda.is_available():
                return choice(self.pos[0]).cuda(), choice(
                    self.pos[0]).cuda(), choice(self.neg).cuda()
            return choice(self.pos[0]).cpu(), choice(
                self.pos[0]).cpu(), choice(self.neg).cpu()

    def __len__(self):
        return self.__len
