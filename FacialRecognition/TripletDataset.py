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
            from_numpy(array(Image.open(f"{negPath}{f}").resize(
                (160, 160)))).permute(2, 0, 1).float() / 255.0
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
            A: Tensor = choice(self.pos[posClass])
            P: Tensor = choice(self.pos[posClass])
            N: Tensor = choice(negClass)
            y: Tensor = from_numpy(array([posClass])).squeeze().long()
            if cuda.is_available():
                return A.cuda(), P.cuda(), N.cuda(), y.cuda()
            return A.cpu(), P.cpu(), N.cpu(), y.cpu()
        else:
            A: Tensor = choice(self.pos[0])
            P: Tensor = choice(self.pos[0])
            N: Tensor = choice(self.neg)
            y: Tensor = from_numpy(array([0])).squeeze().long()
            if cuda.is_available():
                return A.cuda(), P.cuda(), N.cuda(), y.cuda()
            return A.cpu(), P.cpu(), N.cpu(), y.cpu()

    def __len__(self):
        return self.__len
