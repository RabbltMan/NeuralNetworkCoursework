from numpy import array
from torch import from_numpy, randn, Tensor, cuda
from torch.utils.data import Dataset
from typing import List
from PIL import Image
from os import listdir
from random import choice, choices


class TripletDataset(Dataset):

    def __init__(self, posPathList: List[str], size: int = 500) -> None:
        self.pos = []
        for d in posPathList:
            self.pos.append([
                from_numpy(array(Image.open(f"{d}/{f}"))).permute(
                    2, 0, 1).float() / 255.0
                for f in choices(listdir(d), k=size)
            ])
        self.__len = size * 2

    def __getitem__(self, index):
        if len(self.pos) > 1:
            posClass: int = choice([i for i in range(len(self.pos))])
            negClass: int = choice([i for i in range(len(self.pos))if i != posClass])
            A: Tensor = choice(self.pos[posClass])
            P: Tensor = choice(self.pos[posClass])
            N: Tensor = choice(self.pos[negClass])
            yp: Tensor = from_numpy(array([posClass])).squeeze().long()
            yn: Tensor = from_numpy(array([negClass])).squeeze().long()
            if cuda.is_available():
                return A.cuda(), P.cuda(), N.cuda(), yp.cuda(), yn.cuda()
            return A.cpu(), P.cpu(), N.cpu(), yp.cpu(), yn.cpu()
        else:
            A: Tensor = choice(self.pos[0])
            P: Tensor = choice(self.pos[0])
            N: Tensor = randn([3, 160, 160], requires_grad=True)
            yp: Tensor = from_numpy(array([0])).squeeze().long()
            yn: Tensor = from_numpy(array([0])).squeeze().long()
            if cuda.is_available():
                return A.cuda(), P.cuda(), N.cuda(), yp.cuda(), yn.cuda()
            return A.cpu(), P.cpu(), N.cpu(), yp.cpu(), yn.cpu()

    def __len__(self):
        return self.__len


if __name__ == "__main__":
    prp = "./FacialRecognition/.faces/"
    TripletDataset([prp + d for d in listdir(prp)[:-1]])
