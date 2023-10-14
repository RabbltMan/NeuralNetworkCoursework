from typing import List
from numpy import int8, ndarray
from pandas import read_table
from sklearn.model_selection import train_test_split


class DataSplit(object):

    def __init__(self,
                 floc="./IndoorLocalization/wifi_localization.txt") -> None:
        dataset = read_table(floc, header=None)
        self.__X = dataset.iloc[:, :-1].to_numpy(int8)
        self.__y = dataset.iloc[:, -1].to_numpy(int8)

    def __call__(self, test_size=0.25) -> List[ndarray]:
        return train_test_split(self.__X, self.__y, test_size=test_size)


if __name__ == "__main__":
    DataSplit()()
