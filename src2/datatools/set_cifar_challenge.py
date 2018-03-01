import torch.utils.data as data
from PIL import Image
import numpy as np
class Dataset(data.Dataset):
    def __init__(self, data, labels, transform ):
        self.data = data
        self.labels = labels
        self.transform=transform
        assert(len(data) == len(labels))

    def __getitem__(self, idx):
        np_array=np.transpose(np.reshape(self.data[idx,:],[ 32,32,3], order="F" ),[1,0,2] )
        img=Image.fromarray(np_array)
        if self.transform is not None:
            img=self.transform(img)
        return img, self.labels[idx]

    def __len__(self):
        return len(self.labels)

    def split(self, index):
        return Dataset(self.data[:index,:], self.labels[:index], self.transform), Dataset(self.data[index:,:], self.labels[index:], self.transform )

    def shuffle(self):
        p=np.random.permutation(len(self.data))
        self.data = self.data[p]
        self.labels = np.asarray(self.labels)[p].tolist()

def make_train_val_datasets(data,labels,index, transform, shuf=False):
    combined=Dataset(data,labels,transform)
    if shuf:
        combined.shuffle()
    val, train= combined.split(index)
    return train, val


def make_collater(args):
    def collater():
        pass
    pass

# from http://corochann.com/cifar-10-cifar-100-dataset-introduction-1258.html
CIFAR100_LABELS_LIST = [
            'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
                'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
                    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
                        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
                            'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
                                'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
                                    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
                                        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
                                            'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
                                                'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
                                                    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
                                                        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
                                                            'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
                                                                'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
                                                                    'worm'
                      ]
 
