import argparse
from tqdm import tqdm
import os
import PIL.Image as Image

import torch

from model import VitNetAug

parser = argparse.ArgumentParser(description='RecVis A3 evaluation script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
parser.add_argument('--model-checkpoint', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='experiment/kaggle.csv', metavar='D',
                    help="name of the output csv file")
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='input batch size for testing (default: 64)')

args = parser.parse_args()

from data import data_transforms_val

test_dir = args.data + '/test_images/mistery_category'

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class TestDataset(torch.utils.data.Dataset):
  'Dataset for testing'
  def __init__(self,dir_path, transform=None):
        self.img_paths = [path for path in os.listdir(dir_path) if 'jpg' in path]
        self.transform = transform
        self.dir_path = dir_path

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.img_paths)

  def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = np.array(pil_loader(self.dir_path + '/' + img_path))
        if self.transform:
            data = self.transform(image = img)['image']
        id = img_path[:-4]
        return data, id
data_transforms_test = A.Compose([
        A.Resize(384, 384),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
                                  
])


def make_kaggle_submission(model,data_path,output_file,batch_size=32):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    test_data = TestDataset(data_path, transform = data_transforms_test)
    test_loader = torch.utils.data.DataLoader(
                    test_data,
                    batch_size=batch_size, 
                    shuffle=False
                    )
    output_array = np.array(['Id','Category']).reshape(2,1)
    for _, (data, id) in enumerate(tqdm(test_loader)):
        output = model(data.to(device))
        pred = output.data.max(1, keepdim=True)[1]
        submissions = np.array([id,pred.cpu().numpy().flatten()])
        output_array = np.hstack((output_array,submissions))
    import pandas as pd
    pd.DataFrame(output_array.T).to_csv(output_file,header=False,index=False)

def __main__():
    model = VitNetAug().load_from_checkpoint(args.model)
    model.eval()
    make_kaggle_submission(model,args.data,args.outfile,args.batch_size)

        


