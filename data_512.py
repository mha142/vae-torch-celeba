from torch.utils.data import Dataset
import torchvision.transforms as transforms
from pathlib import Path
#from skimage import io 
import os
#import cv2
from PIL import Image
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


#import sys
#sys.path.append('../VAE/ffhq52000/')

class FFHQ_Data(Dataset):
    #def __init__(self, data_dir: Path=Path('./facekit_data'), transform:Optional[Callable]=None) -> None:
    def __init__(self, data_dir: Path, transform=None):
        super().__init__()
        self.root_dir = data_dir
        self.file_list = os.listdir(data_dir)
        self.transform = transform 
        
    def __len__(self):
        return len(self.file_list)
        

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        img_path = os.path.join(self.root_dir, file_name)
        #img_path = os.path.join(self.root_dir, index)
        #print(img_path)
        #image = io.imread(img_path)
        #image_cv2 = cv2.imread(img_path)#this will read the image as a blue image beacuse it is BGR not RGB  lik pillow library 
        #image_cv2 = cv2.resize(image_cv2, dsize=(128,128), interpolation=cv2.INTER_LINEAR)
        image = Image.open(img_path)

        image = image.resize((512, 512), Image.Resampling.LANCZOS)


        if self.transform is not None:
            image = self.transform(image)
            #image = self.transform(image).unsqueeze(0) 

        #check if the image is between 0 and 1
        #print(image.max())    
        
    
        return image


if __name__ == "__main__":
    p = Path('../VAE/ffhq52000/')
    ffhq_data = FFHQ_Data(data_dir= p, transform= transforms.ToTensor())
    data_length = ffhq_data.__len__() 
    print(f'no. of images in the entire dataset is:', data_length)
    for i in range(data_length):
        ffhq_data.__getitem__(i)
        break