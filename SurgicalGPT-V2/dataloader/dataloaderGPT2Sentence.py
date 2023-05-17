import os
import glob
import h5py
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import ViTFeatureExtractor, AutoFeatureExtractor



'''
To do:
dataloader
validation
blue score and sentence eval
'''
'''
EndoVis18 Sentence dataloader for GPT2 + ResNet18
'''
class EndoVis18VQAGPTSentence(Dataset):
    '''
    	seq: train_seq  = [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]
    	     val_seq    = [1, 5, 16]
    	folder_head     = 'dataset/EndoVis-18-VQA/seq_'
    	folder_tail     = '/vqa2/Sentence/*.txt'
    '''
    def __init__(self, seq, folder_head, folder_tail, model_ver = None, transform=None):
        
        if model_ver == "efvlegpt2ViT": 
            self.transform = None
            self.image_processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        elif model_ver == "efvlegpt2Swin": 
            self.transform = None
            self.image_processor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        elif transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                                    transforms.Resize((300,256)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                    ])
        

        # files, question and answers
        filenames = []
        for curr_seq in seq: filenames = filenames + glob.glob(folder_head + str(curr_seq) + folder_tail)
        self.vqas = []
        for file in filenames:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            for line in lines: 
                q_s, ans = line.split('|')
                q_s = q_s.split('&')
                for q in q_s:
                    q_a = q+'|'+ans
                    self.vqas.append([file, q_a])
        print('Total files: %d | Total question: %.d' %(len(filenames), len(self.vqas)))
        
        # Labels
        self.labels = ['kidney', 'Idle', 'Grasping', 'Retraction', 'Tissue Manipulation',
                        'Tool Manipulation', 'Cutting', 'Cauterization', 'Suction', 
                        'Looping', 'Suturing', 'Clipping', 'Staple', 'Ultrasound Sensing',
                        'left-top', 'right-top', 'left-bottom', 'right-bottom']
        
    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):        
        loc = self.vqas[idx][0].split('/')

        # img
        img_loc = os.path.join(loc[0],loc[1],loc[2], loc[3], 'left_frames',loc[-1].split('_')[0]+'.png')
        if self.transform: 
            img = Image.open(img_loc)
            img = self.transform(img)
        else: 
            img = self.image_processor(Image.open(img_loc), return_tensors="pt")
            
        # question and answer
        question = self.vqas[idx][1].split('|')[0]
        label = self.labels.index(str(self.vqas[idx][1].split('|')[1]))

        return img_loc, img, question, label
    
# class EndoVis18VQAGPTSEntence(Dataset):

#     def __init__(self, root_dir, ids_file, mode='train',length=None):
#         self.root_dir = root_dir
#         self.tokenizer = add_special_tokens()
        
#         self.idxs = os.listdir(root_dir)
#         self.mode = mode
#         if len == None:
#             self.len = len(self.idxs)
#         else:
#             self.len = length

#     def __len__(self):
#         return self.len

#     def __getitem__(self,idx):

#         if self.mode=='valid':
#             idx = self.idxs[-idx]
#         elif self.mode=='test':
#             idx = self.idxs[-idx-self.len]   # assuming valid and test set of same sizes
#         else:
#             idx = self.idxs[idx]
#         # file_name = os.path.join(self.root_dir,str(idx)+".json")
#         file_name = os.path.join(self.root_dir,str(idx))
#         with open(file_name,'r') as f:
#               data = json.load(f)
#         text = self.tokenizer.encode(self.tokenizer.pad_token)*1024
#         content = data['article'] + self.tokenizer.encode(self.tokenizer.sep_token) + data['abstract']
#         text[:len(content)] = content
#         text = torch.tensor(text)
#         sample = {'article': text, 'sum_idx': len(data['article'])}
#         return sample