import os
import glob
import h5py
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import ViTFeatureExtractor, AutoFeatureExtractor


'''
EndoVis18 classification dataloader for GPT2 + ResNet18
'''
class EndoVis18VQAGPTClassification(Dataset):
    '''
    	seq: train_seq  = [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]
    	     val_seq    = [1, 5, 16]
    	folder_head     = 'dataset/EndoVis-18-VQA/seq_'
    	folder_tail     = '/vqa/Classification/*.txt'
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


'''
Cholec80 classification dataloader GPT
'''
class Cholec80VQAGPTClassification(Dataset):
    '''
    	seq: train_seq  = ['1','2','3','4','6','7','8','9','10','13','14','15','16','18','20',
                          '21','22','23','24','25','28','29','30','32','33','34','35','36','37','38','39','40']
             val_seq    = ['5','11','12','17','19','26','27','31']
    	folder_head     = 'dataset/Cholec80-VQA/Classification/'
	    folder_tail     = '/*.txt'
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
        new_filenames = []
        for filename in filenames:
            frame_num = int(filename.split('/')[-1].split('.')[0].split('_')[0])
            if frame_num % 100 == 0: new_filenames.append(filename)
    		
        self.vqas = []
        for file in new_filenames:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            for line in lines: self.vqas.append([file, line])
        print('Total files: %d | Total question: %.d' %(len(filenames), len(self.vqas)))
        
        # labels
        self.labels = ['no', 'calot triangle dissection', 'yes', '1', '2', 'gallbladder dissection', 
                        'clipping cutting', 'gallbladder retraction', '0', 'cleaning coagulation', 
                        'gallbladder packaging', 'preparation', '3']
        
    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        
        # img
        loc = self.vqas[idx][0].split('/')
        
        # img
        img_loc = os.path.join(loc[0],loc[1], 'cropped_image',loc[3],loc[-1].split('_')[0]+'.png')
        if self.transform: 
            img = Image.open(img_loc)
            img = self.transform(img)
        else: 
            img = self.image_processor(Image.open(img_loc), return_tensors="pt")
            
        # question and answer
        question = self.vqas[idx][1].split('|')[0]
        label = self.labels.index(str(self.vqas[idx][1].split('|')[1]))

        return os.path.join(loc[0],loc[1], 'cropped_image',loc[3],loc[-1].split('_')[0]+'.png'), img, question, label



'''
PSIAVA classification dataloader GPT
'''
class PSIAVAVQAGPTClassification(Dataset):
    '''
    	seq: train_seq  =   [
                            "dataset/PSI-AVA-VQA/Train/C1_location.txt", 
                            "dataset/PSI-AVA-VQA/Train/C2_action.txt", 
                            "dataset/PSI-AVA-VQA/Train/C3_phase.txt", 
                            "dataset/PSI-AVA-VQA/Train/C4_step.txt"
                            ]
             val_seq    =   [
                            "dataset/PSI-AVA-VQA/Val/C1_location.txt",
                            "dataset/PSI-AVA-VQA/Val/C2_action.txt",
                            "dataset/PSI-AVA-VQA/Val/C3_phase.txt",
                            "dataset/PSI-AVA-VQA/Val/C4_step.txt"
                            ]
    '''
    def __init__(self, seq, model_ver = None, transform=None):

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
        for curr_seq in seq: 
            filenames = filenames + glob.glob((curr_seq))
        
        self.vqas = []
        for file in filenames:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            for line in lines: self.vqas.append([file, line])
                    
        print('Total classes: %d | Total question: %.d' %(len(filenames), len(self.vqas)))
        
        # labels
        self.labels =  ["top left", "top right", "bottom left", "bottom right", #location
                "Complejo_venoso_dorsal", "Control_Pediculos", "Espacio_de_Retzius", "Fascia_Denonvilliers","Id_Cuello_Vesical", 
                "LPAD", "LPAI", "Rec_Cuello_Vesical", "Separacion_Prostata_Uretra", "Tiempo_muerto", "Vesículas_Seminales", #phase
                "Anudar", "Clip_Pediculos", "Corte", "Corte_Prostata", "Corte_Vejiga", "Diseccion_Denon", "Diseccion_Ganglios_Iliacos",
                "Diseccion_Ganglios_Obturadores", "Diseccion_Prevesical", "Diseccion_Prostata", "Diseccion_Seminal", "Empacar_Ganglios",  
                "Empacar_Prostata", "Halar_sutura", "Id_Vena_Arteria_Iliaca", "Pasar_Aguja_Cuello", "Pasar_Aguja_Cvdp", "Pasar_Aguja_Uretra", 
                "Succion","Sujetar_Prostata" ]#
                
        
    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        
        # img
        vqa_data = self.vqas[idx][1].split('|')
        
        # img
        img_loc = os.path.join('dataset/PSI-AVA-VQA/keyframes',vqa_data[0])
        if self.transform: 
            img = Image.open(img_loc)
            img = self.transform(img)
        else: 
            img = self.image_processor(Image.open(img_loc), return_tensors="pt")
            
        # question and answer
        question = vqa_data[1]
        label = self.labels.index(str(vqa_data[2]))

        return vqa_data[0], img, question, label