import os
import argparse
import pandas as pd
from PIL import Image
from lib2to3.pytree import convert

from torch import nn
from torch import optim
import torch.utils.data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data  import DataLoader
from transformers import BertTokenizer, GPT2Tokenizer

from utils import *
from dataloader.dataloaderGPT2Classification import *
from model.EFGPT2Classification import EFVLEGPT2RS18Classification, EFVLEGPT2SwinClassification, EFVLEGPT2ViTClassification


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


class config_emb:
    visual_embedding_dim = 2048
    vocab_size = 30522
    type_vocab_size = 2
    pad_token_id = 1
    hidden_size = 768
    max_position_embeddings = 512
    layer_norm_eps = 1e-12
    hidden_dropout_prob = 0.1
    special_visual_initialize = True


'''
Seed randoms
'''
def seed_everything(seed=27):
    '''
    Set random seed for reproducible experiments
    Inputs: seed number 
    '''
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args, train_dataloader, model, criterion, optimizer, epoch, tokenizer, device):
    
    model.train()
    
    total_loss = 0.0    
    label_true = None
    label_pred = None
    label_score = None
        
    for i, (_, v_f, q, labels) in enumerate(train_dataloader,0):
        # print('train')
        # prepare questions
        questions = []
        for question in q: questions.append(question)
        
        if args.model_ver == 'efvlegpt2rs18' or args.model_ver == "efvlegpt2Swin" or args.model_ver == 'efvlegpt2ViT':  
            inputs = tokenizer(questions, padding="max_length",max_length= args.question_len, return_tensors="pt")

        # Visual features
        if args.model_ver == "efvlegpt2Swin" or args.model_ver == 'efvlegpt2ViT':
            visual_features = v_f
            visual_features['pixel_values'] = torch.squeeze(visual_features['pixel_values'],1)
        else:
            visual_features = v_f.to(device)

        # labels
        labels = labels.to(device)

        # model forward pass
        outputs = model(inputs, visual_features)
        
        # loss
        loss = criterion(outputs, labels)

        # zero the parameter gradients
        optimizer.zero_grad()        
        loss.backward()
        optimizer.step()

        # print statistics
        total_loss += loss.item()
        
        scores, predicted = torch.max(F.softmax(outputs, dim=1).data, 1)    
        label_true = labels.data.cpu() if label_true == None else torch.cat((label_true, labels.data.cpu()), 0)
        label_pred = predicted.data.cpu() if label_pred == None else torch.cat((label_pred, predicted.data.cpu()), 0)
        label_score = scores.data.cpu() if label_score == None else torch.cat((label_score, scores.data.cpu()), 0)

    # loss and acc
    acc, c_acc = calc_acc(label_true, label_pred), calc_classwise_acc(label_true, label_pred)
    precision, recall, fscore = calc_precision_recall_fscore(label_true, label_pred)
    print('Train: epoch: %d loss: %.6f | Acc: %.6f | Precision: %.6f | Recall: %.6f | FScore: %.6f' %(epoch, total_loss, acc, precision, recall, fscore))
    return acc


def validate(args, val_loader, model, criterion, epoch, tokenizer, device, save_output = False):
    
    model.eval()

    total_loss = 0.0    
    label_true = None
    label_pred = None
    label_score = None
    file_names = list()
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for i, (file_name, v_f, q, labels) in enumerate(val_loader,0):
            
            # prepare questions
            questions = []
            for question in q: questions.append(question)

            if args.model_ver == 'efvlegpt2rs18' or args.model_ver == "efvlegpt2Swin" or args.model_ver == 'efvlegpt2ViT':
                
                # inputs = tokenizer(questions, padding=True, truncation=True, return_tensors="pt",)
                inputs = tokenizer(questions, padding="max_length",max_length=args.question_len, return_tensors="pt")

            # GPU / CPU
            # Visual features
            if args.model_ver == "efvlegpt2Swin" or args.model_ver == 'efvlegpt2ViT':
                     
                visual_features = v_f
                visual_features['pixel_values'] = torch.squeeze(visual_features['pixel_values'],1)
            else:
                visual_features = v_f.to(device)
            
            # label
            labels = labels.to(device)
            
            # model forward pass
            outputs = model(inputs, visual_features)
    
            # loss
            loss = criterion(outputs,labels)

            total_loss += loss.item()
        
            scores, predicted = torch.max(F.softmax(outputs, dim=1).data, 1)    
            label_true = labels.data.cpu() if label_true == None else torch.cat((label_true, labels.data.cpu()), 0)
            label_pred = predicted.data.cpu() if label_pred == None else torch.cat((label_pred, predicted.data.cpu()), 0)
            label_score = scores.data.cpu() if label_score == None else torch.cat((label_score, scores.data.cpu()), 0)
            for f in file_name: file_names.append(f)
            
    acc = calc_acc(label_true, label_pred) 
    c_acc = 0.0
    # c_acc = calc_classwise_acc(label_true, label_pred)
    precision, recall, fscore = calc_precision_recall_fscore(label_true, label_pred)

    print('Test: epoch: %d loss: %.6f | Acc: %.6f | Precision: %.6f | Recall: %.6f | FScore: %.6f' %(epoch, total_loss, acc, precision, recall, fscore))

    if save_output:
        '''
            Saving predictions
        '''
        if os.path.exists(args.checkpoint_dir + 'text_files') == False:
            os.mkdir(args.checkpoint_dir + 'text_files' ) 
        file1 = open(args.checkpoint_dir + 'text_files/labels.txt', 'w')
        file1.write(str(label_true))
        file1.close()

        file1 = open(args.checkpoint_dir + 'text_files/predictions.txt', 'w')
        file1.write(str(label_pred))
        file1.close()

        if args.dataset_type == 'm18':
            convert_arr = ['kidney', 'Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation',
                            'Tool_Manipulation', 'Cutting', 'Cauterization', 'Suction', 
                            'Looping', 'Suturing', 'Clipping', 'Staple', 'Ultrasound_Sensing',
                            'left-top', 'right-top', 'left-bottom', 'right-bottom']
        elif args.dataset_type == 'c80':
            convert_arr = ['no', 'calot triangle dissection', 'yes', '1', '2', 'gallbladder dissection', 
                            'clipping cutting', 'gallbladder retraction', '0', 'cleaning coagulation', 
                            'gallbladder packaging', 'preparation', '3']
        
        df = pd.DataFrame(columns=["Img", "Ground Truth", "Prediction"])
        for i in range(len(label_true)):
            df = df.append({'Img': file_names[i], 'Ground Truth': convert_arr[label_true[i]], 'Prediction': convert_arr[label_pred[i]]}, ignore_index=True)
        
        df.to_csv(args.checkpoint_dir + args.checkpoint_dir.split('/')[1] + '_' + args.checkpoint_dir.split('/')[2] + '_eval.csv')
    
    return (acc, c_acc, precision, recall, fscore)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VisualQuestionAnswerClassification')
    
    # Training parameters
    parser.add_argument('--epochs',         type=int,   default=80,                                 help='number of epochs to train for (if early stopping is not triggered).') #80, 26
    parser.add_argument('--batch_size',     type=int,   default=64,                                 help='batch_size')
    parser.add_argument('--workers',        type=int,   default=1,                                  help='for data-loading; right now, only 1 works with h5pys.')
    parser.add_argument('--print_freq',     type=int,   default=100,                                help='print training/validation stats every __ batches.')
    
    # existing checkpoint
    parser.add_argument('--checkpoint',     default=None,                                           help='path to checkpoint, None if none.')
    
    parser.add_argument('--lr',             type=float, default=0.00001,                            help=' 0.00001, 0.000005')
    parser.add_argument('--checkpoint_dir', default= 'checkpoints/efvlegpt2rs18/m18/v3_z_qf_nmwqw_',     help='m18/c80')
    parser.add_argument('--dataset_type',   default= 'm18',                                         help='m18/c80')
    parser.add_argument('--tokenizer_ver',  default= 'gpt2v1',                                      help='btv2/btv3/gpt2v1')
    parser.add_argument('--model_subver',   default= 'v3',                                          help='V0,v1/v2/v3/v4')
    parser.add_argument('--question_len',   default= 25,                                            help='25')
    parser.add_argument('--model_ver',      default= 'efvlegpt2rs18',                               help='efvlegpt2rs18/efvlegpt2Swin/"')  #vrvb/gpt2rs18/gpt2ViT/gpt2Swin/biogpt2rs18/vilgpt2vqa/efgpt2rs18gr/efvlegpt2Swingr
    parser.add_argument('--vis_pos_emb',   default= 'zeroes',                                           help='None, zeroes, pos')
    parser.add_argument('--patch_size',     default= 5,                                             help='1/2/3/4/5')
    
    parser.add_argument('--num_class',      default= 2,                                             help='25')
    # parser.add_argument('--temporal_size',  default= 1,                                             help='1/2/3/4/5')
    parser.add_argument('--validate',       default=False,                                          help='When only validation required False/True')
    
    args = parser.parse_args()


    '''
    EFVLEGPT2RS18Classification:
        v0: visual embedding : Default patch1 + embedding form VB + GPT2 decoder
        v1: visual embedding : Default patch1 + from nn.linear    + GPT2 decoder
        v2: visual embedding : visual patches + embedding form VB + GPT2 decoder
        v3: visual embedding : visual patches + from nn.linear    + GPT2 decoder
    EFVLEGPT2SwinClassification:
        v0: visual embedding : Default patch1 + embedding form VB + GPT2 decoder
        v1: visual embedding : Default patch1 + GPT2 decoder
    '''

    print(args.model_ver, args.model_subver, args.vis_pos_emb, args.dataset_type, args.lr, args.checkpoint_dir)
    
    seed_everything()
    
    # GPU or CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
    cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
    print('device =', device)

    # best model initialize
    start_epoch = 1
    best_epoch = [0]
    best_results = [0.0]
    epochs_since_improvement = 0


    if args.dataset_type == 'm18':
        '''
        Train and test dataloader for EndoVis18
        '''
        # tokenizer
        tokenizer = None
        if args.tokenizer_ver == 'gpt2v1':
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        
        # data location
        train_seq = [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]
        val_seq = [1, 5, 16]
        # train_seq = [1, 2, 3, 5, 6, 7, 9, 10, 14, 15, 16]
        # val_seq = [4, 11, 12]
        folder_head = '../dataset/EndoVis-18-VQA/seq_'
        folder_tail = '/vqa2/Classification/*.txt'
        
        # dataloader
        if args.model_ver == 'efvlegpt2rs18' or args.model_ver == "efvlegpt2Swin" or args.model_ver == 'efvlegpt2ViT':
            train_dataset = EndoVis18VQAGPTClassification(train_seq, folder_head, folder_tail, model_ver=args.model_ver)
            train_dataloader = DataLoader(dataset=train_dataset, batch_size= args.batch_size, shuffle=True, num_workers=8)
            val_dataset = EndoVis18VQAGPTClassification(val_seq, folder_head, folder_tail, model_ver=args.model_ver)
            val_dataloader = DataLoader(dataset=val_dataset, batch_size= args.batch_size, shuffle=False, num_workers=8)

        # num_classes
        args.num_class = 18

    elif args.dataset_type == 'c80':
        '''
        Train and test for cholec dataset
        '''
        # tokenizer
        if args.tokenizer_ver == 'gpt2v1':
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        
        # dataloader
        train_seq = [1, 2, 3, 4, 6, 7, 8, 9, 10, 13, 14, 15, 16, 18, 20, 21, 22, 23, 24, 25, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40]
        val_seq = [5, 11, 12, 17, 19, 26, 27, 31]
        folder_head = 'dataset/Cholec80-VQA/Classification/'
        folder_tail = '/*.txt'

        if args.model_ver == 'efvlegpt2rs18' or args.model_ver == "efvlegpt2Swin" or args.model_ver == 'efvlegpt2ViT':
            
            train_dataset = Cholec80VQAGPTClassification(train_seq, folder_head, folder_tail, model_ver=args.model_ver)
            train_dataloader = DataLoader(dataset=train_dataset, batch_size= args.batch_size, shuffle=True, num_workers=8)
            val_dataset = Cholec80VQAGPTClassification(val_seq, folder_head, folder_tail, model_ver=args.model_ver)
            val_dataloader = DataLoader(dataset=val_dataset, batch_size= args.batch_size, shuffle=False, num_workers=8)

        # num_classes
        args.num_class = 13

    elif args.dataset_type == 'psi':
        '''
        Train and test for psi-ava-vqa dataset
        '''
        # tokenizer
        if args.tokenizer_ver == 'btv2': tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif args.tokenizer_ver == 'btv3': tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        elif args.tokenizer_ver == 'gpt2v1':
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
        
        # dataloader
        train_seq  =[
                        "dataset/PSI-AVA-VQA/Train/C1_location.txt", 
                        "dataset/PSI-AVA-VQA/Train/C3_phase.txt", 
                        "dataset/PSI-AVA-VQA/Train/C4_step.txt"
                    ] 
        val_seq    =[
                        "dataset/PSI-AVA-VQA/Val/C1_location.txt",
                        "dataset/PS I-AVA-VQA/Val/C3_phase.txt",
                        "dataset/PSI-AVA-VQA/Val/C4_step.txt"
                    ] 

        # dataloader
        if args.model_ver == 'efvlegpt2rs18' or args.model_ver == "efvlegpt2Swin" or args.model_ver == 'efvlegpt2ViT':
            
            train_dataset = PSIAVAVQAGPTClassification(train_seq, model_ver=args.model_ver)
            train_dataloader = DataLoader(dataset=train_dataset, batch_size= args.batch_size, shuffle=True, num_workers=8)
            val_dataset = PSIAVAVQAGPTClassification(val_seq, model_ver=args.model_ver)
            val_dataloader = DataLoader(dataset=val_dataset, batch_size= args.batch_size, shuffle=False, num_workers=8)

        # num_classes
        args.num_class = 35 #155 #35
    

    # Initialize / load checkpoint
    if args.checkpoint is None:
        if args.model_ver == 'efvlegpt2rs18':
            model = EFVLEGPT2RS18Classification(num_class = args.num_class, model_subver = args.model_subver, vis_pos_emb = args.vis_pos_emb)
        elif args.model_ver == 'efvlegpt2Swin':
            model = EFVLEGPT2SwinClassification(num_class = args.num_class, model_subver = args.model_subver, vis_pos_emb = args.vis_pos_emb)
        elif args.model_ver == 'efvlegpt2ViT':
            model = EFVLEGPT2ViTClassification(num_class = args.num_class, model_subver = args.model_subver, vis_pos_emb = args.vis_pos_emb)
        
        print(model)
        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    else:
        checkpoint = torch.load(args.checkpoint, map_location=str(device))
        start_epoch = checkpoint['epoch']
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        best_Acc = checkpoint['Acc']
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']
        final_args = checkpoint['final_args']
        for key in final_args.keys(): args.__setattr__(key, final_args[key])


    # Move to GPU, if available
    model = model.to(device)
    # print(final_args)    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('model params: ', pytorch_total_params)
    # print(model)

    # Loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # validation
    if args.validate:
        test_acc, test_c_acc, test_precision, test_recall, test_fscore = validate(args, val_loader=val_dataloader, model = model, criterion=criterion, epoch=(args.epochs-1), tokenizer = tokenizer, device = device)
    else:     
        for epoch in range(start_epoch, args.epochs):

            if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
                adjust_learning_rate(optimizer, 0.8)
            
            # train
            train_acc = train(args, train_dataloader=train_dataloader, model = model, criterion=criterion, optimizer=optimizer, epoch=epoch, tokenizer = tokenizer, device = device)

            # validation
            test_acc, test_c_acc, test_precision, test_recall, test_fscore = validate(args, val_loader=val_dataloader, model = model, criterion=criterion, epoch=epoch, tokenizer = tokenizer, device = device)
            
            if test_acc >= best_results[0]:
                epochs_since_improvement = 0
                
                best_results[0] = test_acc
                best_epoch[0] = epoch
                # print('Best epoch: %d | Best acc: %.6f' %(best_epoch[0], best_results[0]))
                save_clf_checkpoint(args.checkpoint_dir, epoch, epochs_since_improvement, model, optimizer, best_results[0])
                        
            else:
                epochs_since_improvement += 1
                print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
            
            if train_acc >= 1.0: break
