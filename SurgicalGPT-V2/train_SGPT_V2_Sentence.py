import os
import argparse

from tqdm import tqdm

# import numpy as np
# from torch import nn
# import torch.nn.functional as F

from torch import optim
from torch.nn import CrossEntropyLoss
from torch.utils.data  import DataLoader
import torch.backends.cudnn as cudnn

from transformers import GPT2Tokenizer #, AdamW

from utils import *
from gpt2utils import generate_sample
from dataloader.dataloaderGPT2Sentence import *
from model.EFGPT2Sentence import EFVLEGPT2RS18Sentence

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from torch.nn.utils.rnn import pack_padded_sequence
from nltk.translate.bleu_score import corpus_bleu

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
    
    total_loss = AverageMeter()  
        
    for i, (_, visual_features, questions, answers) in enumerate(tqdm(train_dataloader),0):
        
        # prepare questions and answers
        question_list = []
        answer_list = []
        for question in questions: question_list.append(question)
        for answer in answers: answer_list.append(answer)
        
        if args.model_ver == 'efvlegpt2rs18' or args.model_ver == "efvlegpt2Swin" or args.model_ver == 'efvlegpt2ViT':  
            question_inputs = tokenizer(question_list, padding="max_length",max_length= args.question_len, return_tensors="pt")
            answer_inputs = tokenizer(answer_list, padding="max_length",max_length= args.answer_len, return_tensors="pt")
            

        # Visual features
        if args.model_ver == "efvlegpt2Swin" or args.model_ver == 'efvlegpt2ViT':
            visual_features['pixel_values'] = torch.squeeze(visual_features['pixel_values'],1)
        else:
            visual_features = visual_features.to(device)
            visual_len = 80

        # model forward(question, img, answer)
        logits = model(question_inputs, visual_features, answer_inputs)[0]
        
        # only consider loss on reference summary just like seq2seq models
        idx = args.question_len + visual_len
        shift_logits = logits[..., idx:-1, :].contiguous()
        shift_labels = answer_inputs['input_ids'][..., 1:].contiguous() # 1 because answer has '<|sep|>' in front
        shift_labels = shift_labels.to(device)

        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss.update(loss.item())
                      
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        # predict_token_number =  [50 - a for a in idx]
        # losses.update(loss.item(), sum(predict_token_number))
        # epoch_loss += loss.item()
        # scheduler.step()  # Update learning rate schedule
        # model.zero_grad()
    
    print("Epoch: {}/{} Loss: {:.6f} AVG_Loss: {:.6f}".format(epoch, args.epochs, total_loss.val, total_loss.avg))
    

def validate(args, val_loader, model, criterion, epoch, tokenizer, device, save_output = False):
    
    references = []
    hypotheses = []

    model.eval()
    total_loss = AverageMeter()  
        
    with torch.no_grad():
        for i, (_, visual_features, questions, answers) in enumerate(tqdm(val_loader),0):
        
            # prepare questions and answers
            question_list = []
            answer_list = []
            for question in questions: question_list.append(question)
            for answer in answers: answer_list.append(answer)
            
            if args.model_ver == 'efvlegpt2rs18' or args.model_ver == "efvlegpt2Swin" or args.model_ver == 'efvlegpt2ViT':  
                question_inputs = tokenizer(question_list, padding="max_length",max_length= args.question_len, return_tensors="pt")
                answer_inputs = tokenizer(answer_list, padding="max_length",max_length= args.answer_len, return_tensors="pt")
                

            # Visual features
            if args.model_ver == "efvlegpt2Swin" or args.model_ver == 'efvlegpt2ViT':
                visual_features['pixel_values'] = torch.squeeze(visual_features['pixel_values'],1)
            else:
                visual_features = visual_features.to(device)
                visual_len = 80

            # model forward(question, img, answer)
            logits = model(question_inputs, visual_features, answer_inputs)[0]

            
            # only consider loss on reference summary just like seq2seq models
            idx = args.question_len + visual_len
            shift_logits = logits[..., idx:-1, :].contiguous()
            shift_labels = answer_inputs['input_ids'][..., 1:].contiguous() # 1 because answer has '<|sep|>' in front
            
            # copy for logits and labels for sentence decoding and blue-4 score calculation
            logits_copy = logits.clone() 
            shift_labels_copy = shift_labels.clone()

            # loss calculation
            shift_labels = shift_labels.to(device)
            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss.update(loss.item())

            # references    - Ground truth answer
            answer_GT_dec = tokenizer.batch_decode(shift_labels_copy, skip_special_tokens= True)
            for answer_GT_dec_i in answer_GT_dec: references.append([answer_GT_dec_i.split()])
            # print(references)

            # Hypotheses - predicted answer
            _, answer_Gen_id = torch.max(logits_copy, dim=2)
            answer_Gen_dec = tokenizer.batch_decode(answer_Gen_id, skip_special_tokens= True)
            for answer_Gen_dec_i in answer_Gen_dec: hypotheses.append(answer_Gen_dec_i.split())
            # print(hypotheses)

            
        # Calculate BLEU1~4
        metrics = {}
        metrics["Bleu_1"] = corpus_bleu(references, hypotheses, weights=(1.00, 0.00, 0.00, 0.00))
        metrics["Bleu_2"] = corpus_bleu(references, hypotheses, weights=(0.50, 0.50, 0.00, 0.00))
        metrics["Bleu_3"] = corpus_bleu(references, hypotheses, weights=(0.33, 0.33, 0.33, 0.00))
        metrics["Bleu_4"] = corpus_bleu(references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25))

        print("Epoch: {}/{} EVA LOSS: {:.6f} BLEU-1 {:.6f} BLEU2 {:.6f} BLEU3 {:.6f} BLEU-4 {:.6f}".format
          (epoch, args.epochs, total_loss.avg, metrics["Bleu_1"],  metrics["Bleu_2"],  metrics["Bleu_3"],  metrics["Bleu_4"]))
                      
    return metrics



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='VisualQuestionAnswerClassification')
    
    # Training parameters
    parser.add_argument('--epochs',         type=int,   default=80,                                 help='number of epochs to train for (if early stopping is not triggered).') #80, 26
    parser.add_argument('--batch_size',     type=int,   default=50,                                 help='batch_size')
    parser.add_argument('--workers',        type=int,   default=1,                                  help='for data-loading; right now, only 1 works with h5pys.')
    
    # existing checkpoint
    parser.add_argument('--checkpoint',     default=None,                                           help='path to checkpoint, None if none.')
    
    parser.add_argument('--lr',             type=float, default=0.00005,                            help=' 0.00001, 0.000005')
    parser.add_argument('--checkpoint_dir', default= 'checkpoints/efvlegpt2rs18/m18/v3_p_qf_',      help='m18/c80')
    parser.add_argument('--dataset_type',   default= 'm18',                                         help='m18/c80')
    parser.add_argument('--tokenizer_ver',  default= 'gpt2v1',                                      help='btv2/btv3/gpt2v1')
    parser.add_argument('--model_subver',   default= 'v3',                                          help='V0,v1/v2/v3/v4')
    parser.add_argument('--question_len',   default= 25,                                            help='25')
    parser.add_argument('--answer_len',     default= 35,                                            help='25')
    parser.add_argument('--model_ver',      default= 'efvlegpt2rs18',                               help='efvlegpt2rs18/efvlegpt2Swin/"')  #vrvb/gpt2rs18/gpt2ViT/gpt2Swin/biogpt2rs18/vilgpt2vqa/efgpt2rs18gr/efvlegpt2Swingr
    parser.add_argument('--vis_pos_emb',    default= 'pos',                                         help='None, zeroes, pos')
    parser.add_argument('--patch_size',     default= 5,                                             help='1/2/3/4/5')
    
    parser.add_argument('--validate',       default=False,                                          help='When only validation required False/True')
    
    args = parser.parse_args()


    '''
    EFVLEGPT2RS18Sentence:
        v0: visual embedding : Default patch1 + embedding form VB + GPT2 decoder
        v1: visual embedding : Default patch1 + from nn.linear    + GPT2 decoder
        v2: visual embedding : visual patches + embedding form VB + GPT2 decoder
        v3: visual embedding : visual patches + from nn.linear    + GPT2 decoder
    EFVLEGPT2SwinSentence:
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
        tokenizer_length = len(tokenizer)
        # data location
        train_seq = [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]
        val_seq = [1, 5, 16]
        # train_seq = [1, 2, 3, 5, 6, 7, 9, 10, 14, 15, 16]
        # val_seq = [4, 11, 12]
        folder_head = '../dataset/EndoVis-18-VQA/seq_'
        folder_tail = '/vqa2/Sentence/*.txt'
        
        # dataloader
        if args.model_ver == 'efvlegpt2rs18' or args.model_ver == "efvlegpt2Swin" or args.model_ver == 'efvlegpt2ViT':
            
            train_dataset = EndoVis18VQAGPTSentence(train_seq, folder_head, folder_tail, model_ver=args.model_ver)
            train_dataloader = DataLoader(dataset=train_dataset, batch_size= args.batch_size, shuffle=True, num_workers=8)
            val_dataset = EndoVis18VQAGPTSentence(val_seq, folder_head, folder_tail, model_ver=args.model_ver)
            val_dataloader = DataLoader(dataset=val_dataset, batch_size= args.batch_size, shuffle=False, num_workers=8)        

    # elif args.dataset_type == 'c80':
    #     '''
    #     Train and test for cholec dataset
    #     '''
    #     # tokenizer
    #     if args.tokenizer_ver == 'gpt2v1':
    #         tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    #         tokenizer.pad_token = tokenizer.eos_token
        
    #     # dataloader
    #     train_seq = [1, 2, 3, 4, 6, 7, 8, 9, 10, 13, 14, 15, 16, 18, 20, 21, 22, 23, 24, 25, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    #     val_seq = [5, 11, 12, 17, 19, 26, 27, 31]
    #     folder_head = 'dataset/Cholec80-VQA/Classification/'
    #     folder_tail = '/*.txt'

    #     if args.model_ver == 'efvlegpt2rs18' or args.model_ver == "efvlegpt2Swin" or args.model_ver == 'efvlegpt2ViT':
            
    #         train_dataset = Cholec80VQAGPTClassification(train_seq, folder_head, folder_tail, model_ver=args.model_ver)
    #         train_dataloader = DataLoader(dataset=train_dataset, batch_size= args.batch_size, shuffle=True, num_workers=8)
    #         val_dataset = Cholec80VQAGPTClassification(val_seq, folder_head, folder_tail, model_ver=args.model_ver)
    #         val_dataloader = DataLoader(dataset=val_dataset, batch_size= args.batch_size, shuffle=False, num_workers=8)



    # elif args.dataset_type == 'psi':
    #     '''
    #     Train and test for psi-ava-vqa dataset
    #     '''
    #     # tokenizer
    #     if args.tokenizer_ver == 'btv2': tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    #     elif args.tokenizer_ver == 'btv3': tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    #     elif args.tokenizer_ver == 'gpt2v1':
    #         tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    #         tokenizer.pad_token = tokenizer.eos_token
        
    #     # dataloader
    #     train_seq  =[
    #                     "dataset/PSI-AVA-VQA/Train/C1_location.txt", 
    #                     "dataset/PSI-AVA-VQA/Train/C3_phase.txt", 
    #                     "dataset/PSI-AVA-VQA/Train/C4_step.txt"
    #                 ] 
    #     val_seq    =[
    #                     "dataset/PSI-AVA-VQA/Val/C1_location.txt",
    #                     "dataset/PS I-AVA-VQA/Val/C3_phase.txt",
    #                     "dataset/PSI-AVA-VQA/Val/C4_step.txt"
    #                 ] 

    #     # dataloader
    #     if args.model_ver == 'efvlegpt2rs18' or args.model_ver == "efvlegpt2Swin" or args.model_ver == 'efvlegpt2ViT':
            
    #         train_dataset = PSIAVAVQAGPTClassification(train_seq, model_ver=args.model_ver)
    #         train_dataloader = DataLoader(dataset=train_dataset, batch_size= args.batch_size, shuffle=True, num_workers=8)
    #         val_dataset = PSIAVAVQAGPTClassification(val_seq, model_ver=args.model_ver)
    #         val_dataloader = DataLoader(dataset=val_dataset, batch_size= args.batch_size, shuffle=False, num_workers=8)

    

    # Initialize / load checkpoint
    if args.checkpoint is None:
        if args.model_ver == 'efvlegpt2rs18':
            model = EFVLEGPT2RS18Sentence(model_subver = args.model_subver, tokenizer_len=len(tokenizer), vis_pos_emb = args.vis_pos_emb)
        # elif args.model_ver == 'efvlegpt2Swin':
        #     model = EFVLEGPT2SwinClassification(num_class = args.num_class, model_subver = args.model_subver, vis_pos_emb = args.vis_pos_emb)
        # elif args.model_ver == 'efvlegpt2ViT':
        #     model = EFVLEGPT2ViTClassification(num_class = args.num_class, model_subver = args.model_subver, vis_pos_emb = args.vis_pos_emb)
        
        print(model)
        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    else:
        checkpoint = torch.load(args.checkpoint, map_location=str(device))
        start_epoch = checkpoint['epoch']
        epochs_since_improvement = checkpoint['epochs_since_improvement']
        # best_Acc = checkpoint['Acc']
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
    criterion = CrossEntropyLoss().to(device)

    for epoch in range(start_epoch, args.epochs):

        if epochs_since_improvement > 0 and epochs_since_improvement % 5 == 0:
            adjust_learning_rate(optimizer, 0.8)
            
        # train
        train(args, train_dataloader=train_dataloader, model = model, criterion=criterion, optimizer=optimizer, epoch=epoch, tokenizer = tokenizer, device = device)

        # validation
        metrics = validate(args, val_loader=val_dataloader, model = model, criterion=criterion, epoch=epoch, tokenizer = tokenizer, device = device)

        if metrics["Bleu_4"] >= best_results[0]:
            epochs_since_improvement = 0
                
            best_results[0] = metrics["Bleu_4"]
            best_epoch[0] = epoch
            # print('Best epoch: %d | Best acc: %.6f' %(best_epoch[0], best_results[0]))
            save_clf_checkpoint(args.checkpoint_dir, epoch, epochs_since_improvement, model, optimizer, best_results[0])
                        
        else:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))