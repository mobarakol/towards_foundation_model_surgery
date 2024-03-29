# import math
# import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# from timm.models import create_model
from transformers import  VisualBertConfig, GPT2Config
from transformers import VisualBertModel, GPT2Model, ViTModel, SwinModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


''' Early Fusion GPT with CNN/Transformers'''

class EFVLEGPT2RS18Classification(nn.Module):
    def __init__(self, num_class = 12, model_subver = 'v0', vis_pos_emb = None):
        super(EFVLEGPT2RS18Classification, self).__init__()
        '''
        v0: visual embedding : Default patch1 + embedding form VB + GPT2 decoder
        v1: visual embedding : Default patch1 + from nn.linear    + GPT2 decoder
        v2: visual embedding : visual patches + embedding form VB + GPT2 decoder
        v3: visual embedding : visual patches + from nn.linear    + GPT2 decoder
        '''
        
        self.sub_ver = model_subver
        self.vis_pos_emb = vis_pos_emb
        
        ## image processing
        self.img_feature_extractor = models.resnet18(pretrained=True)
        if self.sub_ver == 'v0' or self.sub_ver =='v1':
            new_fc = nn.Sequential(*list(self.img_feature_extractor.fc.children())[:-1])
            self.img_feature_extractor.fc = new_fc
        elif self.sub_ver == 'v2' or self.sub_ver =='v3':
            self.img_feature_extractor = torch.nn.Sequential(*(list(self.img_feature_extractor.children())[:-2]))
        
        ## Visual_embedding
        if self.sub_ver == 'v0' or self.sub_ver =='v2':
            # visual bert embedding
            VB_config = VisualBertConfig.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
            VB_config.visual_embedding_dim = 512
            visualbert = VisualBertModel(config=VB_config)
            self.visual_embedder = visualbert.embeddings.visual_projection
        elif self.sub_ver == 'v1' or self.sub_ver =='v3':
            self.visual_embedder = nn.Linear(512, 768)

        ## Question_embedding
        # No pre-trained weights for question embedding
        gpt2configuration = GPT2Config()
        question_embedder = GPT2Model(gpt2configuration)
        self.question_embedder = question_embedder.wte

        # pre-trained weights for question embedding
        # question_embedder = GPT2Model.from_pretrained('gpt2')
        # self.question_embedder = question_embedder.wte


        ## GPT2 visual_cotext_aware_decoder
        # No Pre-trained weights for GPT2
        gpt2configuration = GPT2Config()
        self.VCAdecoder = GPT2Model(gpt2configuration)

        # Pre-trained weights for GPT2
        # self.VCAdecoder = GPT2Model.from_pretrained('gpt2')
        
        
        # for name,child in self.VCAdecoder.h.named_children():
        #     child.mlp._modules['act'] = SwiGLU()
        # self.VCAdecoder.wte = nn.Embedding(2, 768)  # later used for position embedding
 
        ## intermediate_layers
        self.intermediate_layer = nn.Linear(768, 512)  #(512+768)
        self.LayerNorm = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.1)

        ## classifier
        self.classifier = nn.Linear(512, num_class)

    def forward(self, input, img):
        
        ## image encoder features
        img_feature = self.img_feature_extractor(img)
        
        if self.sub_ver == 'v0' or self.sub_ver =='v1':
            img_feature = torch.unsqueeze(img_feature, dim=1)
        if self.sub_ver == 'v2'or self.sub_ver =='v3':
            img_feature = torch.permute(torch.flatten(img_feature, start_dim=2),(0,2,1))
        
        
        ## visual Embedding : id type 1, pos: zero / incremental
        visual_embeds = self.visual_embedder(img_feature)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
        visual_attention_mask = visual_attention_mask.to(device)

        if self.vis_pos_emb == 'zeroes':
            visual_id_type = torch.ones(*visual_embeds.size()[:-1], dtype=torch.long, device=device)
            visual_position_id = torch.zeros(*visual_embeds.size()[:-1], dtype=torch.long, device=device)
        elif self.vis_pos_emb == 'pos':
            visual_id_type = torch.ones(*visual_embeds.size()[:-1], dtype=torch.long, device=device)
            visual_position_id = torch.arange(0,visual_embeds.size()[1])
            visual_position_id = torch.unsqueeze(visual_position_id,0)
            visual_position_id = visual_position_id.repeat(visual_embeds.size()[0], 1)
            visual_position_id = visual_position_id.to(device)

        
        ## question embedding: id type 0, pose incremental
        input['input_ids'] = input['input_ids'].to(device)
        input['attention_mask'] = input['attention_mask'].to(device)

        question_embeds = self.question_embedder(input['input_ids'])
        question_attention_mask = input['attention_mask']
        
        if self.vis_pos_emb == 'zeroes' or self.vis_pos_emb == 'pos':
            question_id_type = torch.zeros(*question_embeds.size()[:-1], dtype=torch.long, device=device)
            question_position_id = torch.arange(0,question_embeds.size()[1])
            question_position_id = torch.unsqueeze(question_position_id,0)
            question_position_id = question_position_id.repeat(question_embeds.size()[0], 1)
            question_position_id = question_position_id.to(device)
        
        
        ## combine visual and question embeds
        ## vision first
        # inputs_embeds = torch.cat((visual_embeds, question_embeds), dim=1)
        # attention_mask = torch.cat((visual_attention_mask, question_attention_mask), dim=1)

        # if self.vis_pos_emb == 'zeroes' or self.vis_pos_emb == 'pos':
        #     token_type_ids = torch.cat((visual_id_type, question_id_type), dim=1)
        #     position_ids = torch.cat((visual_position_id, question_position_id), dim=1)

        ## question first
        inputs_embeds = torch.cat((question_embeds, visual_embeds), dim=1)
        attention_mask = torch.cat((question_attention_mask, visual_attention_mask), dim=1)

        if self.vis_pos_emb == 'zeroes' or self.vis_pos_emb == 'pos':
            token_type_ids = torch.cat((question_id_type, visual_id_type), dim=1)
            position_ids = torch.cat((question_position_id, visual_position_id), dim=1)


        ## VCA_GPT2 decoder
        if self.vis_pos_emb == 'zeroes' or self.vis_pos_emb == 'pos':
            decoder_output = self.VCAdecoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids = position_ids, token_type_ids = token_type_ids)
        else:
            decoder_output = self.VCAdecoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        decoder_output = decoder_output.last_hidden_state.swapaxes(1,2)
        decoder_output = F.adaptive_avg_pool1d(decoder_output,1)
        decoder_output = decoder_output.swapaxes(1,2).squeeze(1)       

        ## intermediate layers
        out =self.intermediate_layer(decoder_output)
        out = self.LayerNorm(out)
        out = self.dropout(out)

        ## classifier
        out = self.classifier(out)
        # print(out.size())
        return out


class EFVLEGPT2SwinClassification(nn.Module):
    def __init__(self, num_class = 12, model_subver = 'v0', vis_pos_emb = None):
        super(EFVLEGPT2SwinClassification, self).__init__()
        '''
        v0: visual embedding : Default patch1 + embedding form VB + GPT2 decoder
        v1: visual embedding : Default patch1 + GPT2 decoder
        '''
        
        self.sub_ver = model_subver
        self.vis_pos_emb = vis_pos_emb
        
        ## image processing
        self.img_feature_extractor = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        
        ## Visual_embedding
        # visual bert embedding
        if self.sub_ver == "v0":
            VB_config = VisualBertConfig.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
            VB_config.visual_embedding_dim = 768
            visualbert = VisualBertModel(config=VB_config)
            self.visual_embedder = visualbert.embeddings.visual_projection

        ## Question_embedding
        question_embedder = GPT2Model.from_pretrained('gpt2')
        self.question_embedder = question_embedder.wte

        ## GPT2 visual_cotext_aware_decoder
        self.VCAdecoder = GPT2Model.from_pretrained('gpt2')
        # for name,child in self.VCAdecoder.h.named_children():
        #     child.mlp._modules['act'] = SwiGLU()
        # self.VCAdecoder.wte = nn.Embedding(2, 768)  # later used for position embedding
 
        ## intermediate_layers
        self.intermediate_layer = nn.Linear(768, 512)  #(512+768)
        self.LayerNorm = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.1)

        ## classifier
        self.classifier = nn.Linear(512, num_class)

    def forward(self, input, img):
        
        ## image encoder features
        img['pixel_values'] = img['pixel_values'].to(device)
        img_feature = self.img_feature_extractor(**img)
        

        ## visual Embedding : id type 1, pos: zero / incremental
        if self.sub_ver == 'v0':
            visual_embeds = self.visual_embedder(img_feature[0])
        elif self.sub_ver == 'v1':
            visual_embeds = img_feature[0]
        
        # if self.training is True:
        #     sigma = torch.tensor(np.random.uniform(0, 0.4, visual_embeds.size()[1])).to(device)
        #     smoothing_layer = get_gaussian_filter(ksize=3, sigma=sigma, channels=visual_embeds.size()[1]).to(device)
        #     visual_embeds = smoothing_layer(visual_embeds)

        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
        visual_attention_mask = visual_attention_mask.to(device)

        if self.vis_pos_emb == 'zeroes':
            visual_id_type = torch.ones(*visual_embeds.size()[:-1], dtype=torch.long, device=device)
            visual_position_id = torch.zeros(*visual_embeds.size()[:-1], dtype=torch.long, device=device)
        elif self.vis_pos_emb == 'pos':
            visual_id_type = torch.ones(*visual_embeds.size()[:-1], dtype=torch.long, device=device)
            visual_position_id = torch.arange(0,visual_embeds.size()[1])
            visual_position_id = torch.unsqueeze(visual_position_id,0)
            visual_position_id = visual_position_id.repeat(visual_embeds.size()[0], 1)
            visual_position_id = visual_position_id.to(device)
        
        
        ## question embedding: id type 0, pose incremental
        input['input_ids'] = input['input_ids'].to(device)
        input['attention_mask'] = input['attention_mask'].to(device)

        question_embeds = self.question_embedder(input['input_ids'])
        question_attention_mask = input['attention_mask']
        
        if self.vis_pos_emb == 'zeroes' or self.vis_pos_emb == 'pos':
            question_id_type = torch.zeros(*question_embeds.size()[:-1], dtype=torch.long, device=device)
            question_position_id = torch.arange(0,question_embeds.size()[1])
            question_position_id = torch.unsqueeze(question_position_id,0)
            question_position_id = question_position_id.repeat(question_embeds.size()[0], 1)
            question_position_id = question_position_id.to(device)
        

        ## combine visual and question embeds
        ## vision first
        # inputs_embeds = torch.cat((visual_embeds, question_embeds), dim=1)
        # attention_mask = torch.cat((visual_attention_mask, question_attention_mask), dim=1)

        # if self.vis_pos_emb == 'zeroes' or self.vis_pos_emb == 'pos':
        #     token_type_ids = torch.cat((visual_id_type, question_id_type), dim=1)
        #     position_ids = torch.cat((visual_position_id, question_position_id), dim=1)

        ## question first
        inputs_embeds = torch.cat((question_embeds, visual_embeds), dim=1)
        attention_mask = torch.cat((question_attention_mask, visual_attention_mask), dim=1)

        if self.vis_pos_emb == 'zeroes' or self.vis_pos_emb == 'pos':
            token_type_ids = torch.cat((question_id_type, visual_id_type), dim=1)
            position_ids = torch.cat((question_position_id, visual_position_id), dim=1)

        ## VCA_GPT2 decoder
        if self.vis_pos_emb == 'zeroes' or self.vis_pos_emb == 'pos':
            decoder_output = self.VCAdecoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids = position_ids, token_type_ids = token_type_ids)
        else:
            decoder_output = self.VCAdecoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        decoder_output = decoder_output.last_hidden_state.swapaxes(1,2)
        decoder_output = F.adaptive_avg_pool1d(decoder_output,1)
        decoder_output = decoder_output.swapaxes(1,2).squeeze(1)       

        ## intermediate layers
        out =self.intermediate_layer(decoder_output)
        out = self.LayerNorm(out)
        out = self.dropout(out)

        ## classifier
        out = self.classifier(out)
        # print(out.size())
        return out


class EFVLEGPT2ViTClassification(nn.Module):
    def __init__(self, num_class = 12, model_subver = 'v0', vis_pos_emb = None):
        super(EFVLEGPT2ViTClassification, self).__init__()
        '''
        v0: visual embedding : Default patch1 + embedding form VB + GPT2 decoder
        v1: visual embedding : Default patch1 + GPT2 decoder
        '''
        
        self.sub_ver = model_subver
        self.vis_pos_emb = vis_pos_emb
        
        ## image processing
        self.img_feature_extractor = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        # self.img_feature_extractor = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        
        ## Visual_embedding
        # visual bert embedding
        if self.sub_ver == "v0":
            VB_config = VisualBertConfig.from_pretrained("uclanlp/visualbert-vqa-coco-pre")
            VB_config.visual_embedding_dim = 768
            visualbert = VisualBertModel(config=VB_config)
            self.visual_embedder = visualbert.embeddings.visual_projection

        ## Question_embedding
        question_embedder = GPT2Model.from_pretrained('gpt2')
        self.question_embedder = question_embedder.wte

        ## GPT2 visual_cotext_aware_decoder
        self.VCAdecoder = GPT2Model.from_pretrained('gpt2')
        # for name,child in self.VCAdecoder.h.named_children():
        #     child.mlp._modules['act'] = SwiGLU()
        # self.VCAdecoder.wte = nn.Embedding(2, 768)  # later used for position embedding
 
        ## intermediate_layers
        self.intermediate_layer = nn.Linear(768, 512)  #(512+768)
        self.LayerNorm = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(0.1)

        ## classifier
        self.classifier = nn.Linear(512, num_class)

    def forward(self, input, img):
        
        ## image encoder features
        img['pixel_values'] = img['pixel_values'].to(device)
        img_feature = self.img_feature_extractor(**img)
        

        ## visual Embedding : id type 1, pos: zero / incremental
        if self.sub_ver == 'v0':
            visual_embeds = self.visual_embedder(img_feature[0])
        elif self.sub_ver == 'v1':
            visual_embeds = img_feature[0]
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
        visual_attention_mask = visual_attention_mask.to(device)

        if self.vis_pos_emb == 'zeroes':
            visual_id_type = torch.ones(*visual_embeds.size()[:-1], dtype=torch.long, device=device)
            visual_position_id = torch.zeros(*visual_embeds.size()[:-1], dtype=torch.long, device=device)
        elif self.vis_pos_emb == 'pos':
            visual_id_type = torch.ones(*visual_embeds.size()[:-1], dtype=torch.long, device=device)
            visual_position_id = torch.arange(0,visual_embeds.size()[1])
            visual_position_id = torch.unsqueeze(visual_position_id,0)
            visual_position_id = visual_position_id.repeat(visual_embeds.size()[0], 1)
            visual_position_id = visual_position_id.to(device)
        
        
        ## question embedding: id type 0, pose incremental
        input['input_ids'] = input['input_ids'].to(device)
        input['attention_mask'] = input['attention_mask'].to(device)

        question_embeds = self.question_embedder(input['input_ids'])
        question_attention_mask = input['attention_mask']
        
        if self.vis_pos_emb == 'zeroes' or self.vis_pos_emb == 'pos':
            question_id_type = torch.zeros(*question_embeds.size()[:-1], dtype=torch.long, device=device)
            question_position_id = torch.arange(0,question_embeds.size()[1])
            question_position_id = torch.unsqueeze(question_position_id,0)
            question_position_id = question_position_id.repeat(question_embeds.size()[0], 1)
            question_position_id = question_position_id.to(device)
        

        ## combine visual and question embeds
        ## vision first
        # inputs_embeds = torch.cat((visual_embeds, question_embeds), dim=1)
        # attention_mask = torch.cat((visual_attention_mask, question_attention_mask), dim=1)

        # if self.vis_pos_emb == 'zeroes' or self.vis_pos_emb == 'pos':
        #     token_type_ids = torch.cat((visual_id_type, question_id_type), dim=1)
        #     position_ids = torch.cat((visual_position_id, question_position_id), dim=1)

        # question first
        inputs_embeds = torch.cat((question_embeds, visual_embeds), dim=1)
        attention_mask = torch.cat((question_attention_mask, visual_attention_mask), dim=1)

        if self.vis_pos_emb == 'zeroes' or self.vis_pos_emb == 'pos':
            token_type_ids = torch.cat((question_id_type, visual_id_type), dim=1)
            position_ids = torch.cat((question_position_id, visual_position_id), dim=1)

        ## VCA_GPT2 decoder
        if self.vis_pos_emb == 'zeroes' or self.vis_pos_emb == 'pos':
            decoder_output = self.VCAdecoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids = position_ids, token_type_ids = token_type_ids)
        else:
            decoder_output = self.VCAdecoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        decoder_output = decoder_output.last_hidden_state.swapaxes(1,2)
        decoder_output = F.adaptive_avg_pool1d(decoder_output,1)
        decoder_output = decoder_output.swapaxes(1,2).squeeze(1)       

        ## intermediate layers
        out =self.intermediate_layer(decoder_output)
        out = self.LayerNorm(out)
        out = self.dropout(out)

        ## classifier
        out = self.classifier(out)
        # print(out.size())
        return out