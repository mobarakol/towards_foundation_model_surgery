import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from transformers import  VisualBertConfig, GPT2Config
from transformers import VisualBertModel, GPT2Model, ViTModel, SwinModel
from transformers import GPT2LMHeadModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


''' Early Fusion GPT with CNN/Transformers'''

class EFVLEGPT2RS18Sentence(nn.Module):
    def __init__(self, model_subver = 'v3', tokenizer_len=50258, vis_pos_emb = None):
        super(EFVLEGPT2RS18Sentence, self).__init__()
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

        ## word_embedding
        # default GPT2 word embedding
        gpt2configuration = GPT2Config()
        word_embedder = GPT2Model(gpt2configuration)
        word_embedder.resize_token_embeddings(tokenizer_len)
        self.word_embedder = word_embedder.wte

        ## GPT2 visual context aware decoder
        self.VCAdecoder = GPT2LMHeadModel.from_pretrained('gpt2')
        
        
    def forward(self, question, img, answer):
        
        ## image encoder features
        img_feature = self.img_feature_extractor(img)
        
        if self.sub_ver == 'v0' or self.sub_ver =='v1':
            img_feature = torch.unsqueeze(img_feature, dim=1)
        if self.sub_ver == 'v2'or self.sub_ver =='v3':
            img_feature = torch.flatten(img_feature, start_dim=2)
            img_feature = img_feature.permute((0,2,1))
        
        
        ## visual Embedding : id type 1, pos: zero / incremental
        visual_embeds = self.visual_embedder(img_feature)
        visual_attention_mask = torch.ones(visual_embeds.shape[:-1], dtype=torch.float)
        visual_attention_mask = visual_attention_mask.to(device)

        ## question embedding:
        question['input_ids'] = question['input_ids'].to(device)
        question_embeds = self.word_embedder(question['input_ids'])
        question_attention_mask = question['attention_mask'].to(device)

        ## answer embedding        
        answer['input_ids'] = answer['input_ids'].to(device)        
        answer_embeds = self.word_embedder(answer['input_ids'])
        answer_attention_mask = answer['attention_mask'].to(device)

        ## token type and position id for question
        if self.vis_pos_emb == 'zeroes' or self.vis_pos_emb == 'pos':
            question_id_type = torch.zeros(*question_embeds.size()[:-1], dtype=torch.long, device=device)
            question_position_id = torch.arange(0,question_embeds.size()[1])
            question_position_id = torch.unsqueeze(question_position_id,0)
            question_position_id = question_position_id.repeat(question_embeds.size()[0], 1)
            question_position_id = question_position_id.to(device)
            question_len = len(question_position_id[0])
        
        ## token type and position id for vision
        if self.vis_pos_emb == 'zeroes':
            visual_id_type = torch.ones(*visual_embeds.size()[:-1], dtype=torch.long, device=device)
            visual_position_id = torch.zeros(*visual_embeds.size()[:-1], dtype=torch.long, device=device)
        elif self.vis_pos_emb == 'pos':
            visual_id_type = torch.ones(*visual_embeds.size()[:-1], dtype=torch.long, device=device)
            visual_position_id = torch.arange(0,visual_embeds.size()[1])
            visual_position_id = torch.unsqueeze(visual_position_id,0)
            visual_position_id = visual_position_id.repeat(visual_embeds.size()[0], 1)
            visual_position_id += (question_len)
            visual_position_id = visual_position_id.to(device)
        visual_len = len(visual_position_id[0])

        ## token type and position id for answer
        if self.vis_pos_emb == 'zeroes' or self.vis_pos_emb == 'pos':
            answer_id_type = torch.zeros(*answer_embeds.size()[:-1], dtype=torch.long, device=device)
            answer_position_id = torch.arange(0,answer_embeds.size()[1])
            answer_position_id = torch.unsqueeze(answer_position_id,0)
            answer_position_id = answer_position_id.repeat(answer_embeds.size()[0], 1)
            answer_position_id += (question_len+visual_len)
            answer_position_id = answer_position_id.to(device)

        ## combine visual and question embeds
        ## vision first
        # inputs_embeds = torch.cat((visual_embeds, question_embeds), dim=1)
        # attention_mask = torch.cat((visual_attention_mask, question_attention_mask), dim=1)

        # if self.vis_pos_emb == 'zeroes' or self.vis_pos_emb == 'pos':
        #     token_type_ids = torch.cat((visual_id_type, question_id_type), dim=1)
        #     position_ids = torch.cat((visual_position_id, question_position_id), dim=1)

        ## question first
        inputs_embeds = torch.cat((question_embeds, visual_embeds, answer_embeds), dim=1)
        attention_mask = torch.cat((question_attention_mask, visual_attention_mask, answer_attention_mask), dim=1)

        if self.vis_pos_emb == 'zeroes' or self.vis_pos_emb == 'pos':
            token_type_ids = torch.cat((question_id_type, visual_id_type, answer_id_type), dim=1)
            position_ids = torch.cat((question_position_id, visual_position_id, answer_position_id), dim=1)


        ## VCA_GPT2 decoder
        if self.vis_pos_emb == 'zeroes' or self.vis_pos_emb == 'pos':
            out = self.VCAdecoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask, position_ids = position_ids, token_type_ids = token_type_ids)
        else:
            out = self.VCAdecoder(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        

        return out