
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import transformers
import peft

from torch.cuda.amp import autocast as autocast
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torchvision import models
from torchvision.ops import roi_align
from modelscope.hub.snapshot_download import snapshot_download

from lavis.common.registry import registry
from lavis.models.affordance_models.PointNet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from lavis.models.blip2_models.modeling_llama import LlamaForCausalLM
from lavis.models.blip2_models.modeling_opt import OPTForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, LlamaTokenizer

from fvcore.nn import FlopCountAnalysis
from thop import profile

class Img_Encoder(nn.Module):
    def __init__(self):
        super(Img_Encoder, self).__init__()

        self.model = models.resnet18(weights=None)
        self.model.relu = nn.ReLU()

    def forward(self, img):
        # img - [B, C, H, W]
        x = self.model.conv1(img)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x) 
        x_1 = self.model.layer1(x)   
        x_2 = self.model.layer2(x_1)         
        x_3 = self.model.layer3(x_2)       
        out = self.model.layer4(x_3)

        return out

class Point_Encoder(nn.Module):
    def __init__(self, emb_dim, normal_channel, additional_channel, N_p):
        super().__init__()

        self.N_p = N_p
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstractionMsg(self.N_p, [0.2,0.4], [16, 32], 256+256, [[128, 128, 256], [128, 196, 256]])

    def forward(self, xyz):

        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)  #[B, 3, npoint_sa1] --- [B, 320, npoint_sa1]
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)  #[B, 3, npoint_sa2] --- [B, 512, npoint_sa2]
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)  #[B, 3, N_p]        --- [B, 512, N_p]

        return [[l0_xyz, l0_points], [l1_xyz, l1_points], [l2_xyz, l2_points], [l3_xyz, l3_points]]

class Self_Attention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(Self_Attention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x):
        batch_size, seq_len, embed_dim = x.size()
        queries = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)    # (batch_size, num_heads, seq_len, head_dim)
        keys = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)         # (batch_size, num_heads, seq_len, head_dim)
        values = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)     # (batch_size, num_heads, seq_len, head_dim)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.hidden_size ** 0.5)                  # (batch_size, num_heads, seq_len, seq_len)
        attention_weights = nn.functional.softmax(scores, dim=-1)                                           # (batch_size, num_heads, seq_len, seq_len)
        out = torch.matmul(attention_weights, values)                                                       # (batch_size, num_heads, seq_len, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)                         # (batch_size, seq_len, embed_dim)
        out = self.ln(out + x)
        return out

class Cross_Attention(nn.Module):
    def __init__(self, emb_dim, proj_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.proj_dim = proj_dim
        self.proj_q = nn.Linear(self.emb_dim, proj_dim)
        self.proj_sk = nn.Linear(self.emb_dim, proj_dim)
        self.proj_sv = nn.Linear(self.emb_dim, proj_dim)
        self.proj_ek = nn.Linear(self.emb_dim, proj_dim)
        self.proj_ev = nn.Linear(self.emb_dim, proj_dim)
        self.scale = self.proj_dim ** (-0.5)

        self.layernorm = nn.LayerNorm(self.emb_dim)
    def forward(self, obj, sub, scene):
        '''
        obj: [B,N_p+HW,C]
        others : [B, HW, C]
        '''
        B, seq_length, C = obj.size()
        query = self.proj_q(obj)                                         #[B, N_q, proj_dim]
        s_key = self.proj_sk(sub)                                        #[B, N_i, proj_dim]
        s_value = self.proj_sv(sub)

        e_key = self.proj_ek(scene)
        e_value = self.proj_ev(scene)

        atten_I1 = torch.bmm(query, s_key.mT)*self.scale                 #[B, N_q, N_i]
        atten_I1 = atten_I1.softmax(dim=-1)
        I_1 = torch.bmm(atten_I1, s_value)

        atten_I2 = torch.bmm(query, e_key.mT)*self.scale                 #[B, N_q, N_i]
        atten_I2 = atten_I2.softmax(dim=-1)
        I_2 = torch.bmm(atten_I2, e_value)

        I_1 = self.layernorm(obj + I_1)                                  #[B, N_q+N_i, emb_dim]
        I_2 = self.layernorm(obj + I_2)
        return I_1, I_2

class Fusion(nn.Module):
    def __init__(self, emb_dim = 512, num_heads = 4):
        super().__init__()
        self.emb_dim = emb_dim
        self.div_scale = self.emb_dim ** (-0.5)
        self.num_heads = num_heads

        self.mlp = nn.Sequential(
            nn.Conv1d(self.emb_dim, 2*self.emb_dim, 1, 1),
            nn.BatchNorm1d(2*self.emb_dim),
            nn.ReLU(),
            nn.Conv1d(2*self.emb_dim, self.emb_dim, 1, 1),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU()         
        )

        self.img_attention = Self_Attention(self.emb_dim, self.num_heads)
        self.point_attention = Self_Attention(self.emb_dim, self.num_heads)
        self.joint_attention = Self_Attention(self.emb_dim, self.num_heads)

    def forward(self, img_feature, point_feature):
        '''
        i_feature: [B, C, H, W]
        p_feature: [B, C, N_p]
        HW = N_i
        '''
        B, C, H, W = img_feature.size()
        img_feature = img_feature.view(B, self.emb_dim, -1)                            #[B, C, N_i]
        point_feature = point_feature[-1][1]

        p_feature = self.mlp(point_feature)
        i_feature = self.mlp(img_feature)

        phi = torch.bmm(p_feature.permute(0, 2, 1), i_feature)*self.div_scale          #[B, N_p, N_i]
        phi_p = F.softmax(phi,dim=1)
        phi_i = F.softmax(phi,dim=-1)  
        I_enhance = torch.bmm(p_feature, phi_p)                                        #[B, C, N_i]
        P_enhance = torch.bmm(i_feature, phi_i.permute(0,2,1))                         #[B, C, N_p]
        I = self.img_attention(I_enhance.mT)                                           #[B, N_i, C]
        P = self.point_attention(P_enhance.mT)                                         #[B, N_p, C]

        joint_patch = torch.cat((P, I), dim=1)                                       
        multi_feature = self.joint_attention(joint_patch)                              #[B, N_p+N_i, C]

        return multi_feature

class Affordance_Decoder(nn.Module):
    def __init__(self, emb_dim, proj_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.proj_dim = proj_dim
        self.cross_atten = Cross_Attention(emb_dim = self.emb_dim, proj_dim = self.proj_dim)
        self.fusion = nn.Sequential(
            nn.Conv1d(2*self.emb_dim, self.emb_dim, 1, 1),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU()
        )

    def forward(self, query, key, value):
        '''
        query: [B, N_p + N_i, C]
        F_s: [B, H, W, C]
        F_e: [B, H, W, C]
        '''
        B,_,C = query.size()

        key = key.view(B, C, -1)                                        #[B, N_p + N_i + N_l, C]
        value = value.view(B, C, -1)                                    #[B, N_p + N_i + N_l, C]
        Theta_1, Theta_2 = self.cross_atten(query, key.mT, value.mT)    #[B, C, N_p + N_i]

        joint_context = torch.cat((Theta_1.mT, Theta_2.mT), dim=1)      #[B, 2C, N_p + N_i]
        affordance = self.fusion(joint_context)                         #[B, C, N_p + N_i]
        affordance = affordance.permute(0, 2, 1)                        #[B, N_p + N_i, C]

        return affordance

class Head(nn.Module):
    def __init__(self, additional_channel, emb_dim, N_p, N_raw):
        super().__init__()
        
        self.emb_dim = emb_dim
        self.N_p = N_p
        self.N_raw = N_raw
        #upsample
        self.fp3 = PointNetFeaturePropagation(in_channel=512+self.emb_dim, mlp=[768, 512])  
        self.fp2 = PointNetFeaturePropagation(in_channel=832, mlp=[768, 512]) 
        self.fp1 = PointNetFeaturePropagation(in_channel=518+additional_channel, mlp=[512, 512]) 
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.out_head = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim // 8),
            nn.BatchNorm1d(self.N_raw),
            nn.ReLU(),
            nn.Linear(self.emb_dim // 8, 1),
            nn.Sigmoid()
        )

    def forward(self, multi_feature, affordance_feature, encoder_p):
        '''
        multi_feature ---> [B, N_p + N_i, C]
        affordance_feature ---> [B, N_p + N_i, C]
        encoder_p ---> [Hierarchy feature]
        '''
        B,N,C = multi_feature.size()
        p_0, p_1, p_2, p_3 = encoder_p
        P_align, _ = torch.split(multi_feature, split_size_or_sections=self.N_p, dim=1)         #[B, N_p, C] --- [B, N_i, C]
        F_pa, _ = torch.split(affordance_feature, split_size_or_sections = self.N_p, dim=1)     #[B, N_p, C] --- [B, N_i, C]

        up_sample = self.fp3(p_2[0], p_3[0], p_2[1], P_align.mT)                                #[B, emb_dim, npoint_sa2]
        up_sample = self.fp2(p_1[0], p_2[0], p_1[1], up_sample)                                 #[B, emb_dim, npoint_sa1]                        
        up_sample = self.fp1(p_0[0], p_1[0], torch.cat([p_0[0], p_0[1]],1), up_sample)          #[B, emb_dim, N_raw]
        F_pa_pool = self.pool(F_pa.mT)                                                          #[B, emb_dim, 1]
        
        affordance = up_sample * F_pa_pool.expand(-1,-1,self.N_raw)                             #[B, emb_dim, 2048]
        
        out = self.out_head(affordance.mT)                                                      #[B, 2048, 1]

        return out
    
class Loss_HM(nn.Module):
    def __init__(self):
        super().__init__()
        self.gamma = 2
        self.alpha = 0.25

    def forward(self, pred, target):
        #[B, N, 18]
        temp1 = -(1-self.alpha)*torch.mul(pred**self.gamma,
                           torch.mul(1-target, torch.log(1-pred+1e-6)))
        temp2 = -self.alpha*torch.mul((1-pred)**self.gamma,
                           torch.mul(target, torch.log(pred+1e-6)))
        temp = temp1+temp2
        mean_focal_loss = torch.sum(torch.mean(temp, (0, 1)))

        intersection_positive = torch.sum(pred*target, 1)
        cardinality_positive = torch.sum(torch.abs(pred)+torch.abs(target), 1)
        dice_positive = (intersection_positive+1e-6) / \
            (cardinality_positive+1e-6)

        intersection_negative = torch.sum((1.-pred)*(1.-target), 1)
        cardinality_negative = torch.sum(
            2-torch.abs(pred)-torch.abs(target), 1)
        dice_negative = (intersection_negative+1e-6) / \
            (cardinality_negative+1e-6)
        temp3 = torch.mean(1.5-dice_positive-dice_negative, 0)

        dice_loss = torch.sum(temp3)
        return mean_focal_loss+1.0*dice_loss

@registry.register_model("lm_affordance_3d")
class LMAffordance3D(Blip2Base):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "vicuna7b": "configs/models/blip2/blip2_instruct_vicuna7b.yaml",
    }
    def __init__(self,
                 pre_train=True,
                 pretrained_ckpt=None,
                 llm_model="llava-v1.5-7b",
                 has_lora=True,
                 has_qformer=False,
                 normal_channel=False,
                 N_p = 64, 
                 emb_dim = 512, 
                 proj_dim = 512, 
                 num_heads = 4, 
                 N_raw = 2048, 
                 max_txt_len=32,
                 ):
        super().__init__()

        self.emb_dim = emb_dim
        self.N_p = N_p
        self.N_raw = N_raw
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.normal_channel = normal_channel
        if self.normal_channel:
            self.additional_channel = 3
        else:
            self.additional_channel = 0

        self.has_lora = has_lora
        self.has_qformer = has_qformer
        self.max_txt_len = max_txt_len
        self.tokenizer = self.init_tokenizer(truncation_side="left")

        llm_model = snapshot_download('AI-ModelScope/llava-v1.6-vicuna-7b')

        if 'opt' in llm_model:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side='left')
            self.llm_model = OPTForCausalLM.from_pretrained(llm_model, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        else:
            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model, use_fast=False, truncation_side="left")
            self.llm_model = LlamaForCausalLM.from_pretrained(llm_model, torch_dtype=torch.float16, low_cpu_mem_usage=True)         
            # self.llm_model = AutoModelForCausalLM.from_pretrained(llm_model, torch_dtype=torch.float16, low_cpu_mem_usage=True)


        self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.llm_tokenizer.add_special_tokens({'bos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'eos_token': '</s>'})
        self.llm_tokenizer.add_special_tokens({'unk_token': '</s>'})

        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        if self.has_qformer:
            print('Loading Q-Former')
            self.Qformer, self.query_tokens = self.init_Qformer(
                4, self.emb_dim
            )
            self.Qformer.resize_token_embeddings(len(self.llm_tokenizer))
            self.Qformer.cls = None
            self.llm_proj = nn.Linear(self.Qformer.config.hidden_size, self.llm_model.config.hidden_size)

        if self.has_lora:
            loraconfig = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["q_proj","v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.llm_model = prepare_model_for_kbit_training(self.llm_model)
            self.llm_model = get_peft_model(self.llm_model, loraconfig)
            self.llm_model.print_trainable_parameters()
        else:
            for name, param in self.llm_model.named_parameters():
                param.requires_grad = False

        self.img_encoder = Img_Encoder()
        if pre_train:
            pretrain_dict = torch.load(pretrained_ckpt)
            img_model_dict = self.img_encoder.state_dict()
            for k in list(pretrain_dict.keys()):
                new_key = 'model.' + k
                pretrain_dict[new_key] = pretrain_dict.pop(k)
            pretrain_dict={ k : v for k, v in pretrain_dict.items() if k in img_model_dict}
            img_model_dict.update(pretrain_dict)
            self.img_encoder.load_state_dict(img_model_dict)

        self.point_encoder = Point_Encoder(self.emb_dim, self.normal_channel, self.additional_channel, self.N_p)

        self.fusion = Fusion(self.emb_dim, self.num_heads)

        self.adapter_up = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.llm_model.config.hidden_size)
        )
        self.adapter_down = nn.Sequential(
            nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.llm_model.config.hidden_size, self.emb_dim)
        )
        self.affordance_decoder = Affordance_Decoder(self.emb_dim, self.proj_dim)

        self.head = Head(self.additional_channel, self.emb_dim, self.N_p, self.N_raw)

        self.loss_hm = Loss_HM()
        self.w_hm = 1.0
        
    def forward(self, img, point, description, label, inference_mode=False):
        '''
        img: [B, 3, H, W]
        point: [B, 3, 2048]
        description: nature language
        '''
        B, C, H, W = img.size()
        B, D, N = point.size()
        device = img.device

        img_feature = self.img_encoder(img)
        point_feature = self.point_encoder(point)
        spatial_feature = self.fusion(img_feature, point_feature)

        if self.has_qformer:
            query_tokens = self.query_tokens.expand(spatial_feature.shape[0], -1, -1)
            text_Qformer = self.llm_tokenizer(
                description,
                padding='longest',
                truncation=True,
                max_length=self.max_txt_len,
                return_tensors="pt",
            ).to(device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(device)
            Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)
            image_atts = torch.ones(spatial_feature.size()[:-1], dtype=torch.long).to(device)

            query_output = self.Qformer.bert(
                text_Qformer.input_ids,
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=spatial_feature,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            multi_embeds = self.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
            multi_embeds = multi_embeds.view(B, 1, *multi_embeds.size()[1:])
        else:
            multi_embeds = self.adapter_up(spatial_feature)
            image_atts = None

        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'
        text_input_tokens = self.llm_tokenizer(
            description,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(device)

        inputs_embeds = self.llm_model.get_input_embeddings()(text_input_tokens.input_ids)
        # inputs_embeds shape: (batch_size, sequence_length, hidden_size)

        llm_inputs, llm_attention_mask = self.concat_input(inputs_embeds, text_input_tokens.attention_mask, multi_embeds, image_atts)
        with self.maybe_autocast():
            hidden_states = self.llm_model(
                inputs_embeds=llm_inputs,
                attention_mask=llm_attention_mask,
                return_dict=False,
            )

        hidden_states = self.adapter_down(hidden_states)
        semantic_feature, instructional_feature= torch.split(hidden_states, split_size_or_sections = spatial_feature.size(1), dim=1)

        affordance_feature = self.affordance_decoder(spatial_feature, instructional_feature, semantic_feature)

        out = self.head(spatial_feature, affordance_feature, point_feature)

        if inference_mode == True:
            return out
        else:
            loss_hm = self.loss_hm(out, label)
            loss = loss_hm * self.w_hm
            return {"out": out, "loss": loss, "loss_hm": loss_hm}


    def concat_input(self, input_embeds, input_atts, multi_embeds, image_atts=None):
        '''
        input_embeds: (batch_size, sequence_length, hidden_size)
        input_atts: (batch_size, sequence_length)
        multi_embeds: (batch_size, n, hidden_size)
        image_atts: (batch_size, sequence_length)

        mask:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        '''
        llm_inputs = []
        llm_attention_mask = []
        bs = multi_embeds.size()[0]
        for i in range(bs):
            bs, n, dim = multi_embeds.size()
            this_input_ones = input_atts[i].sum()
            llm_inputs.append(
                torch.cat([
                    input_embeds[i][:this_input_ones],
                    multi_embeds[i],
                    input_embeds[i][this_input_ones:]
                ])
            )
            if image_atts is None:
                llm_attention_mask.append(
                    torch.cat([
                        input_atts[i][:this_input_ones],
                        torch.ones((n), device=multi_embeds.device, dtype=torch.long),
                        input_atts[i][this_input_ones:]
                    ])
                )
            else: 
                llm_attention_mask.append(
                    torch.cat([
                        input_atts[i][:this_input_ones],
                        image_atts[i],
                        input_atts[i][this_input_ones:]
                    ])
                )
        llm_inputs = torch.stack(llm_inputs, 0)
        llm_attention_mask = torch.stack(llm_attention_mask, 0)

        return llm_inputs, llm_attention_mask
    
    def get_optimizer_params(self, weight_decay, lr_scale=1):
        parameter_group_names = {}
        parameter_group_vars = {}

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias"):
                group_name = "no_decay"
                this_weight_decay = 0.
            else:
                group_name = "decay"
                this_weight_decay = weight_decay
            if 'visual_encoder' in name:
                group_name = "vit_layer_%s" % (group_name)
            else:
                layer_id = None

            if group_name not in parameter_group_names:
                scale = 1
                parameter_group_names[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
                parameter_group_vars[group_name] = {
                    "weight_decay": this_weight_decay,
                    "params": [],
                    "lr_scale": scale
                }
            parameter_group_vars[group_name]["params"].append(param)
            parameter_group_names[group_name]["params"].append(name)
        optim_params = list(parameter_group_vars.values())
        return optim_params
    
    @classmethod
    def from_config(cls, cfg):
        pre_train = cfg.get('pre_train', True)
        pretrained_ckpt = cfg.get("pretrained_ckpt")
        llm_model = cfg.get("llm_model")
        has_lora = cfg.get('has_lora', True)
        has_qformer = cfg.get('has_qformer', False)
        normal_channel = cfg.get('normal_channel', False)
        N_p = cfg.get("N_p", 64)
        emb_dim = cfg.get("emb_dim", 512)
        proj_dim = cfg.get("proj_dim", 512)
        num_heads = cfg.get("num_heads", 4)
        N_raw = cfg.get("N_raw", 2048)
        max_txt_len = cfg.get("max_txt_len", 64)

        model = cls(
            pre_train=pre_train,
            pretrained_ckpt=pretrained_ckpt,
            llm_model=llm_model,
            has_lora=has_lora,
            has_qformer=has_qformer,
            normal_channel=normal_channel,
            N_p = N_p, 
            emb_dim = emb_dim, 
            proj_dim = proj_dim, 
            num_heads = num_heads, 
            N_raw = N_raw, 
            max_txt_len=max_txt_len,
        )

        return model