import torch
import torch.nn as nn
import torch.nn.functional as F


class PSJALoss(nn.Module):
    def __init__(self, t, r_L2,n_classes):
        super(PSJALoss, self).__init__()
        self.t = t
        self.r_L2 = r_L2
        self.class_num = n_classes
        self.feature_dict = {}

    def update_feature_dict(self, fea, score, label):
        B,_,_=fea.shape
        for b in range(B):
            feab = fea[b]
            scoreb = score[b]
            labelb = int(label[b])
            sorted_indices = torch.argsort(scoreb)
            sorted_fea = feab[sorted_indices]
            sorted_score = scoreb[sorted_indices]
    
            max_score = torch.max(score).item()
            min_score = torch.min(score).item()
            
            selected_fea_high = []
            high_attention_indices = (sorted_score >= max_score - 0.3) & (sorted_score <= max_score)
            high_attention_fea = sorted_fea[high_attention_indices]
            selected_fea_high.extend(high_attention_fea[:15])
            self.feature_dict[labelb].extend(selected_fea_high)

            selected_fea_low = []
            low_attention_indices = (sorted_score >= min_score) & (sorted_score <= min_score + 1e-9)
            low_attention_fea = sorted_fea[low_attention_indices]
            selected_fea_low.extend(low_attention_fea[::2][:5])
            self.feature_dict[self.class_num].extend(selected_fea_low)
        for l in range(self.class_num):
            self.feature_dict[l] = self.feature_dict[l][-20*15:]
        self.feature_dict[self.class_num] = self.feature_dict[self.class_num][-20*5:]

    def patch_contrastive_estimation_loss(self, fea, score, label):
        B, _, _ = fea.shape
        total_loss = 0.0
        for b in range(B):
            loss = 0.0
            feab = fea[b]
            scoreb = score[b]
            labelb = int(label[b])
            sorted_indices = torch.argsort(scoreb)
            sorted_fea = feab[sorted_indices]
            sorted_score = scoreb[sorted_indices]

            max_score = torch.max(score).item()
            min_score = torch.min(score).item()

            positive_samples = torch.stack(self.feature_dict[labelb]).to(fea.device)
            negative_samples = torch.stack(self.feature_dict[self.class_num]).to(fea.device)
            all_samples = torch.cat([torch.stack(self.feature_dict[k]) for k in self.feature_dict]).to(fea.device)

            selected_fea_high = []
            high_attention_indices = (sorted_score >= max_score - 0.3) & (sorted_score <= max_score)
            high_attention_fea = sorted_fea[high_attention_indices]
            selected_fea_high.extend(high_attention_fea[:15])

            selected_fea_low = []
            low_attention_indices = (sorted_score >= min_score) & (sorted_score <= min_score + 1e-9)
            low_attention_fea = sorted_fea[low_attention_indices]
            selected_fea_low.extend(low_attention_fea[::2][:5])
            num_K=0
            for f in selected_fea_high:
                numerator = torch.exp(torch.mm(positive_samples, f.unsqueeze(1)).squeeze() / self.tau).sum()
                denominator = torch.exp(torch.mm(all_samples, f.unsqueeze(1)).squeeze() / self.tau).sum()
                loss += -torch.log(numerator / denominator)
                num_K+=1
            for f in selected_fea_low:
                numerator = torch.exp(torch.mm(negative_samples, f.unsqueeze(1)).squeeze() / self.tau).sum()
                denominator = torch.exp(torch.mm(all_samples, f.unsqueeze(1)).squeeze() / self.tau).sum()
                loss += -torch.log(numerator / denominator)
                num_K += 1

            total_loss += loss / num_K

        contrastive_loss = total_loss / B

        return contrastive_loss

    def adaptive_cross_entropy_loss(self, t_logit,ori_logits, score,l2_wei):
        batch_size, num_classes = ori_logits.size()
        loss = 0.0
        for i in range(batch_size):
            numerator = torch.exp(t_logit)
            denominator = torch.exp(ori_logits[i]).sum()
            loss += -torch.log(numerator / denominator)
        cross_entropy_loss = loss / batch_size

        D = torch.mean(score) / torch.max(score)
        g = self.r_L2 * (1 - D) ** 2
        l2_loss = g * l2_wei
        total_loss = cross_entropy_loss + l2_loss
        return total_loss

    def forward(self, fea, t_logit,ori_logits, score, label,l2_wei):
        self.update_feature_dict(fea, score, label)
        
        contrastive_loss = self.patch_contrastive_estimation_loss(fea,score, label)
        cross_entropy_loss = self.adaptive_cross_entropy_loss(t_logit,ori_logits, score,l2_wei)

        total_loss = contrastive_loss + cross_entropy_loss

        return total_loss


