import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import random
import torch.distributed as dist
from collections import defaultdict

# 既存の損失クラスはそのまま残し、DINO系損失を追加

class DINOLoss(nn.Module):
    """
    DINOの自己教師あり学習用損失関数。
    学習中にteacherの出力の中心化・温度スケジューリングを行う。
    """
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, args=None, logger=None,
                 unl_pos_weight=None):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.batch_size = args.batch_size_per_gpu
        self.logger = logger
        self.unl_pos_weight = unl_pos_weight

    def forward(self, student_output, teacher_output, epoch, iteration=0, 
                class_ids=None, student_raw_out=None, pos_output=False, pos_batch=False, use_weight=False):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        if use_weight and torch.is_tensor(class_ids): #unlとposの不均衡を解消するため, lossに重みをつける
            unl_pos = class_ids[:,0] #batch_size  [0,0,0,1,0,0,0,...]
            batch_weight = (1-unl_pos)*self.unl_pos_weight["unl"] + unl_pos*self.unl_pos_weight["pos"]
            batch_weight = batch_weight.to("cuda")
        else:
            batch_weight = None

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                if batch_weight is not None:
                    loss = loss * batch_weight
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)

        if self.logger and (iteration%500==0 or iteration==0):
            self.visualize_output(student_out, student_raw_out, teacher_out, iteration, temp)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def visualize_output(self, student_sharpend_out, student_raw_out, teacher_sf_out, iteration, temp):
        ### plotting 2 teacher softmax outputs
        plot_minibatch = np.random.randint(0, self.batch_size)
        plot_teacher_0=teacher_sf_out[0][plot_minibatch].detach().cpu().numpy() #emb_dim
        plot_teacher_1=teacher_sf_out[1][plot_minibatch].detach().cpu().numpy() #emb_dim
        top_100_indices_from0 = np.argsort(plot_teacher_0)[::-1][:100]
        top_100_indices_from1 = np.argsort(plot_teacher_1)[::-1][:100]
        top_100_indices = []
        ind_0, ind_1 = 0, 0
        for i in range(50):
            while True:
                if top_100_indices_from0[ind_0] not in top_100_indices:
                    top_100_indices.append(top_100_indices_from0[ind_0])
                    ind_0+=1
                    break
                else: ind_0+=1
            while True:
                if top_100_indices_from1[ind_1] not in top_100_indices:
                    top_100_indices.append(top_100_indices_from1[ind_1])
                    ind_1+=1
                    break
                else: ind_1+=1
        top_100_indices = np.array(top_100_indices)
        top_100_indices = np.sort(top_100_indices)
        plot_teacher_0 = plot_teacher_0[top_100_indices]
        plot_teacher_1 = plot_teacher_1[top_100_indices]
        string_top_100_indices = [str(i) for i in top_100_indices]
        self.logger.report_histogram(
            f"iteration:{iteration} teacher_out 0 S:{temp.item():.2f}, C:{self.center.mean().item():.2f}",
            "value",
            # iteration=epoch,
            values=plot_teacher_0,
            # xlabels=string_top_100_indices,
            xaxis="title x",
            yaxis="title y",
        )
        self.logger.report_histogram(
            f"iteration:{iteration} teacher_out 1 S:{temp.item():.2f}, C:{self.center.mean().item():.2f}",
            "value",
            # iteration=epoch,
            values=plot_teacher_1,
            # xlabels=string_top_100_indices,
            xaxis="title x",
            yaxis="title y",
        )

        ### plotting student arcface softmax output
        plot_view = np.random.randint(0, self.ncrops)
        plot_student=(F.softmax(student_sharpend_out[plot_view][plot_minibatch], dim=-1)).detach().cpu().float().numpy()
        plot_student = plot_student[top_100_indices]
        self.logger.report_histogram(
            f"iteration:{iteration} student_out {plot_view} w/ arcface,softmax S:{self.student_temp:.2f}",
            "value",
            # iteration=epoch,
            values=plot_student,
            xaxis="title x",
            yaxis="title y",
        )

        ### plotting student raw softmax output
        if student_raw_out is not None:
            student_raw_out = student_raw_out.chunk(self.ncrops)
            # plot_view = np.random.randint(0, self.ncrops)
            plot_student=F.softmax(student_raw_out[plot_view][plot_minibatch], dim=-1)
            plot_student = plot_student[top_100_indices]
            self.logger.report_histogram(
                f"iteration:{iteration} student_out {plot_view} w/o arcface w/ softmax S:{self.student_temp:.2f}",
                "value",
                # iteration=epoch,
                values=plot_student,
                xaxis="title x",
                yaxis="title y",
            )

        ### plotting student arcface output
        # plot_view = np.random.randint(0, self.ncrops)
        plot_student=(student_sharpend_out[plot_view][plot_minibatch]*self.student_temp).detach().cpu().float().numpy()
        plot_student = plot_student[top_100_indices]
        self.logger.report_histogram(
            f"iteration:{iteration} student_out {plot_view} w/ arcface w/o softmax",
            "value",
            # iteration=epoch,
            values=plot_student,
            xaxis="title x",
            yaxis="title y",
        )

        if student_raw_out is not None:
            ### plotting student raw output
            plot_student=student_raw_out[plot_view][plot_minibatch]
            plot_student = plot_student[top_100_indices]
            self.logger.report_histogram(
                f"iteration:{iteration} student_out {plot_view} w/o arcface,softmax",
                "value",
                # iteration=epoch,
                values=plot_student,
                xaxis="title x",
                yaxis="title y",
            )

class PrototypeLoss(nn.Module):
    """
    DINOのプロトタイプベクトル学習用の損失関数。
    ArcFace等のマージン付き損失やPU学習の重み付けにも対応。
    """
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9, args=None, logger=None, unl_pos_weight=None, student_temp_schedule=None, 
                 margin_settings=None):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))
        self.batch_size = args.batch_size_per_gpu
        self.logger = logger
        self.unl_pos_weight = unl_pos_weight
        self.student_temp_schedule = student_temp_schedule
        self.register_buffer('m', torch.zeros(args.batch_size_per_gpu, args.prototype_num))

        if margin_settings is not None:
            self.use_margin = True
            self.margin = margin_settings["m"]
            self.easy_margin = margin_settings["easy_margin"]
            self.warmup = margin_settings["warmup"]
        else:
            self.use_margin = False
            self.margin = 0.0

    def forward(self, student_output, teacher_output, epoch, iteration=0, 
                class_ids=None, student_raw_out=None, pos_output=False, pos_batch=False, 
                use_weight=False, margin_type=False, normal_prototype_dict=None, positive_prototype_dict=None, pt_num=10):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """

        if self.student_temp_schedule is not None:
            self.student_temp = self.student_temp_schedule[iteration]

        student_out = student_output.chunk(self.ncrops)
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = (teacher_output - self.center)
        teacher_out = teacher_out.detach().chunk(2)

        if use_weight and torch.is_tensor(class_ids):
            unl_pos = class_ids[:,0]
            batch_weight = (1-unl_pos)*self.unl_pos_weight["unl"] + unl_pos*self.unl_pos_weight["pos"]
            batch_weight = batch_weight.to("cuda")
        else:
            batch_weight = None
        
        ##########################################
        freeze_ep=50
        if epoch>freeze_ep and margin_type != False and self.margin > 0.0:
            unl_pos = class_ids[:,0]
            norm_margin_index = torch.tensor(list(normal_prototype_dict.keys())[:pt_num])
            pos_margin_index = torch.tensor(list(positive_prototype_dict.keys())[:pt_num])
            margin=True
        else:
            margin = None
        ##########################################

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                if margin == None:
                    loss = torch.sum(-F.softmax(q/temp, dim=-1) * F.log_softmax(student_out[v]/self.student_temp, dim=-1), dim=-1)              
                elif margin_type.lower() == 'student':
                    loss = torch.sum(-F.softmax(q/temp, dim=-1) * F.log_softmax(self.additive_margin(student_out[v], unl_pos, norm_margin_index, pos_margin_index, iteration)/self.student_temp, dim=-1), dim=-1)
                elif margin_type.lower() == 'teacher':
                    loss = torch.sum(-F.softmax(self.additive_margin(q, unl_pos, norm_margin_index, pos_margin_index, iteration, sub=True)/temp, dim=-1) * F.log_softmax(student_out[v]/self.student_temp, dim=-1), dim=-1)
                elif margin_type.lower() == 'tea-stu' or margin_type.lower() == 'stu-tea':
                    loss = torch.sum(-F.softmax(self.additive_margin(q, unl_pos, norm_margin_index, pos_margin_index, iteration, sub=True)/temp, dim=-1) * F.log_softmax(self.additive_margin(student_out[v], unl_pos, norm_margin_index, pos_margin_index, iteration)/self.student_temp, dim=-1), dim=-1)
                else:
                    raise NotImplementedError
                
                if batch_weight is not None:
                    loss = loss * batch_weight
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        #####
        if self.logger and (iteration%100==0 or iteration==0):
            if class_ids is not None:
                self.visualize_output(student_output.detach().clone().chunk(self.ncrops), self.student_temp, teacher_output.detach().clone().chunk(2), self.center, temp, iteration, class_ids[:,1])
        #####

        return total_loss

    def additive_margin(self, cosine, unl_pos, norm_m_ids, pos_m_ids, it, sub=False):
        dtype = cosine.dtype
        margin =  self.margin * self.warmup[it]
        cos_m = math.cos(margin)
        sin_m = math.sin(margin)

        ## Additive
        if not sub:
            th = math.cos(math.pi - margin)
            mm = math.sin(math.pi - margin) * margin
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(1e-7, 1))
            phi = cosine * cos_m - sine * sin_m
            phi = phi.to(dtype)
            phi = torch.clamp(phi, -1 + 1e-7, 1 - 1e-7)
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > th, phi, cosine - mm)
        ### Subtractive
        else:
            s_th = math.cos(margin)
            s_mm = math.sin(margin) * margin
            sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(1e-7, 1))
            phi = cosine * cos_m + sine * sin_m
            phi = phi.to(dtype)
            phi = torch.clamp(phi, -1 + 1e-7, 1 - 1e-7)
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine < s_th, phi, cosine + s_mm)

        pos_row = unl_pos.nonzero().squeeze()
        unl_row = (1-unl_pos).nonzero().squeeze()

        norm_margin_pos_ids = torch.cartesian_prod(pos_row, norm_m_ids)
        norm_margin_unl_ids = torch.cartesian_prod(unl_row, norm_m_ids)
        pos_margin_pos_ids = torch.cartesian_prod(pos_row, pos_m_ids)
        pos_margin_unl_ids = torch.cartesian_prod(unl_row, pos_m_ids)

        one_hot = torch.zeros(cosine.size(), device='cuda')
        for carten in [norm_margin_unl_ids, pos_margin_pos_ids]:
            one_hot[carten[:,0], carten[:,1]] = 1
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return output

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)
    
    def visualize_output(self, student_output, student_temp, teacher_output, center, teacher_temp, iteration, class_ids):
        # breakpoint()
        unique_indices = defaultdict(list)
        class_ids = class_ids.tolist()
        for index, value in enumerate(class_ids):
            unique_indices[value].append(index)
        unique_indices = dict(sorted(unique_indices.items(), key=lambda x:x[0]))
        selected_indices = {k:random.choice(list(v)) for k,v in unique_indices.items()}

        ### plotting each class teacher softmax emb
        for class_id, index in selected_indices.items():
            plot_teacher_0=teacher_output[0][index].cpu().float().numpy() #emb_dim
            self.logger.report_histogram(
            f"iteration:{iteration} class_id:{class_id} teacher_emb view:0 w/o softmax",
            "value",
            values=plot_teacher_0,
            xaxis="title x", yaxis="title y",
            )

            plot_teacher_0=teacher_output[0][index].cpu().float() #emb_dim
            plot_teacher_0 = (F.softmax((plot_teacher_0 - center.cpu())/teacher_temp, dim=-1)).numpy() #emb_dim
            self.logger.report_histogram(
            f"iteration:{iteration} class_id:{class_id} teacher_emb view:0 softmax S:{teacher_temp.item():.2f}, C:{center.mean().item():.2f}",
            "value",
            values=plot_teacher_0,
            xaxis="title x", yaxis="title y",
            )

            plot_teacher_1=teacher_output[1][index].cpu().float().numpy() #emb_dim
            self.logger.report_histogram(
            f"iteration:{iteration} class_id:{class_id} teacher_emb view:1 w/o softmax",
            "value",
            values=plot_teacher_1,
            xaxis="title x", yaxis="title y",
            )

            plot_teacher_1=teacher_output[1][index].cpu().float() #emb_dim
            plot_teacher_1 = (F.softmax((plot_teacher_1 - center.cpu())/teacher_temp, dim=-1)).numpy()#emb_dim
            self.logger.report_histogram(
            f"iteration:{iteration} class_id:{class_id} teacher_emb view:1 softmax S:{teacher_temp.item():.2f}, C:{center.mean().item():.2f}",
            "value",
            values=plot_teacher_1,
            xaxis="title x", yaxis="title y",
            )
            ### plotting student emb
            plot_view = np.random.randint(0, self.ncrops)
            plot_student=(student_output[plot_view][index]).cpu().float().numpy()
            self.logger.report_histogram(
                f"iteration:{iteration} class_id:{class_id} student_emb view:{plot_view} w/o sharpning, softmax",
                "value",
                # iteration=epoch,
                values=plot_student,
                xaxis="title x", yaxis="title y",
            )
            ### plotting student emb softmax
            plot_view = np.random.randint(0, self.ncrops)
            plot_student=(F.softmax(student_output[plot_view][index]/student_temp, dim=-1)).cpu().float().numpy()
            self.logger.report_histogram(
                f"iteration:{iteration} class_id:{class_id} student_emb view:{plot_view} w/ sharpning{student_temp}, softmax",
                "value",
                # iteration=epoch,
                values=plot_student,
                xaxis="title x", yaxis="title y",
            )
       
### Default Losses
class Loss(nn.Module):
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class AELoss(Loss):
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean((1 - target) * output)


class ABCLoss(Loss):
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y_positive = -torch.log1p(-torch.exp(-output))
        y_unlabeled = output
        return torch.mean((1 - target) * y_unlabeled + target * y_positive)


class DeepSADLoss(Loss):
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y_positive = 1 / (output + 1e-6)
        y_unlabeled = output
        return torch.mean((1 - target) * y_unlabeled + target * y_positive)


class PUBaseLoss(Loss):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        positive = target
        unlabeled = 1 - target

        n_positive = torch.sum(positive).clamp_min(1)
        n_unlabeled = torch.sum(unlabeled).clamp_min(1)

        y_positive = self.positive_loss(output)
        y_unlabeled = self.unlabeled_loss(output)

        positive_risk = torch.sum(self.alpha * positive * y_positive / n_positive)
        negative_risk = torch.sum(
            (unlabeled / n_unlabeled - self.alpha * positive / n_positive) * y_unlabeled
        )

        if negative_risk < 0:
            return -1 * negative_risk
        else:
            return positive_risk + negative_risk

    def positive_loss(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def unlabeled_loss(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class PULoss(PUBaseLoss):
    def positive_loss(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(-x)

    def unlabeled_loss(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)


class PUAELoss(PUBaseLoss):
    def __init__(self, alpha: float):
        super().__init__(alpha=alpha)

    def positive_loss(self, x: torch.Tensor) -> torch.Tensor:
        return -torch.log1p(-torch.exp(-x))

    def unlabeled_loss(self, x: torch.Tensor) -> torch.Tensor:
        return x


class PUSVDDLoss(PUBaseLoss):
    def __init__(self, alpha: float):
        super().__init__(alpha=alpha)

    def positive_loss(self, x: torch.Tensor) -> torch.Tensor:
        return -torch.log(1 - torch.exp(-x) + 1e-6)

    def unlabeled_loss(self, x: torch.Tensor) -> torch.Tensor:
        return x
