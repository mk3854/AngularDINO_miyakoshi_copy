"""
学習ループ・epoch処理系の関数
"""
import sys
import math
import torch
import utils
from torch.utils.data import DataLoader
import numpy as np
from eval_utils import test_anomaly_detection

def my_train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule, epoch,
                    fp16_scaler, args, logger, unl_normalize_subset, pos_normalize_subset, lambda_schedule,
                    student_prototype_weight, teacher_prototype_weight, s_prototype_optimizer, prototype_loss, test_dataset, val_dataset,
                    normal_class, normal_prototype_dict, positive_prototype_dict):
    """
    1エポック分の学習処理。
    - 学習率・重み減衰のスケジューリング
    - forward/backward
    - optimizer step
    - teacherのEMA更新
    - ロギング
    - 検証・可視化
    """
    torch.autograd.set_detect_anomaly(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    epoch_total_valid = []
    for it, (images, class_ids) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # --- optimizerのスケジューリング ---
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0: # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]
        for i, param_group2 in enumerate(s_prototype_optimizer.param_groups):
            param_group2["lr"] = lr_schedule[it]
            if i == 0: # only the first group is regularized
                param_group2["weight_decay"] = wd_schedule[it]

        images = [im.cuda(non_blocking=True) for im in images]

        # --- forward & loss計算 ---
        with torch.cuda.amp.autocast(enabled=args.precision != 'fp32', dtype=torch.bfloat16 if args.precision == 'bf16' else torch.float16):
            (teacher_output, max_id), t_emb = teacher(images[:2], "teacher", epoch=epoch) # teacherの出力と中間埋め込みを取得
            (student_output, _), s_emb = student(images, "student", max_id=None, epoch=epoch) # studentの出力と中間埋め込みを取得
            loss_dino = dino_loss(student_output, teacher_output, epoch, it, 
                                                class_ids=class_ids, pos_output=False, pos_batch=False, use_weight=args.use_weight) # 通常のDINO損失を計算
            s_dot = student_prototype_weight(s_emb) # Student側プロトタイプレイヤーと中間埋め込みの内積を計算
            t_dot = teacher_prototype_weight(t_emb) # Teacher側プロトタイプレイヤーと中間埋め込みの内積を計算
            # プロトタイプレイヤーの出力から損失を計算
            loss_prototype = prototype_loss(s_dot, t_dot, epoch, it, 
                                class_ids, pos_output=False, pos_batch=False, use_weight=args.use_weight, margin_type=args.margin_type, 
                                normal_prototype_dict=normal_prototype_dict, positive_prototype_dict=positive_prototype_dict, pt_num=args.pt_num)
            report_margin = prototype_loss.margin * prototype_loss.warmup[it] # 記録用のマージンを計算
            loss = loss_dino + lambda_schedule[it]*loss_prototype

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # --- backward & optimizer step ---
        optimizer.zero_grad()
        s_prototype_optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward(retain_graph=True)
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
        # extra optimizer
        if args.clip_grad:
            fp16_scaler.unscale_(s_prototype_optimizer)
            param_norms = utils.clip_gradients(student_prototype_weight, args.clip_grad)
        if epoch < args.freeze_last_layer:
            for n, p in student_prototype_weight.named_parameters():
                    p.grad = None
        fp16_scaler.step(s_prototype_optimizer)
        fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)
            for param_q, param_k in zip(student_prototype_weight.module.parameters(), teacher_prototype_weight.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # --- ロギング ---
        torch.cuda.synchronize()
        metric_logger.update(total_loss=loss.item())
        metric_logger.update(loss_dino=loss_dino.item())
        metric_logger.update(loss_lambda_emb=lambda_schedule[it]*loss_prototype.item())
        metric_logger.update(loss_emb=loss_prototype.item())
        metric_logger.update(emb_s_temp=prototype_loss.student_temp)
        metric_logger.update(margin=report_margin)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        metric_logger.update(lambda_=lambda_schedule[it])
        if utils.is_main_process() and logger:
            logger.report_scalar("train", "total_loss", iteration=it, value=loss.item())
            logger.report_scalar("train", "dino_loss", iteration=it, value=loss_dino.item())
            logger.report_scalar("train", "lambda_prototype_loss", iteration=it, value=lambda_schedule[it]*loss_prototype.item())
            logger.report_scalar("train", "prototype_loss", iteration=it, value=loss_prototype.item())
            logger.report_scalar("margin", "margin", iteration=it, value=report_margin)
            logger.report_scalar("temp", "prototype_student_temp", iteration=it, value=prototype_loss.student_temp)
            logger.report_scalar("Learning_Rate", "lr", iteration=it, value=optimizer.param_groups[0]["lr"])
            logger.report_scalar("Weight decay", "wb", iteration=it, value=optimizer.param_groups[0]["weight_decay"])
            logger.report_scalar("lambda", "lb", iteration=it, value=lambda_schedule[it])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    
    # Teacher側のbackboneおよびプロトタイプレイヤーでテストデータの異常検知を行う
    auroc, normal_prototype_dict, positive_prototype_dict = test_anomaly_detection(
                                                teacher_without_ddp, unl_normalize_subset, pos_normalize_subset, test_dataset, 
                                                teacher_prototype_weight, args.output_dir, epoch, args.pt_num)

    # 評価・プロトタイプ推定はeval_utils.pyに分離予定
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, 1e6, normal_prototype_dict, positive_prototype_dict

def validation(val_datset, teacher, student, teacher_prototype_weight, student_prototype_weight, dino_loss, prototype_loss, epoch):
    teacher.eval()
    student.eval()
    teacher_prototype_weight.eval()
    student_prototype_weight.eval()
    dino_loss.eval()
    prototype_loss.eval()
    test_loader = DataLoader(val_datset, batch_size=100, shuffle=True)
    for samples, gts in test_loader:
        images = [im.cuda(non_blocking=True) for im in samples]
        with torch.no_grad():
            (teacher_output, max_id), t_emb = teacher(images[:2], "teacher", epoch=epoch)
            (student_output, _), s_emb = student(images, "student", max_id=None, epoch=epoch)
            loss_dino, *_ = dino_loss(student_output, teacher_output, epoch, iteration=-1)
            loss_dino = loss_dino.cpu().item()
            s_dot = student_prototype_weight(s_emb)
            t_dot = teacher_prototype_weight(t_emb)
            loss_emb, *_ = prototype_loss(s_dot, t_dot, epoch, iteration=-1, class_ids=None)
            loss_emb = loss_emb.cpu().item()
            total_loss = loss_dino + loss_emb
            total_loss = total_loss
    teacher.train()
    student.train()
    teacher_prototype_weight.train()
    student_prototype_weight.train()
    dino_loss.train()
    prototype_loss.train()
    return total_loss, loss_dino, loss_emb

def converge_cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0, converge_epoch=None):
    """
    Cosine scheduler with early convergence.

    Args:
        base_value: 初期学習率。
        final_value: 最終学習率。
        epochs: 全エポック数。
        niter_per_ep: 各エポックあたりのイテレーション数。
        warmup_epochs: ウォームアップエポック数。
        start_warmup_value: ウォームアップ時の初期学習率。
        converge_epoch: 学習率を最終値に収束させるエポック。

    Returns:
        np.ndarray: 学習率スケジュール。
    """
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)
    total_iters = epochs * niter_per_ep
    if converge_epoch is None:
        converge_epoch = epochs
    converge_iter = converge_epoch * niter_per_ep
    cosine_iters = converge_iter - warmup_iters
    cosine_schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * np.arange(cosine_iters) / cosine_iters))
    remaining_iters = total_iters - converge_iter
    constant_schedule = np.full(remaining_iters, final_value)
    schedule = np.concatenate((warmup_schedule, cosine_schedule, constant_schedule))
    assert len(schedule) == total_iters
    return schedule
