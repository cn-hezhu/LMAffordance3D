"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import logging
import os

import torch
import torch.distributed as dist
import numpy as np
from sklearn.metrics import roc_auc_score
from lavis.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.common.registry import registry
from lavis.datasets.data_utils import prepare_sample
from lavis.tasks.base_task import BaseTask
from torch.cuda.amp import autocast as autocast
from .utils import calculate_SIM

@registry.register_task("affordance")
class AffordanceTask(BaseTask):
    def __init__(self, setting):
        super().__init__()

        self.inst_id_key = "instance_id"
        self.setting = setting
        self.num_train = 0
        self.num_val = 0
        self.object_cls_list = []
        self.affordance_cls_list = []

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        setting = cfg.run_cfg.setting
        return cls(setting)

    def build_model(self, cfg):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def build_datasets(self, cfg):
        """
        Build a dictionary of datasets, keyed by split 'train', 'valid', 'test'.
        Download dataset and annotations automatically if not exist.

        Args:
            cfg (common.config.Config): _description_

        Returns:
            dict: Dictionary of torch.utils.data.Dataset objects by split.
        """

        datasets = dict()

        datasets_config = cfg.datasets_cfg

        assert len(datasets_config) > 0, "At least one dataset has to be specified."

        for name in datasets_config:
            dataset_config = datasets_config[name]

            builder = registry.get_builder_class(name)(dataset_config)
            dataset = builder.build_datasets()

            datasets[name] = dataset

        return datasets

    def train_step(self, model, samples):
        device = model.device
        img = samples['Img'].to(device)
        points = samples['Points_List']
        description = samples['Description']
        labels = samples['Affordance_label_List']
        object_cls = samples['Object_cls']
        affordance_cls = samples['Affordance_cls']
        self.num_train += img.shape[0]
        loss = 0
        loss_dict = {"loss" : 0, "loss_hm" : 0}
        for point, label in zip(points, labels):
            point, label = point.float(), label.float()
            point = point.to(device)
            label = label.to(device)
            label = label.unsqueeze(dim=-1)
            output = model(img, point, description, label)
            loss += output["loss"]
            for k, _ in loss_dict.items():
                loss_dict[k] += output[k]
        return loss, loss_dict

    def valid_step(self, model, samples):
        device = model.device
        img = samples['Img'].to(device)
        point = samples['Point'].to(device)
        description = samples['Description']
        label = samples['Affordance_label'].to(device)
        object_cls = samples['Object_cls']
        affordance_cls = samples['Affordance_cls']
        B = img.shape[0]
        self.num_val += B
        for i in range(B):
            self.object_cls_list.append(object_cls[i])
            self.affordance_cls_list.append(affordance_cls[i])
        point, label = point.float(), label.float()
        label = label.unsqueeze(dim=-1)
        loss = 0
        loss_dict = {"loss" : 0, "loss_hm" : 0}
        output = model(img, point, description, label)
        loss += output["loss"]
        for k, _ in loss_dict.items():
            loss_dict[k] += output[k]
        result = {"prediction" :output["out"], "ground_truth": label, "loss" : loss}
        return result

    def before_training(self, model, dataset, **kwargs):
        model.before_training(dataset=dataset, task_type=type(self))

    def before_evaluation(self, model, dataset, **kwargs):
        model.before_evaluation(dataset=dataset, task_type=type(self))

    def after_evaluation(self, val_result, epoch, writer=None, cur_epoch=None, output_dir=None, **kwargs):
        if(self.setting == 'Unseen'):
            object_list = ['Bed', 'Dishwasher','Microwave','Scissors','Vase', 'Laptop']
            affordance_list = ['contain', 'lay', 'sit', 'wrapgrasp','open','display','stab','grasp', 'press','cut']
        else:
            object_list = ['Vase', 'Display', 'Bed', 'Microwave', 'Door', 
            'Earphone', 'Bottle', 'Bowl', 'Laptop', 'Clock', 'Scissors', 'Mug', 'Faucet', 
            'StorageFurniture', 'Bag', 'Chair', 'Dishwasher', 'Refrigerator', 
            'Table', 'Hat', 'Keyboard', 'Knife', 'TrashCan']

            affordance_list = ['grasp', 'contain', 'lift', 'open', 
                            'lay', 'sit', 'support', 'wrapgrasp', 'pour', 'move', 'display',
                            'push', 'listen', 'wear', 'press', 'cut', 'stab']

        for obj in object_list:
            exec(f'{obj} = [[], [], [], []]')
        for aff in affordance_list:
            exec(f'{aff} = [[], [], [], []]')
        
        prediction = torch.zeros(self.num_val, 2048, 1)
        ground_truth = torch.zeros(self.num_val, 2048, 1)
        self.num_train = 0
        self.num_val = 0
        # total_MAE = 0
        total_point = 0
        total_loss = 0
        num = 0
        output = {}
        for each in val_result:
            # each_MAE = torch.sum(torch.abs(each["prediction"]-each["ground_truth"]), dim=(0,1))
            # total_MAE += each_MAE.item()
            points_num = each["prediction"].shape[0] * each["prediction"].shape[1]
            total_point += points_num
            total_loss += each["loss"].item()
            pred_num = each["prediction"].shape[0]
            prediction[num : num+pred_num, :, :] = each["prediction"]
            ground_truth[num : num+pred_num, :, :] = each["ground_truth"]
            num += pred_num

        # MAE = total_MAE / total_point
        prediction = prediction.detach().numpy()
        ground_truth = ground_truth.detach().numpy()
        SIM_matrix = np.zeros(ground_truth.shape[0])
        MAE_martrix = np.zeros(ground_truth.shape[0])
        for i in range(ground_truth.shape[0]):
            SIM_matrix[i] = calculate_SIM(prediction[i], ground_truth[i])
            MAE_martrix[i] = np.sum(np.absolute(prediction[i]-ground_truth[i])) / 2048
            object_cls = self.object_cls_list[i]
            aff_cls = self.affordance_cls_list[i]
            exec(f'{object_cls}[1].append({SIM_matrix[i]})')
            exec(f'{aff_cls}[1].append({SIM_matrix[i]})')
            exec(f'{object_cls}[3].append({MAE_martrix[i]})')
            exec(f'{aff_cls}[3].append({MAE_martrix[i]})')
        
        SIM = np.mean(SIM_matrix)
        MAE = np.mean(MAE_martrix)
        AUC = np.zeros((ground_truth.shape[0], ground_truth.shape[2]))
        IOU = np.zeros((ground_truth.shape[0], ground_truth.shape[2]))
        IOU_thres = np.linspace(0, 1, 20)
        ground_truth = ground_truth >= 0.5
        ground_truth = ground_truth.astype(int)
        for i in range(AUC.shape[0]):
            t_true = ground_truth[i]
            p_score = prediction[i]
            object_cls = self.object_cls_list[i]
            aff_cls = self.affordance_cls_list[i]

            if np.sum(t_true) == 0:
                AUC[i] = np.nan
                IOU[i] = np.nan
                exec(f'{object_cls}[2].append(np.nan)')
                exec(f'{aff_cls}[2].append(np.nan)')
                exec(f'{object_cls}[0].append(np.nan)')
                exec(f'{aff_cls}[0].append(np.nan)')
            else:
                auc = roc_auc_score(t_true, p_score)
                AUC[i] = auc

                p_mask = (p_score > 0.5).astype(int)
                temp_iou = []
                for thre in IOU_thres:
                    p_mask = (p_score >= thre).astype(int)
                    intersect = np.sum(p_mask & t_true)
                    union = np.sum(p_mask | t_true)
                    temp_iou.append(1.*intersect/union)
                temp_iou = np.array(temp_iou)
                aiou = np.mean(temp_iou)
                IOU[i] = aiou
                exec(f'{object_cls}[2].append(auc)')   
                exec(f'{aff_cls}[2].append(auc)')
                exec(f'{object_cls}[0].append(aiou)')
                exec(f'{aff_cls}[0].append(aiou)')
        
        AUC = np.nanmean(AUC)
        IOU = np.nanmean(IOU)
        print('------Object-------')
        for obj in object_list:
            aiou = np.nanmean(eval(obj)[0])
            sim_ = np.mean(eval(obj)[1])
            auc_ = np.nanmean(eval(obj)[2])
            mae_ = np.mean(eval(obj)[3])
            print(f'{obj} | IOU:{aiou} | SIM:{sim_} | AUC:{auc_}')

        avg_mertics = [0, 0, 0, 0]
        print('------Affordance-------')
        for i,aff in enumerate(affordance_list):
            aiou = np.nanmean(eval(aff)[0])
            sim_ = np.mean(eval(aff)[1])
            auc_ = np.nanmean(eval(aff)[2])
            mae_ = np.mean(eval(aff)[3])
            avg_mertics[0] += aiou
            avg_mertics[1] += sim_
            avg_mertics[2] += auc_
            avg_mertics[3] += mae_
            print(f'{aff} | IOU:{aiou} | SIM:{sim_} | AUC:{auc_}, MAE:{mae_}')

        logging.info('AUC : %f, IOU : %f, SIM : %f, MAE : %f' % (AUC, IOU, SIM, MAE))
        output['auc_metric'] = AUC
        output['iou_metric'] = IOU
        output['sim_metric'] = SIM
        output['mae_metric'] = MAE
        
        prediction = np.squeeze(prediction)
        output['prediction'] = prediction
        output_file = os.path.join(output_dir, "prediction.txt")
        np.savetxt(output_file, prediction, fmt='%.6f', delimiter=' ')

        return output

    def inference_step(self):
        raise NotImplementedError

    def evaluation(self, model, data_loader, cuda_enabled=True):
        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 50
        results = []

        for samples in metric_logger.log_every(data_loader, print_freq, header):
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            with torch.cuda.amp.autocast(enabled=True):
                eval_output = self.valid_step(model=model, samples=samples)
            results.append(eval_output)

        if is_dist_avail_and_initialized():
            dist.barrier()

        return results

    def train_epoch(
        self,
        epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
        writer=None,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            iters_per_epoch=len(data_loader),
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
            writer=writer,
        )

    def train_iters(
        self,
        epoch,
        start_iters,
        iters_per_inner_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        cuda_enabled=False,
        log_freq=50,
        accum_grad_iters=1,
    ):
        return self._train_inner_loop(
            epoch=epoch,
            start_iters=start_iters,
            iters_per_epoch=iters_per_inner_epoch,
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
            log_freq=log_freq,
            cuda_enabled=cuda_enabled,
            accum_grad_iters=accum_grad_iters,
        )

    def _train_inner_loop(
        self,
        epoch,
        iters_per_epoch,
        model,
        data_loader,
        optimizer,
        lr_scheduler,
        scaler=None,
        start_iters=None,
        log_freq=50,
        cuda_enabled=False,
        accum_grad_iters=1,
        writer=None,
    ):
        """
        An inner training loop compatible with both epoch-based and iter-based training.

        When using epoch-based, training stops after one epoch; when using iter-based,
        training stops after #iters_per_epoch iterations.
        """
        use_amp = scaler is not None

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4f}"))

        # if iter-based runner, schedule lr based on inner epoch.
        logging.info(
            "Start training epoch {}, {} iters per inner epoch.".format(
                epoch, iters_per_epoch
            )
        )
        header = "Train: data epoch: [{}]".format(epoch)
        if start_iters is None:
            # epoch-based runner
            inner_epoch = epoch
        else:
            # In iter-based runner, we schedule the learning rate based on iterations.
            inner_epoch = start_iters // iters_per_epoch
            header = header + "; inner epoch [{}]".format(inner_epoch)

        for i in metric_logger.log_every(range(iters_per_epoch), log_freq, header):
            # if using iter-based runner, we stop after iters_per_epoch iterations.
            if i >= iters_per_epoch:
                break

            samples = next(data_loader)

            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            samples.update(
                {
                    "epoch": inner_epoch,
                    "num_iters_per_epoch": iters_per_epoch,
                    "iters": i,
                }
            )

            lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss, loss_dict = self.train_step(model=model, samples=samples)
                loss /= accum_grad_iters #TODO: not affect loss_dict values for logging

            if is_main_process():
                for key in loss_dict:
                    if isinstance(loss_dict[key], float) or isinstance(loss_dict[key], int):
                        writer.add_scalar('train/%s_iter' % key, loss_dict[key], epoch*iters_per_epoch+i)
                    else:
                        writer.add_scalar('train/%s_iter' % key, loss_dict[key].item(), epoch*iters_per_epoch+i)

            # after_train_step()
            if use_amp:
                    scaler.scale(loss).backward()
            else:
                    loss.backward()

            # update gradients every accum_grad_iters iterations
            if (i + 1) % accum_grad_iters == 0:
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

            metric_logger.update(**loss_dict)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # after train_epoch()
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        if is_main_process():
            for key, meter in metric_logger.meters.items():
                writer.add_scalar('train/%s_epoch' % key, meter.global_avg, epoch)
        return {
            k: "{:.3f}".format(meter.global_avg)
            for k, meter in metric_logger.meters.items()
        }

    @staticmethod
    def save_result(result, result_dir, filename, remove_duplicate=""):
        import json

        result_file = os.path.join(
            result_dir, "%s_rank%d.json" % (filename, get_rank())
        )
        final_result_file = os.path.join(result_dir, "%s.json" % filename)

        json.dump(result, open(result_file, "w"))

        if is_dist_avail_and_initialized():
            dist.barrier()

        if is_main_process():
            logging.warning("rank %d starts merging results." % get_rank())
            # combine results from all processes
            result = []

            for rank in range(get_world_size()):
                result_file = os.path.join(
                    result_dir, "%s_rank%d.json" % (filename, rank)
                )
                res = json.load(open(result_file, "r"))
                result += res

            if remove_duplicate:
                result_new = []
                id_list = []
                for res in result:
                    if res[remove_duplicate] not in id_list:
                        id_list.append(res[remove_duplicate])
                        result_new.append(res)
                result = result_new

            json.dump(result, open(final_result_file, "w"))
            print("result file saved to %s" % final_result_file)

        return final_result_file
