import torch
import numpy as np
import os
from sklearn.metrics import f1_score
from tqdm import tqdm


class Trainer():
    def __init__(self, args, optimizer, lr_scheduler, reg_loss_fn, clf_loss_fn, sl_loss_fn, reg_evaluator,
                 clf_evaluator, result_tracker, summary_writer, device, ddp=False, local_rank=0):
        self.args = args
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.reg_loss_fn = reg_loss_fn
        self.clf_loss_fn = clf_loss_fn
        self.sl_loss_fn = sl_loss_fn
        self.reg_evaluator = reg_evaluator
        self.clf_evaluator = clf_evaluator
        self.result_tracker = result_tracker
        self.summary_writer = summary_writer
        self.device = device
        self.ddp = ddp
        self.local_rank = local_rank
        self.n_updates = 0

    def _forward_epoch(self, model, batched_data):
        (smiles, batched_graph, fps, mds, sl_labels, disturbed_fps, disturbed_mds) = batched_data
        batched_graph = batched_graph.to(self.device)
        fps = fps.to(self.device)
        mds = mds.to(self.device)
        sl_labels = sl_labels.to(self.device)
        disturbed_fps = disturbed_fps.to(self.device)
        disturbed_mds = disturbed_mds.to(self.device)
        sl_predictions, fp_predictions, md_predictions = model(batched_graph, disturbed_fps, disturbed_mds)
        sl_loss = self.sl_loss_fn(sl_predictions, sl_labels).mean()
        fp_loss = self.clf_loss_fn(fp_predictions, fps).mean()
        md_loss = self.reg_loss_fn(md_predictions, mds).mean()
        loss = (sl_loss + fp_loss + md_loss) / 3
        mask_replace_keep = batched_graph.ndata['mask'][batched_graph.ndata['mask'] >= 1].detach().cpu().numpy()
        return loss, sl_loss, fp_loss, md_loss, mask_replace_keep, sl_predictions.detach(), sl_labels, fp_predictions.detach(), fps

    def train_epoch(self, model, train_loader, epoch_idx):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch_idx}", disable=(self.local_rank != 0))
        for batch_idx, batched_data in enumerate(pbar):
            self.optimizer.zero_grad()
            # 前向传播
            loss, sl_loss, fp_loss, md_loss, mask_replace_keep, sl_predictions, sl_labels, fp_predictions, fps = self._forward_epoch(
                model, batched_data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            self.optimizer.step()
            self.n_updates += 1
            self.lr_scheduler.step()
            if self.local_rank == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "sl": f"{sl_loss.item():.3f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}"
                })
            if self.summary_writer is not None and self.n_updates % 10 == 0:
                sl_preds_cpu = sl_predictions.cpu()
                sl_labels_cpu = sl_labels.cpu()
                preds = np.argmax(sl_preds_cpu.numpy(), axis=-1)
                labels = sl_labels_cpu.numpy()
                self.summary_writer.add_scalar('Loss/loss_tot', loss.item(), self.n_updates)
                self.summary_writer.add_scalar('Loss/loss_bert', sl_loss.item(), self.n_updates)
                self.summary_writer.add_scalar('Loss/loss_clf', fp_loss.item(), self.n_updates)
                self.summary_writer.add_scalar('Loss/loss_reg', md_loss.item(), self.n_updates)
                self.summary_writer.add_scalar('F1_micro/all', f1_score(preds, labels, average='micro'), self.n_updates)
            if self.n_updates >= self.args.n_steps:
                if self.local_rank == 0:
                    print(f"saving final pre-trained kamt model...")
                    self.save_model(model, name="final_kamt_model.pth")
                return True

    def fit(self, model, train_loader):
        for epoch in range(1, 1001):
            if self.ddp:
                train_loader.sampler.set_epoch(epoch)
            finished = self.train_epoch(model, train_loader, epoch)
            if self.local_rank == 0:
                self.save_model(model, name="checkpoint_last_epoch.pth")
            if finished:
                break

    def save_model(self, model, name=None):
        if self.local_rank != 0:
            return
        if not os.path.exists(self.args.save_path):
            os.makedirs(self.args.save_path)
        file_name = name if name else f"{self.args.config}.pth"
        save_file = os.path.join(self.args.save_path, file_name)
        state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        torch.save(state_dict, save_file)
        print(f"model saved at: {save_file}")