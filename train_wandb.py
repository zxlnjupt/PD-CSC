import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from accelerate import Accelerator
from ema_pytorch import EMA
from pathlib import Path
from shutil import rmtree
import numpy as np
import torch.nn.functional as F
import wandb
import pickle
from sklearn.model_selection import KFold
import datetime

# 注意：请确保文件名与你保存的新模型文件一致
# 建议将文件名改为 hapticstream_rvq.py 以避免混淆
from model.hapticstream_fourier import HapticStream 
from src.args import parse_args
from src.optimizer import get_optimizer
from src.dataloader import Haptic_DataLoader, Haptic_Dataset
from src.utils import cycle, yes_or_no, noop, accum_log, exists, checkpoint_num_steps

args = parse_args()
torch.manual_seed(42)

class HapticStreamTrainer(nn.Module):
    def __init__(
            self,
            haptic_stream: HapticStream,
            *,
            # 建议修改结果文件夹名称以区分实验
            results_folder: str = './results_rvq_experiment_128_3_fold0_516', 
            use_ema: bool = False,
    ):
        super().__init__()
        self.accelerator = Accelerator(log_with="all", project_dir="./logging_results")
        self.haptic_stream = haptic_stream
        self.use_ema = use_ema
        self.results_folder = Path(results_folder)

        if self.use_ema:
            self.ema_haptic_stream = EMA(haptic_stream,
                                         beta=args.ema_beta,
                                         update_after_step=args.ema_update_after_step,
                                         update_every=args.ema_update_every)

        self.register_buffer('steps', torch.Tensor([0]))

        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.grad_accum_every = args.grad_accum_every
        self.lr = args.lr
        self.wd = args.weight_decay
        self.max_grad_norm = args.max_grad_norm
        self.save_model_every_epoch = args.save_model_every_epoch
        self.log_losses_every = args.log_losses_every
        self.apply_grad_penalty_every = args.apply_grad_penalty_every

        self.optim = get_optimizer(self.unwrapped_haptic_stream.non_discr_parameters(), lr=self.lr, wd=self.wd)

        for discr_optimizer_key, discr in self.multiscale_discriminator_iter():
            one_multiscale_discr_optimizer = get_optimizer(discr.parameters(), lr=self.lr, wd=self.lr)
            setattr(self, discr_optimizer_key, one_multiscale_discr_optimizer)

        # --- Dataset ---
        self.num_folds = 5
        self.val_fold_idx = 0
        
        all_folds = []
        for fold in range(self.num_folds):
            fold_path = f"{args.root_path}/fold_{fold}"
            all_folds.append(Haptic_Dataset(root_path=fold_path, data_path="", json_name="data_path_list.json"))

        from torch.utils.data import ConcatDataset
        train_datasets = [all_folds[i] for i in range(self.num_folds) if i != self.val_fold_idx]
        combined_train_dataset = ConcatDataset(train_datasets)
        val_dataset = all_folds[self.val_fold_idx]

        if self.accelerator.is_main_process:
            print(f"=== Dataset Setup ===")
            print(f"Validation Fold: {self.val_fold_idx}")

        self.train_dataLoader, self.valid_dataLoader = Haptic_DataLoader(
            combined_train_dataset,
            batch_size=args.batch_size,
            dataset2=val_dataset,
            num_workers=args.num_workers
        ).get_dataloader()

        total_samples = len(combined_train_dataset)
        effective_batch_size = args.batch_size * self.grad_accum_every
        self.steps_per_epoch = (total_samples + effective_batch_size - 1) // effective_batch_size
        
        # --- Prepare ---
        discr_optims = [getattr(self, key) for key, _ in self.multiscale_discriminator_iter()]
        
        prepared_objects = self.accelerator.prepare(
            self.haptic_stream, 
            self.optim, 
            self.train_dataLoader, 
            self.valid_dataLoader,
            *discr_optims
        )
        
        self.haptic_stream = prepared_objects[0]
        self.optim = prepared_objects[1]
        self.train_dataLoader = prepared_objects[2]
        self.valid_dataLoader = prepared_objects[3]
        
        for i, (key, _) in enumerate(self.multiscale_discriminator_iter()):
            setattr(self, key, prepared_objects[4 + i])

        self.train_dataLoader_iter = cycle(self.train_dataLoader)

        if self.accelerator.is_main_process:
            if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('Clear previous results?'):
                rmtree(str(self.results_folder))
            self.results_folder.mkdir(parents=True, exist_ok=True)

        self.best_recon_loss = float('inf')
        self.best_total_loss = float('inf')
        self.current_epoch_tracker = 0

    @property
    def unwrapped_haptic_stream(self):
        return self.accelerator.unwrap_model(self.haptic_stream)

    def multiscale_discriminator_iter(self):
        for ind, discr in enumerate(self.unwrapped_haptic_stream.discriminators):
            yield f'multiscale_discr_optimizer_{ind}', discr

    def save(self, path):
        if not self.accelerator.is_main_process:
            return
        pkg = dict(
            model=self.accelerator.get_state_dict(self.haptic_stream),
            optim=self.optim.state_dict(),
            config=self.unwrapped_haptic_stream._configs,
        )
        if self.use_ema:
            pkg['ema_model'] = self.ema_haptic_stream.state_dict()
        for key, _ in self.multiscale_discriminator_iter():
            discr_optim = getattr(self, key)
            pkg[key] = discr_optim.state_dict()
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location='cpu')
        self.unwrapped_haptic_stream.load_state_dict(pkg['model'])
        if self.use_ema:
            self.ema_haptic_stream.load_state_dict(pkg['ema_model'])
        self.optim.load_state_dict(pkg['optim'])
        for key, _ in self.multiscale_discriminator_iter():
            discr_optim = getattr(self, key)
            discr_optim.load_state_dict(pkg[key])
        self.steps = torch.tensor([checkpoint_num_steps(path) + 1], device=self.accelerator.device)
    def train_step(self):
        device = self.accelerator.device
        steps = int(self.steps.item())
        apply_grad_penalty = self.apply_grad_penalty_every > 0 and not (steps % self.apply_grad_penalty_every)
        log_losses = self.log_losses_every > 0 and not (steps % self.log_losses_every)

        self.haptic_stream.train()
        logs = {}

        # 1. Generator
        for _ in range(self.grad_accum_every):
            haptic_wave = next(self.train_dataLoader_iter)
            haptic_wave = haptic_wave.float().to(device)

            # --- [Modification Start] 接收 commit_loss ---
            # 新的 forward 返回: (total_loss, (recon, adv, feature, commit, tau, delta, perceptual))
            loss, (recon_loss, adversarial_loss, feature_loss, commit_loss, tau_mean, delta_mean, perceptual_loss) = \
                self.haptic_stream(haptic_wave, return_loss_breakdown=True)
            # --- [Modification End] ---

            self.accelerator.backward(loss / self.grad_accum_every)
            
            # --- [Modification Start] 记录 commit_loss ---
            accum_log(logs, dict(
                loss=loss.item() / self.grad_accum_every,
                recon_loss=recon_loss.item() / self.grad_accum_every,
                adversarial_loss=adversarial_loss.item() / self.grad_accum_every,
                feature_loss=feature_loss.item() / self.grad_accum_every,
                commit_loss=commit_loss.item() / self.grad_accum_every, # 新增记录
                perceptual_loss=perceptual_loss / self.grad_accum_every
            ))
            # --- [Modification End] ---

        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.haptic_stream.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()

        # 2. Discriminator
        for name, multiscale_discr_optim in self.multiscale_discriminator_iter():
            getattr(self, name).zero_grad()

        for _ in range(self.grad_accum_every):
            haptic_wave = next(self.train_dataLoader_iter)
            haptic_wave = haptic_wave.float().to(device)
            discr_losses = self.haptic_stream(
                haptic_wave,
                apply_grad_penalty=apply_grad_penalty,
                return_discr_loss=True,
                return_discr_losses_separately=True
            )
            for name, discr_loss in discr_losses:
                self.accelerator.backward(discr_loss / self.grad_accum_every, retain_graph=True)
                accum_log(logs, {name: discr_loss.item() / self.grad_accum_every})

        for name, multiscale_discr_optim in self.multiscale_discriminator_iter():
            getattr(self, name).step()

        # 3. 日志记录
        if log_losses:
            tracker_logs = {
                "train/total_loss": logs['loss'],
                "train/recon_loss": logs['recon_loss'],
                "train/adversarial_loss": logs['adversarial_loss'],
                "train/feature_loss": logs['feature_loss'],
                "train/commit_loss": logs['commit_loss'], # 新增记录到 wandb
                "train/perceptual_loss": logs['perceptual_loss'],
                "train/tau_mean": tau_mean,
                "train/delta_mean": delta_mean,
                "epoch": self.current_epoch_tracker
            }
            
            for key, loss in logs.items():
                if key.startswith('scale:'):
                    tracker_logs[f"train/discr_loss_{key}"] = loss
            
            self.accelerator.log(tracker_logs, step=steps)

            if self.accelerator.is_main_process:
                # 打印日志稍微更新一下，把 commit loss 也打印出来（可选）
                print(f"Step {steps}: Total: {logs['loss']:.4f} | Recon: {logs['recon_loss']:.4f} | Commit: {logs['commit_loss']:.4f} | Perceptual: {logs['perceptual_loss']:.4f}")

        if self.use_ema:
            self.ema_haptic_stream.update()
        self.steps += 1
        return logs

    def _evaluate_on_validation_set(self, current_epoch):
        self.haptic_stream.eval()
        total_recon_loss = 0
        total_samples = 0
        
        if self.accelerator.is_main_process:
            print(f"\n===== Evaluating on Fold {self.val_fold_idx} (Epoch {current_epoch}) =====")
        
        with torch.no_grad():
            for batch_idx, haptic_wave in enumerate(self.valid_dataLoader):
                haptic_wave = haptic_wave.float().to(self.accelerator.device)
                batch_size = haptic_wave.size(0)
                
                with torch.inference_mode():
                    recons = self.haptic_stream(haptic_wave, return_recons_only=True)
                    recon_loss = F.mse_loss(haptic_wave, recons)
                
                total_recon_loss += recon_loss.item() * batch_size
                total_samples += batch_size
        
        avg_recon_loss = total_recon_loss / total_samples
        
        if self.accelerator.is_main_process:
            print(f"Validation Complete. Avg Recon Loss: {avg_recon_loss:.6f}")
            self.accelerator.log(
                {"val/recon_loss": avg_recon_loss, "epoch": current_epoch}, 
                step=int(self.steps.item())
            )
        
        return avg_recon_loss

    def train_start(self):
        if self.accelerator.is_main_process:
            current_time = datetime.datetime.now().strftime("%m%d-%H%M")
            # 更新运行名称，体现 RVQ
            run_name = f"Fold4-RVQ-{current_time}-ep{self.num_epochs}"
            
            self.accelerator.init_trackers(
                project_name="HapticStreamProject", 
                config={
                    "epochs": self.num_epochs,
                    "batch_size": self.batch_size,
                    "lr": self.lr,
                    "model": "RVQ" 
                },
                init_kwargs={"wandb": {"name": run_name}} 
            )

        print(f"Starting training for {self.num_epochs} epochs...")
        
        for epoch in range(self.num_epochs):
            self.current_epoch_tracker = epoch
            
            self.haptic_stream.train()
            for _ in range(self.steps_per_epoch):
                logs = self.train_step()

            val_loss = self._evaluate_on_validation_set(epoch)
            
            if self.accelerator.is_main_process:
                wandb_tracker = self.accelerator.get_tracker("wandb")
                if wandb_tracker:
                     wandb_tracker.run.summary["epoch_val_loss"] = val_loss

                if val_loss < self.best_recon_loss:
                    self.best_recon_loss = val_loss
                    best_path = str(self.results_folder / "best_model_recon.pt")
                    self.save(best_path)
                    print(f"New Best Recon Loss! Model saved to {best_path}")
                    if wandb_tracker:
                        wandb_tracker.run.summary["best_recon_loss"] = self.best_recon_loss

                if (epoch + 1) % self.save_model_every_epoch == 0:
                    ckpt_path = str(self.results_folder / f'checkpoint_epoch_{epoch+1}.pt')
                    self.save(ckpt_path)
                    print(f"Checkpoint saved: {ckpt_path}")
            
            self.accelerator.wait_for_everyone()
            
        print("Training Complete!")
        self.accelerator.end_training()

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.set_start_method('spawn', force=True)
    
    # 可以在这里显式传入参数，或者使用默认参数
    # 例如：trainer = HapticStreamTrainer(HapticStream(rvq_dim=16, num_quantizers=8))
    trainer = HapticStreamTrainer(HapticStream()) 
    trainer.train_start()