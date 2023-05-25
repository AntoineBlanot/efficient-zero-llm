from typing import Callable, Dict, Any
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from accelerate import Accelerator

from args import TrainingArgs

class Trainer():

    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader: DataLoader = None,
        eval_dataloader: DataLoader = None,
        tokenizer: AutoTokenizer = None,
        optimizer: torch.optim = None,
        scheduler: torch.optim.lr_scheduler = None,
        train_args: TrainingArgs = None,
        compute_metrics: Callable = None
    ) -> None:
        
        self.accelerator = Accelerator(log_with="wandb")
        train_dataloader, eval_dataloader, model, optimizer, scheduler = self.accelerator.prepare(
            train_dataloader, eval_dataloader,
            model, optimizer, scheduler
        )

        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.compute_metrics = compute_metrics

        self.args = train_args

    def train(self) -> Dict[str, Any]:
        self.model.train()
        Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)
        self.accelerator.init_trackers(project_name=self.args.wandb_project, init_kwargs={'wandb': {'name': self.args.wandb_name}})
        
        max_train_steps = self.args.max_steps if self.args.max_steps is not None else len(self.train_dataloader)
        
        train_pbar = tqdm(range(max_train_steps), desc='Training')
        train_step, train_metrics, eval_metrics = 0, {}, {}
        all_loss = []

        while not self.should_stop_train(current_step=train_step, max_steps=max_train_steps):
            for inputs in self.train_dataloader:
                train_step += 1
                train_pbar.update()

                self.optimizer.zero_grad()

                outputs = self.model(**inputs)
                loss = outputs['loss']

                self.accelerator.backward(loss)
                self.optimizer.step()
                self.scheduler.step()

                all_loss.append(loss.detach().cpu().unsqueeze(0))

                if self.should_log(current_step=train_step):
                    loss = torch.cat(all_loss).mean().item()
                    lr = self.scheduler.get_last_lr()[0]
                    epoch = round(train_step / len(self.train_dataloader), 3)
                    train_metrics = dict(epoch=epoch, loss=loss, lr=lr)

                    train_pbar.set_postfix({**train_metrics, **{'eval.' + k: v for k,v in eval_metrics.items()}})
                    self.accelerator.log({'train': train_metrics}, step=train_step)

                    all_loss = []

                if self.should_eval(current_step=train_step):
                    eval_metrics = self.eval()
                    self.model.train()

                    train_pbar.set_postfix({**train_metrics, **{'eval.' + k: v for k,v in eval_metrics.items()}})
                    self.accelerator.log({'eval': eval_metrics}, step=train_step)

                if self.should_save(current_step=train_step):
                    save_path = Path(f'{self.args.output_dir}/checkpoint-{train_step}')
                    self.model.config.save_pretrained(save_path)
                    self.model.save_pretrained(save_path)
                    self.tokenizer.save_pretrained(save_path)

                if self.should_stop_train(current_step=train_step, max_steps=max_train_steps):
                    self.accelerator.end_training()
                    break

        final_metrics = dict(
            **dict(train=train_metrics),
            **dict(eval=eval_metrics)
        )

        return final_metrics

    def eval(self) -> Dict[str, float]:
        self.model.eval()

        max_eval_steps = len(self.eval_dataloader)

        eval_pbar = tqdm(range(max_eval_steps), desc='Evaluation')
        metrics = {}
        all_loss, all_preds, all_labels = [], [], []

        for inputs in self.eval_dataloader:
            eval_pbar.update()
            
            labels = inputs['labels']

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs['logits']
                loss = outputs['loss']

            preds = logits.argmax(-1)
            preds, labels = self.accelerator.gather_for_metrics((preds, labels))

            all_loss.append(loss.detach().cpu().unsqueeze(0))
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        loss = torch.cat(all_loss).mean().item()
        
        metrics.update({'loss': loss})
        if self.compute_metrics:
            metrics.update(self.compute_metrics(all_preds, all_labels))

        return metrics
    
    def should_stop_train(self, current_step: int, max_steps: int) -> bool:
        return current_step >= max_steps
    
    def should_log(self, current_step: int) -> bool:
        return (current_step % self.args.log_steps) == 0
    
    def should_eval(self, current_step: int) -> bool:
        return (current_step % self.args.eval_steps) == 0
    
    def should_save(self, current_step: int) -> bool:
        return (current_step % self.args.save_steps) == 0
