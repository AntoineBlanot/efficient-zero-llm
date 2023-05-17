from dataclasses import dataclass, field

@dataclass
class ModelArgs:
    """Class for model arguments"""

    pretrained_model_name_or_path: str = field(metadata={'help': 'Local path or name of the model on HuggingFace\'s hub'})


@dataclass
class DataArgs:
    """Class for dataset arguments"""
    
    path: str = field(metadata={'help': 'Local path of the dataset'})
    bs: int = field(metadata={'help': 'Batch size'})
    seq_length: int = field(metadata={'help': 'Maximum sequence length (longer inputs are truncated)'})


@dataclass
class TrainingArgs:
    """Class for training arguments"""

    output_dir: str = field(metadata={'help': 'Name of the local directory where checkpoints will be saved'})

    lr: float = field(metadata={'help': 'Maximum learning rate'})
    wd: float = field(metadata={'help': 'Weight decay'})

    warmup_steps: int  = field(metadata={'help': 'Number of warmup steps for the learning rate scheduler'})
    max_steps: int = field(metadata={'help': 'Maximum number of training steps'})
    log_steps: int = field(metadata={'help': 'Steps interval for logging'})
    eval_steps: int = field(metadata={'help': 'Steps interval for evaluation'})
    save_steps: int = field(metadata={'help': 'Steps interval for saving'})

    wandb_project: str = field(metadata={'help': 'Name of the wandb project'})
    wandb_name: str = field(metadata={'help': 'Name of the wandb run'})
