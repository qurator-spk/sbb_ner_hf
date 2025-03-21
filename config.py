from dataclasses import dataclass, field

#general
task = "ner"

# resources
@dataclass
class Resource:
    path: str
    source: str
    name : str = "default_name"

    def set_name(self):
        name = self.path.split("/")[-1]
        self.name = name
    
    def info(self):
        desc_str = "%s will be loaded from %s (via %s)." % (self.name, self.path, self.source)
        return desc_str

#training, optimization and evaluation parameters
@dataclass
class TrainingParams:
    batch_size: int = 16
    eval_strategy: str = "epoch"
    learning_rate: int = 2e-5
    num_train_epochs: int = 3
    weight_decay: int = 0.01
    warmup_steps: int = 100

@dataclass
class OptimizeParams:
    #search parameters
    hp_space: dict = field(default_factory=lambda: {
        "learning_rate": {
            "param_type": "float",
            "borders": [1e-6, 1e-4]
        },
        "warmup_steps": {
            "param_type": "categorical",
            "borders": [20, 40, 160]
        }
    })
    #number of trials while searching for optimal hyperparameter configuration
    n_trials: int = 20

"""
        "per_device_train_batch_size": {
            "param_type": "categorical",
            "borders": [16, 32, 64, 128, 256, 512, 1024]
        },
        "per_device_eval_batch_size": {
            "param_type": "categorical",
            "borders": [16, 32, 64, 128, 256, 512, 1024]
        }
"""