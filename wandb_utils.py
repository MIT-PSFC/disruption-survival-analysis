import os
import socket
import wandb

def set_up_wandb(model, training_args, seed):
    """Set up wandb for logging.
    
    Args:
        model (PlasmaTransformer): The model to be trained.
        training_args (TrainingArguments): The training arguments.
        seed (int): The random seed.
        
    Returns:
        None
    """
    wandb.init(project="HDL-improvement-transformer",
               entity="mit_psfc",)

    if not check_wandb_connection():
        os.environ["WANDB_MODE"] = "offline"

    # zip model.config and training_args into a single dictionary
    wandb_config = {**vars(model.config), **vars(training_args)}
    wandb.config.update(wandb_config)
    wandb.log({"seed": seed})
    # os.environ["WANDB_LOG_MODEL"] = "end"

    return


def check_wandb_connection(host="api.wandb.ai", port=443, timeout=5):
    """Check if the machine is connected to wandb."""
    try:
        socket.setdefaulttimeout(timeout)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((host, port))
            return True
    except socket.error:
        return False
    