import sys
import os
import argparse
import torch
import random
import numpy as np
import warnings
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

try:
    import distutils.version
except AttributeError:
    from packaging import version
    import distutils

    distutils.version = version
os.environ["SETUPTOOLS_USE_DISTUTILS"] = "stdlib"

sys.path.append('..')

from tools.utils import set_random_seed
from tools.data.featurizer import Vocab, N_BOND_TYPES, N_ATOM_TYPES, smiles_to_graph
from tools.data.pretrain_dataset import MoleculeDataset
from tools.data.collator import Collator_pretrain

from models.kamt import KaMT
from tools.trainer.scheduler import PolynomialDecayLR
from tools.trainer.pretrain_trainer import Trainer
from tools.trainer.evaluator import Evaluator
from tools.trainer.result_tracker import Result_Tracker
from tools.model_config import config_dict
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="KAMT Pre-training Script")
    parser.add_argument("--seed", type=int, default=22)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--n_steps", type=int, default=100000)
    parser.add_argument("--config", type=str, default="base")
    parser.add_argument("--n_threads", type=int, default=8)
    parser.add_argument("--n_devices", type=int, default=1)
    return parser.parse_args()


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

if __name__ == '__main__':
    args = parse_args()
    config = config_dict[args.config]
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(local_rank)
    if 'RANK' in os.environ:
        torch.distributed.init_process_group(backend='nccl')
        is_ddp = True
    else:
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.distributed.init_process_group(backend='nccl', rank=0, world_size=1)
        is_ddp = False
    device = torch.device('cuda', local_rank)
    set_random_seed(args.seed)
    if local_rank == 0:
        print(f"Start KAMT pre-training...")
        print(f"Model Configuration: {args.config}")
        print(f"Data Path: {args.data_path}")
    vocab = Vocab(N_ATOM_TYPES, N_BOND_TYPES)
    collator = Collator_pretrain(
        vocab,
        max_length=config['path_length'],
        n_virtual_nodes=2,
        candi_rate=config['candi_rate'],
        fp_disturb_rate=config.get('fp_disturb_rate', 0.15),
        md_disturb_rate=config.get('md_disturb_rate', 0.15)
    )
    train_dataset = MoleculeDataset(root_path=args.data_path)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=config['batch_size'] // args.n_devices,
        num_workers=args.n_threads,
        worker_init_fn=seed_worker,
        drop_last=True,
        collate_fn=collator
    )
    sample_data = train_dataset[0]
    sample_smiles = sample_data[0]
    temp_graph = smiles_to_graph(sample_smiles, vocab, max_length=config['path_length'], n_virtual_nodes=2)
    actual_node_in_dim = temp_graph.ndata['begin_end'].shape[-1]
    actual_edge_in_dim = temp_graph.ndata['edge'].shape[-1]
    if local_rank == 0:
        print(f"Feature Detection Results - Node Dimension: {actual_node_in_dim}, Edge Dimension: {actual_edge_in_dim}")
    config['d_node_feats'] = actual_node_in_dim
    config['d_edge_feats'] = actual_edge_in_dim
    config['d_fp_feats'] = train_dataset.d_fps
    config['d_md_feats'] = train_dataset.d_mds
    config['n_main_tasks'] = vocab.vocab_size
    model = KaMT(config).to(device)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False
    )
    optimizer = Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    lr_scheduler = PolynomialDecayLR(
        optimizer,
        warmup_updates=20000,
        tot_updates=args.n_steps,
        lr=config['lr'],
        end_lr=1e-9,
        power=1
    )
    reg_loss_fn = MSELoss(reduction='none')
    clf_loss_fn = BCEWithLogitsLoss(weight=train_dataset._task_pos_weights.to(device), reduction='none')
    sl_loss_fn = CrossEntropyLoss(reduction='none')
    reg_evaluator = Evaluator("chembl", "r2", train_dataset.d_mds)
    clf_evaluator = Evaluator("chembl", "rocauc_resp", train_dataset.d_fps)
    result_tracker = Result_Tracker("r2")
    summary_writer = None
    if local_rank == 0:
        summary_writer = SummaryWriter(f"tensorboard/pretrain-kamt-{args.config}")
    trainer = Trainer(
        args,
        optimizer,
        lr_scheduler,
        reg_loss_fn,
        clf_loss_fn,
        sl_loss_fn,
        reg_evaluator,
        clf_evaluator,
        result_tracker,
        summary_writer,
        device=device,
        ddp=True,
        local_rank=local_rank
    )
    trainer.fit(model, train_loader)
    if local_rank == 0:
        summary_writer.close()
    torch.distributed.destroy_process_group()