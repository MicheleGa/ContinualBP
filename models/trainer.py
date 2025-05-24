import os
import sys
folders_to_add = ['data', 'training_utils']
for folder in folders_to_add:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), folder)))
import shutil
import gc
import copy
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import pytorch_lightning as pl
from ResGRUNet import ResGRUNet
from PhysioFormer import PhysioFormer
from data.preprocessing_utils.augmentations import RandomAugmentor, Identity, Jitter, Scaling, MagnitudeWarp, Flip
from training_utils.helpers import save_status, load_status, EarlyStopping, set_trainable_parameters, configure_optimizer_and_scheduler, get_model_architecture
from training_utils.metrics import AverageMeter, log_meter_to_tensorboard, setup_meter, update_meter, get_metric_values, get_simclr_metric_values

class SupervisedModel(pl.LightningModule):
    def __init__(self, model_kwargs):
        super().__init__()
        self.save_hyperparameters()
        print(model_kwargs)
        if model_kwargs['model_name'] == 'ResGRUNet':   
            self.model = ResGRUNet(
                ecg=model_kwargs['ecg'],
                resp=model_kwargs['resp'], 
                ppg_derivatives=model_kwargs['ppg_derivatives'], 
                ppg_emd=model_kwargs['ppg_emd'], 
                ppg_freqs=model_kwargs['ppg_freqs'],
                channels=model_kwargs['channels'], 
                kernel_size=model_kwargs['kernel_size'], 
                act=model_kwargs['act'], 
                pooling=model_kwargs['pooling'], 
                proj_head_dim=model_kwargs['proj_head_dim'], 
                input_seq_len=int(model_kwargs['input_seq_len_s'] * model_kwargs['fs']),
                return_embedding=model_kwargs['return_embedding'],
                set_tunable_params=model_kwargs['set_tunable_params']
                )
        elif model_kwargs['model_name'] == 'PhysioFormer':
            self.model = PhysioFormer(
                ecg=model_kwargs['ecg'],
                resp=model_kwargs['resp'], 
                ppg_derivatives=model_kwargs['ppg_derivatives'], 
                ppg_emd=model_kwargs['ppg_emd'], 
                ppg_freqs=model_kwargs['ppg_freqs'],
                embed_dim=model_kwargs['embed_dim'], 
                n_head=model_kwargs['n_head'], 
                head_dim=model_kwargs['head_dim'], 
                hidden_dim=model_kwargs['hidden_dim'],
                num_layers=model_kwargs['num_layers'], 
                input_seq_len=int(model_kwargs['input_seq_len_s'] * model_kwargs['fs']),
                return_embedding=model_kwargs['return_embedding'],
                set_tunable_params=model_kwargs['set_tunable_params']
                )
        else:
            raise ValueError("Invalid model name ...")
        
        in_channels = self.model.get_input_channels()
        self.example_input_array = torch.rand((model_kwargs['batchsize'], int(model_kwargs['input_seq_len_s'] * model_kwargs['fs']), (in_channels[0] + in_channels[1] + in_channels[2])))
               
        # Train Loss per epoch
        self.train_loss_epoch = [] # Initialize an empty list to store batch losses
 
        # Test Metrics
        self.test_outputs = np.empty((0, 2), dtype=float)
        self.test_targets = np.empty((0, 2), dtype=float)
        
        # Data Augmentation
        if model_kwargs['aug']:
            self.augments = RandomAugmentor(
                [
                    Identity(prob=0.2),
                    Jitter(prob=0.2),
                    Scaling(prob=0.2),
                    MagnitudeWarp(prob=0.2),
                    Flip(prob=0.2)
                ]
            )
        
    def forward(self, x):
        return self.model(x)
    
    def configure_optimizers(self):
        
        if self.hparams.model_kwargs['optimizer_type'] == 'RMSprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.hparams.model_kwargs['lr'], weight_decay=self.hparams.model_kwargs['l2norm'])
        elif self.hparams.model_kwargs['optimizer_type'] == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.model_kwargs['lr'], weight_decay=self.hparams.model_kwargs['l2norm'])
        elif self.hparams.model_kwargs['optimizer_type'] == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.model_kwargs['lr'], weight_decay=self.hparams.model_kwargs['l2norm'], momentum=self.hparams.model_kwargs['sgd_momentum'])
        elif self.hparams.model_kwargs['optimizer_type'] == 'AdamW':
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.model_kwargs['lr'], weight_decay=self.hparams.model_kwargs['l2norm'])
        else:
            raise ValueError("Invalid optimizer ...")

        if self.hparams.model_kwargs['lr_scheduler_enable']:
            if self.hparams.model_kwargs['lr_scheduler_type'] == 'MultiStepLR':
                scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    optimizer=optimizer,
                    milestones=tuple([int(s) for s in self.hparams.model_kwargs['lrsched_step'].split(sep=",")]),
                    gamma=self.hparams.model_kwargs['lrsched_gamma']
                )
                
                return {"optimizer": optimizer, "lr_scheduler": {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }}
                
            elif self.hparams.model_kwargs['lr_scheduler_type'] == 'CosineAnnealingWarmupScheduler':
                warmup_steps = self.hparams.model_kwargs['lr_scheduler_warmup'] * self.hparams.model_kwargs['steps_per_epoch']
                annealing_steps = (self.hparams.model_kwargs['max_training_epochs'] - self.hparams.model_kwargs['lr_scheduler_warmup']) * self.hparams.model_kwargs['steps_per_epoch']

                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 
                                                                     start_factor=self.hparams.model_kwargs['lr_scheduler_min_lr'] / self.hparams.model_kwargs['lr'], 
                                                                     total_iters=warmup_steps)
                cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, 
                    T_max=annealing_steps, 
                    eta_min=self.hparams.model_kwargs['lr_scheduler_min_lr'])

                lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[warmup_steps]
                )

                return {"optimizer": optimizer, "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"}}
            else:
                raise ValueError("Invalid lr scheduler ...")
            
        else:
            return optimizer
        
    def supcon_loss(self, features, labels=None, mask=None):
        r"""
        Supervised Contrastive Loss 
        from https://github.com/pulp-platform/fscil/blob/main/code/lib/torch_blocks.py#L114
        and from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html#SimCLR-implementation
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                            'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # Shape: [batch_size * n_views, embedding_dim]

        # Normalize embeddings
        contrast_feature = F.normalize(contrast_feature, dim=1)

        # Compute cosine similarity
        cos_sim = torch.matmul(contrast_feature, contrast_feature.T)  # Shape: [batch_size * n_views, batch_size * n_views]

        # Mask out self-contrast cases
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=device)
        cos_sim.masked_fill_(self_mask, -9e15)

        # Create positive pair mask
        if labels is not None:
            # Supervised case: Use labels to create positive pairs
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            labels = torch.cat([labels] * contrast_count, dim=0)  # Repeat labels for all views
            pos_mask = torch.eq(labels, labels.T).float().to(device)
        else:
            # Unsupervised case: Assume positive pairs are `batch_size // 2` apart
            pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0).float()

        # Scale cosine similarity by temperature
        cos_sim = cos_sim / self.hparams.model_kwargs['temperature']

        # Compute InfoNCE loss
        log_prob = cos_sim - torch.logsumexp(cos_sim, dim=-1, keepdim=True)
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)

        # Handle cases where pos_mask.sum(1) == 0
        mean_log_prob_pos[pos_mask.sum(1) == 0] = 0

        # Final loss
        loss = -mean_log_prob_pos.mean()

        return loss
    
    def training_step(self, batch, batch_idx):
        
        # Prepare input data
        signals = batch[0]
        _, _, num_modalities = signals.shape
        
        # Prepare labels
        sbp = batch[1][0]
        dbp = batch[1][1] 
        target = torch.cat((sbp, dbp), dim=-1)
        
        # Create two augmented views of the signals
        if self.hparams.model_kwargs['return_embedding'] and self.hparams.model_kwargs['aug']:
            augmented_signals_1 = torch.empty_like(signals, device=target.device)
            augmented_signals_2 = torch.empty_like(signals, device=target.device)
            for mod in range(num_modalities):
                # Apply augmentations to the entire batch for the current modality
                augmented_mod_1 = self.augments(signals[:, :, mod].clone().cpu())
                augmented_signals_1[:, :, mod] = augmented_mod_1.to(target.device)

                augmented_mod_2 = self.augments(signals[:, :, mod].clone().cpu())
                augmented_signals_2[:, :, mod] = augmented_mod_2.to(target.device)
    
        # Forward propagation
        if self.hparams.model_kwargs['return_embedding'] and self.hparams.model_kwargs['aug']:
            output, embedding = self.model(signals)
            embedding_1 = self.model(augmented_signals_1)[1]
            embedding_2 = self.model(augmented_signals_2)[1]
        else:
            output = self.model(signals)
        
        # Supervised loss 
        if self.hparams.model_kwargs['criterion'] == 'MSELoss':
            loss = F.mse_loss(output, target)
        elif self.hparams.model_kwargs['criterion'] == 'SmoothL1Loss':
            loss = F.smooth_l1_loss(output, target)
        else:
            raise ValueError("Invalid criterion ...")

        if self.hparams.model_kwargs['lambda_contrastive'] != 0.:
            loss = self.hparams.model_kwargs['lambda_contrastive'] * loss
            
        # Contrastive loss (SimCLR)
        if self.hparams.model_kwargs['lambda_contrastive'] != 0. and self.hparams.model_kwargs['temperature'] != 0. and self.hparams.model_kwargs['return_embedding'] and self.hparams.model_kwargs['aug']:
            embeddings = torch.stack([embedding_1, embedding_2], dim=1)  # Shape: [batch_size, n_views, embedding_dim]
            contrastive_loss = self.supcon_loss(embeddings)
            loss = loss + self.hparams.model_kwargs['lambda_contrastive'] * contrastive_loss

        # Add orthogonal loss
        if self.hparams.model_kwargs['lambda_ortho'] != 0. and self.hparams.model_kwargs['return_embedding']:
            proto = F.normalize(embedding, dim=0, p=2)
            loss_reg = proto.t() @ proto - torch.eye(proto.shape[1], device=proto.device)
            loss_reg = torch.mean(loss_reg * loss_reg)
            loss = loss + self.hparams.model_kwargs['lambda_ortho'] * loss_reg

        self.log('train/loss', loss)
        
        self.train_loss_epoch.append(loss.detach().cpu())

        return loss  
    
    def on_train_epoch_end(self):
        
        avg_loss = torch.stack(self.train_loss_epoch).mean()
        
        self.log('train/loss_epoch', avg_loss)
        
        self.train_loss_epoch.clear() # Clear the list for the next epoch

    def validation_step(self, batch, batch_idx):
        
        # Prepare input data
        signals = batch[0]
        
        # Prepare labels
        sbp = batch[1][0]
        dbp = batch[1][1] 
        target = torch.cat((sbp, dbp), dim=-1)
        
        # Forward propagation
        if self.hparams.model_kwargs['return_embedding']:
            output, _ = self.model(signals)
        else:
            output = self.model(signals)
        
        # Loss 
        if self.hparams.model_kwargs['criterion'] == 'MSELoss':
            loss = F.mse_loss(output, target)
        elif self.hparams.model_kwargs['criterion'] == 'SmoothL1Loss':
            loss = F.smooth_l1_loss(output, target)
        else:
            raise ValueError("Invalid criterion ...")
        
        # Metrics
        sbp_error = output[:, 0] - target[:, 0]
        dbp_error = output[:, 1] - target[:, 1]

        sbp_mae = torch.mean(torch.abs(sbp_error), dim=-1)
        dbp_mae = torch.mean(torch.abs(dbp_error), dim=-1)
        sbp_mae_std = torch.std(torch.abs(sbp_error), dim=-1)
        dbp_mae_std = torch.std(torch.abs(dbp_error), dim=-1)
        
        sbp_me = torch.mean(sbp_error, dim=-1)
        dbp_me = torch.mean(dbp_error, dim=-1)
        sbp_std = torch.std(sbp_error, dim=-1)
        dbp_std = torch.std(dbp_error, dim=-1)
        
        self.log('val/sbp_mae', sbp_mae, on_step=False, on_epoch=True)
        self.log('val/dbp_mae', dbp_mae, on_step=False, on_epoch=True)
        self.log('val/sbp_mae_std', sbp_mae_std, on_step=False, on_epoch=True)
        self.log('val/dbp_mae_std', dbp_mae_std, on_step=False, on_epoch=True)
        
        self.log('val/sbp_me', sbp_me, on_step=False, on_epoch=True)
        self.log('val/dbp_me', dbp_me, on_step=False, on_epoch=True)
        self.log('val/sbp_me_std', sbp_std, on_step=False, on_epoch=True)
        self.log('val/dbp_me_std', dbp_std, on_step=False, on_epoch=True)
        
        self.log('val/loss', loss)

    def test_step(self, batch, batch_idx):
        
        # Prepare input data
        signals = batch[0]
        
        # Prepare labels
        sbp = batch[1][0]
        dbp = batch[1][1] 
        target = torch.cat((sbp, dbp), dim=-1)

        # Forward propagation
        if self.hparams.model_kwargs['return_embedding']:
            output, _ = self.model(signals)
        else:
            output = self.model(signals)
        # Loss 
        if self.hparams.model_kwargs['criterion'] == 'MSELoss':
            loss = F.mse_loss(output, target)
        elif self.hparams.model_kwargs['criterion'] == 'SmoothL1Loss':
            loss = F.smooth_l1_loss(output, target)
        else:
            raise ValueError("Invalid criterion ...")
        
        # Metrics
        sbp_error = output[:, 0] - target[:, 0]
        dbp_error = output[:, 1] - target[:, 1]

        sbp_mae = torch.mean(torch.abs(sbp_error), dim=-1)
        dbp_mae = torch.mean(torch.abs(dbp_error), dim=-1)
        sbp_mae_std = torch.std(torch.abs(sbp_error), dim=-1)
        dbp_mae_std = torch.std(torch.abs(dbp_error), dim=-1)
        
        sbp_me = torch.mean(sbp_error, dim=-1)
        dbp_me = torch.mean(dbp_error, dim=-1)
        sbp_std = torch.std(sbp_error, dim=-1)
        dbp_std = torch.std(dbp_error, dim=-1)
        
        self.log('test/sbp_mae', sbp_mae, on_step=False, on_epoch=True)
        self.log('test/dbp_mae', dbp_mae, on_step=False, on_epoch=True)
        self.log('test/sbp_mae_std', sbp_mae_std, on_step=False, on_epoch=True)
        self.log('test/dbp_mae_std', dbp_mae_std, on_step=False, on_epoch=True)
        
        self.log('test/sbp_me', sbp_me, on_step=False, on_epoch=True)
        self.log('test/dbp_me', dbp_me, on_step=False, on_epoch=True)
        self.log('test/sbp_me_std', sbp_std, on_step=False, on_epoch=True)
        self.log('test/dbp_me_std', dbp_std, on_step=False, on_epoch=True)
        
        self.log('test/loss', loss)
        
        self.test_outputs = np.concatenate((self.test_outputs, output.detach().cpu().numpy()), axis=0)
        self.test_targets = np.concatenate((self.test_targets, target.detach().cpu().numpy()), axis=0)


def supcon_loss(features, config, labels=None, mask=None):
    r"""
    Supervised Contrastive Loss 
    from https://github.com/pulp-platform/fscil/blob/main/code/lib/torch_blocks.py#L114
    abd from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html#SimCLR-implementation
    """
    device = features.device

    if len(features.shape) < 3:
        raise ValueError('`features` needs to be [bsz, n_views, ...],'
                        'at least 3 dimensions are required')
    if len(features.shape) > 3:
        features = features.view(features.shape[0], features.shape[1], -1)

    batch_size = features.shape[0]
    contrast_count = features.shape[1]
    contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # Shape: [batch_size * n_views, embedding_dim]

    # Normalize embeddings
    contrast_feature = F.normalize(contrast_feature, dim=1)

    # Compute cosine similarity
    cos_sim = torch.matmul(contrast_feature, contrast_feature.T)  # Shape: [batch_size * n_views, batch_size * n_views]

    # Mask out self-contrast cases
    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=device)
    cos_sim.masked_fill_(self_mask, -9e15)

    # Create positive pair mask
    if labels is not None:
        # Supervised case: Use labels to create positive pairs
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        labels = torch.cat([labels] * contrast_count, dim=0)  # Repeat labels for all views
        pos_mask = torch.eq(labels, labels.T).float().to(device)
    else:
        # Unsupervised case: Assume positive pairs are `batch_size // 2` apart
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0).float()

    # Scale cosine similarity by temperature
    cos_sim = cos_sim / config['temperature']

    # Compute InfoNCE loss
    log_prob = cos_sim - torch.logsumexp(cos_sim, dim=-1, keepdim=True)
    mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)

    # Handle cases where pos_mask.sum(1) == 0
    mean_log_prob_pos[pos_mask.sum(1) == 0] = 0

    # Final loss
    loss = -mean_log_prob_pos.mean()

    # Return loss and metrics
    return loss, cos_sim, pos_mask


def pretraining_training_validation_testing(save_name, checkpoint_path, tensorboard_path, model_name, dataloaders, config, device):
    
    # Define model architecture
    model = get_model_architecture(config)
    model = model.to(device)
    
    # Set the model to training mode
    set_trainable_parameters(model, config['set_tunable_params'], config)
    
    # Print trainable and non-trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    print(f"Trainable parameters: {trainable_params} / {trainable_params + non_trainable_params} ({trainable_params / (trainable_params + non_trainable_params) * 100:.2f}%)")
    print(f"Non-trainable parameters: {non_trainable_params} / {trainable_params + non_trainable_params} ({non_trainable_params / (trainable_params + non_trainable_params) * 100:.2f}%)")
    
    # Dataloaders
    train_dataloader = dataloaders['train']
    val_dataloader = dataloaders['val']
    test_dataloader = dataloaders['test']
    
    # Optimizater and scheduler
    if config['lr_scheduler_enable'] and config['lr_scheduler_type'] == 'CosineAnnealingWarmupScheduler':
        config['steps_per_epoch'] = len(train_dataloader)
        
    optim_sched = configure_optimizer_and_scheduler(model, config)
    
    optimizer = optim_sched['optimizer']
    if config['lr_scheduler_enable']:
        scheduler = optim_sched['lr_scheduler']
    else:
        scheduler = None
    
    # Logging to TensorBoard Summary Writer
    writer = SummaryWriter(log_dir=tensorboard_path)
    
    in_channels = model.get_input_channels()
    example_input_array = torch.rand((config['batchsize'], int(config['input_seq_len_s'] * config['fs']), (in_channels[0] + in_channels[1] + in_channels[2])))
    writer.add_graph(model, example_input_array.to(device))

    # Best lowest val loss in case of supervised pretraining, otherwise highest accuracy of SimCLR
    best_val = float("+inf")
    
    # Early stopping
    if config['es_enable']:
        early_stopping = EarlyStopping(
            patience=config['es_patience'], 
            delta=config['es_min_delta'], 
            verbose=True,
            mode='min'
            )
    
    # Augmentations
    if config['return_embedding'] and config['aug']:
        augments = RandomAugmentor(
                [
                    Identity(prob=0.2),
                    Jitter(prob=0.2),
                    Scaling(prob=0.2),
                    MagnitudeWarp(prob=0.2),
                    Flip(prob=0.2)
                ]
            )
    
    # Pretraining loop
    for epoch in range(config['max_training_epochs']):
        
        # Training loop
        set_trainable_parameters(model, config['set_tunable_params'], config)
        train_losses = AverageMeter(name='train/loss')
        if config['lambda_supervised'] == 0.:
            metrics = ['loss', 'acc_top1', 'acc_top5', 'acc_mean_pos']
            simclr_meter = setup_meter('train_sim_clr', metrics)
            
        for batch_idx, batch in enumerate(train_dataloader):
            signals, targets = batch

            # Move data and labels to the same device as the model
            signals = signals.to(device)
            
            # Prepare input data
            _, _, num_modalities = signals.shape
            targets = torch.cat((batch[1][0].to(device), batch[1][1].to(device)), dim=-1)
            
            # Create two augmented views of the signals
            if config['return_embedding'] and config['aug']:
                augmented_signals_1 = torch.empty_like(signals, device=device)
                augmented_signals_2 = torch.empty_like(signals, device=device)
                for mod in range(num_modalities):
                    # Apply augmentations to the entire batch for the current modality
                    augmented_mod_1 = augments(signals[:, :, mod].clone().cpu())
                    augmented_signals_1[:, :, mod] = augmented_mod_1.to(device)

                    augmented_mod_2 = augments(signals[:, :, mod].clone().cpu())
                    augmented_signals_2[:, :, mod] = augmented_mod_2.to(device)
        
            # Forward propagation
            if config['return_embedding'] and config['aug']:
                outputs, embeddings = model(signals)
                embeddings_1 = model(augmented_signals_1)[1]
                embeddings_2 = model(augmented_signals_2)[1]
            elif config['return_embedding']:
                outputs, embeddings = model(signals)
            else:
                outputs = model(signals)
            
            # Supervised loss 
            if config['lambda_supervised'] != 0.:
                if config['criterion'] == 'MSELoss':
                    supervised_loss = F.mse_loss(outputs, targets)
                elif config['criterion'] == 'SmoothL1Loss':
                    supervised_loss = F.smooth_l1_loss(outputs, targets)
                else:
                    raise ValueError("Invalid criterion ...")
            else:
                supervised_loss = 0.
                
            # Contrastive loss (SimCLR)
            if config['lambda_contrastive'] != 0. and config['temperature'] != 0. and config['return_embedding'] and config['aug']:
                embeddings = torch.stack([embeddings_1, embeddings_2], dim=1)  # Shape: [batch_size, n_views, embedding_dim]
                contrastive_loss, cos_sim, pos_mask = supcon_loss(embeddings, config)
                
                simclr_metric_values = get_simclr_metric_values(contrastive_loss, cos_sim, pos_mask)
                update_meter(simclr_meter, simclr_metric_values, signals.size(0))
            else:
                contrastive_loss = 0.

            # Add orthogonal loss
            if config['lambda_ortho'] != 0. and config['return_embedding']:
                proto = F.normalize(embeddings, dim=0, p=2)
                loss_reg = proto.t() @ proto - torch.eye(proto.shape[1], device=proto.device)
                loss_reg = torch.mean(loss_reg * loss_reg)
            else:
                loss_reg = 0.
                        
            # Total loss
            loss = config['lambda_supervised'] * supervised_loss + config['lambda_contrastive'] * contrastive_loss + config['lambda_ortho'] * loss_reg
            
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # N.B. schedulers are usually called at the end of the epoch, but here we apply it at the end of each batch
            if config['lr_scheduler_enable']:
                scheduler.step()
            
            train_losses.update(loss.item(), signals.size(0))
            
            # Log training loss to TensorBoard
            writer.add_scalar('train/loss', loss.item(), epoch * len(train_dataloader) + batch_idx)
            
        # Log epoch training loss to TensorBoard
        writer.add_scalar('train/loss_epoch', train_losses.avg, epoch)
        
        # Validation loop    
        set_trainable_parameters(model, 'none', config)
        if config['lambda_supervised'] == 0.:
            metrics = ['loss', 'acc_top1', 'acc_top5', 'acc_mean_pos']
        else:
            metrics = ['loss', 'sbp_mae', 'dbp_mae', 'sbp_me', 'dbp_me', 'sbp_mae_std',  'dbp_mae_std', 'sbp_me_std', 'dbp_me_std']
        val_metrics = setup_meter('val', metrics)
            
        for batch_idx, batch in enumerate(val_dataloader):
            signals, targets = batch

            # Move data and labels to the same device as the model
            signals = signals.to(device)
            targets = torch.cat((batch[1][0].to(device), batch[1][1].to(device)), dim=-1)
            
            # Create two augmented views of the signals also in validation in case of self-supervised contrastive pretraining
            if config['return_embedding'] and config['aug'] and config['lambda_supervised'] == 0.:
                augmented_signals_1 = torch.empty_like(signals, device=device)
                augmented_signals_2 = torch.empty_like(signals, device=device)
                for mod in range(num_modalities):
                    # Apply augmentations to the entire batch for the current modality
                    augmented_mod_1 = augments(signals[:, :, mod].clone().cpu())
                    augmented_signals_1[:, :, mod] = augmented_mod_1.to(device)

                    augmented_mod_2 = augments(signals[:, :, mod].clone().cpu())
                    augmented_signals_2[:, :, mod] = augmented_mod_2.to(device)
            
            # Forward propagation
            if config['return_embedding'] and config['aug'] and config['lambda_supervised'] == 0.:
                outputs, embeddings = model(signals)
                embeddings_1 = model(augmented_signals_1)[1]
                embeddings_2 = model(augmented_signals_2)[1]
            elif config['return_embedding']:
                outputs, embeddings = model(signals)
            else:
                outputs = model(signals)
            
            # Supervised loss 
            if config['lambda_supervised'] != 0.:
                if config['criterion'] == 'MSELoss':
                    supervised_loss = F.mse_loss(outputs, targets)
                elif config['criterion'] == 'SmoothL1Loss':
                    supervised_loss = F.smooth_l1_loss(outputs, targets)
                else:
                    raise ValueError("Invalid criterion ...")
            else:
                supervised_loss = 0
                
            # Contrastive loss (SimCLR)
            if config['lambda_contrastive'] != 0. and config['temperature'] != 0. and config['return_embedding'] and config['aug'] and config['lambda_supervised'] == 0.:
                embeddings = torch.stack([embeddings_1, embeddings_2], dim=1)  # Shape: [batch_size, n_views, embedding_dim]
                contrastive_loss, cos_sim, pos_mask = supcon_loss(embeddings, config)
            else:
                contrastive_loss = 0
            
            # Note that in case we do contrastive learning, the accuracy in finding the positive pairs can be used as a metric for validation
            val_loss = config['lambda_supervised'] * supervised_loss + config['lambda_contrastive'] * contrastive_loss
            
            if config['lambda_supervised'] == 0.:
                metric_values = get_simclr_metric_values(val_loss, cos_sim, pos_mask)
            else:
                metric_values = get_metric_values(val_loss, outputs, targets)
                
            update_meter(val_metrics, metric_values, signals.size(0))
        
        # Log epoch validation metrics to TensorBoard
        log_meter_to_tensorboard(writer, val_metrics, epoch, name='val')
        
        # Log LR
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar(f'{config["optimizer_type"]}', current_lr, epoch)
        
        # Print losses
        print(f"Epoch {epoch + 1}/{config['max_training_epochs']} - Train Loss {train_losses.avg}, Val Loss {val_metrics['loss'].avg}")
        
        # Update best model 
        if val_metrics['loss'].avg < best_val:
            save_status(None, epoch, model_name, save_name, model, optimizer, scheduler, val_metrics, checkpoint_path, config)
            best_val = val_metrics['loss'].avg
        
        # Early stopping
        if config['es_enable']:
            early_stopping(val_metrics['loss'].avg)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
    # Load the best model
    model = get_model_architecture(config)
    load_status(None, model_name, save_name, model, optimizer, scheduler, checkpoint_path, config)
    model = model.to(device)

    # In case of self-supervised contrastive pretraining, we need to train the final linear regression model on top of it to test the learned representaiton and proceed with the test phase
    if config['lambda_supervised'] == 0.:
        set_trainable_parameters(model, 'last_linear', config)
        
        # Same as before, but now we need to only the last linear layer
        # Optimizater and scheduler
        if config['lr_scheduler_enable'] and config['lr_scheduler_type'] == 'CosineAnnealingWarmupScheduler':
            config['steps_per_epoch'] = len(train_dataloader)
            
        optim_sched = configure_optimizer_and_scheduler(model, config)
        
        optimizer = optim_sched['optimizer']
        if config['lr_scheduler_enable']:
            scheduler = optim_sched['lr_scheduler']
        else:
            scheduler = None
            
        best_val = float("+inf")
        
        # Early stopping
        if config['es_enable']:
            early_stopping = EarlyStopping(
                patience=config['es_patience'], 
                delta=config['es_min_delta'], 
                verbose=True,
                mode='min'
                )
        
        # Linear probing loop
        for epoch in range(config['max_training_epochs']):
            
            # Training loop
            set_trainable_parameters(model, 'last_linear', config)
            train_losses = AverageMeter(name='train/linear_probing_loss')
            for batch_idx, batch in enumerate(train_dataloader):
                signals, targets = batch

                # Move data and labels to the same device as the model
                signals = signals.to(device)
                targets = torch.cat((batch[1][0].to(device), batch[1][1].to(device)), dim=-1)
                            
                # Perform forward pass
                if config['return_embedding']:
                    outputs, _ = model(signals)
                else:
                    outputs = model(signals)
                    
                # Loss 
                if config['criterion'] == 'MSELoss':
                    loss = F.mse_loss(outputs, targets)
                elif config['criterion'] == 'SmoothL1Loss':
                    loss = F.smooth_l1_loss(outputs, targets)
                else:
                    raise ValueError("Invalid criterion ...")
                
                # Backpropagation and optimization
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                
                # N.B. schedulers are usually called at the end of the epoch, but here we apply it at the end of each batch
                if config['lr_scheduler_enable']:
                    scheduler.step()
                
                train_losses.update(loss.item(), signals.size(0))
                
                # Log training loss to TensorBoard
                writer.add_scalar('train/linear_probing_loss', loss.item(), epoch * len(train_dataloader) + batch_idx)
                
            # Log epoch training loss to TensorBoard
            writer.add_scalar('train/linear_probing_loss_epoch', train_losses.avg, epoch)
            
            # Validation loop    
            set_trainable_parameters(model, 'none', config)
            metrics = ['loss', 'sbp_mae', 'dbp_mae', 'sbp_me', 'dbp_me', 'sbp_mae_std',  'dbp_mae_std', 'sbp_me_std', 'dbp_me_std']
            val_metrics = setup_meter('linear_probing_val', metrics)
            for batch_idx, batch in enumerate(val_dataloader):
                signals, targets = batch

                # Move data and labels to the same device as the model
                signals = signals.to(device)
                targets = torch.cat((batch[1][0].to(device), batch[1][1].to(device)), dim=-1)
                
                # Perform forward pass
                if config['return_embedding']:
                    outputs, _ = model(signals)
                else:
                    outputs = model(signals)
                
                # Loss 
                if config['criterion'] == 'MSELoss':
                    val_loss = F.mse_loss(outputs, targets)
                elif config['criterion'] == 'SmoothL1Loss':
                    val_loss = F.smooth_l1_loss(outputs, targets)
                else:
                    raise ValueError("Invalid criterion ...")
                
                metric_values = get_metric_values(val_loss, outputs, targets)
                update_meter(val_metrics, metric_values, signals.size(0))
            
            # Log epoch validation metrics to TensorBoard
            log_meter_to_tensorboard(writer, val_metrics, epoch, name='linear_probing_val')
            
            # Log LR
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar(f'{config["optimizer_type"]}', current_lr, epoch)
            
            # Print losses
            print(f"Epoch {epoch + 1}/{config['max_training_epochs']} - Train Loss {train_losses.avg}, Val Loss {val_metrics['loss'].avg}")
            
            # Update best model 
            if val_metrics['loss'].avg < best_val:
                save_status(None, epoch, model_name, save_name, model, optimizer, scheduler, val_metrics, checkpoint_path, config)
                best_val = val_metrics['loss'].avg
            
            #  Early stopping
            if config['es_enable']:
                early_stopping(val_metrics['loss'].avg)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
    
    # Set the model to eval mode for final testing
    set_trainable_parameters(model, 'none', config)
    
    # Test loop
    metrics = ['loss', 'sbp_mae', 'dbp_mae', 'sbp_me', 'dbp_me', 'sbp_mae_std',  'dbp_mae_std', 'sbp_me_std', 'dbp_me_std']
    test_metrics = setup_meter('test', metrics)
    
    all_test_outputs = np.empty((0, 2), dtype=float)
    all_test_targets = np.empty((0, 2), dtype=float)
    for batch_idx, batch in enumerate(test_dataloader):
        signals, targets = batch
        
        # Move data and labels to the same device as the model
        signals = signals.to(device)
        targets = torch.cat((batch[1][0].to(device), batch[1][1].to(device)), dim=-1)
        
        # Perform forward pass
        if config['return_embedding']:
            outputs, _ = model(signals)
        else:
            outputs = model(signals)
        
        # Loss 
        if config['criterion'] == 'MSELoss':
            test_loss = F.mse_loss(outputs, targets)
        elif config['criterion'] == 'SmoothL1Loss':
            test_loss = F.smooth_l1_loss(outputs, targets)
        else:
            raise ValueError("Invalid criterion ...")
    
        all_test_outputs = np.concatenate((all_test_outputs, outputs.detach().cpu().numpy()), axis=0)
        all_test_targets = np.concatenate((all_test_targets, targets.detach().cpu().numpy()), axis=0)
        
        metric_values = get_metric_values(test_loss, outputs, targets)
        update_meter(test_metrics, metric_values, signals.size(0))
    
    # Log personalization test metrics to TensorBoard
    log_meter_to_tensorboard(writer, test_metrics, epoch, name='test')
    
    writer.close()
    
    return all_test_targets, all_test_outputs
    

def personalization_training_validation_testing(save_name, checkpoint_path, tensorboard_path, model_name, subject_id, dataloaders, config, device, save_models=False):
    
    # Load pretrained model
    pretrained_model = get_model_architecture(config)
    checkpoint = torch.load(config['pretrained_model_checkpoint'], weights_only=False)
    print(f"Checkpoint loaded from {config['pretrained_model_checkpoint']}")
    pretrained_model.load_state_dict(checkpoint['model'])
    pretrained_model.eval() # to avoind incosistent results as explicitly reported at https://pytorch.org/tutorials/beginner/saving_loading_models.html
    set_trainable_parameters(pretrained_model, 'none', config)
    
    # Copy pretrained model for fine-tuning to avoid touching the original model
    personalized_model = copy.deepcopy(pretrained_model)
    personalized_model = personalized_model.to(device)
    
    # Deallocate the pretrained model to free memory
    del pretrained_model
    gc.collect()
    
    # Set the model to training model
    set_trainable_parameters(personalized_model, config['set_tunable_params'], config)
    
    # Print trainable and non-trainable parameters
    trainable_params = sum(p.numel() for p in personalized_model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in personalized_model.parameters() if not p.requires_grad)

    print(f"Trainable parameters: {trainable_params} / {trainable_params + non_trainable_params} ({trainable_params / (trainable_params + non_trainable_params) * 100:.2f}%)")
    print(f"Non-trainable parameters: {non_trainable_params} / {trainable_params + non_trainable_params} ({non_trainable_params / (trainable_params + non_trainable_params) * 100:.2f}%)")
    
    # Dataloaders
    train_dataloader = dataloaders['train']
    val_dataloader = dataloaders['val']
    test_dataloader = dataloaders['test']
    
    # Optimizater and scheduler
    if config['lr_scheduler_enable'] and config['lr_scheduler_type'] == 'CosineAnnealingWarmupScheduler':
        config['steps_per_epoch'] = len(train_dataloader)
        
    optim_sched = configure_optimizer_and_scheduler(personalized_model, config)
    
    optimizer = optim_sched['optimizer']
    if config['lr_scheduler_enable']:
        scheduler = optim_sched['lr_scheduler']
    else:
        scheduler = None
    
    # Logging to TensorBoard Summary Writer
    writer = SummaryWriter(log_dir=os.path.join(tensorboard_path, str(subject_id)))

    # Monitor lowest loss
    best_val = float("+inf")    
    if config['es_enable']:
        early_stopping = EarlyStopping(patience=config['es_patience'], delta=config['es_min_delta'], verbose=True)
    
    # Personalization loop
    for epoch in range(config['max_training_epochs']):
        
        # Training loop
        set_trainable_parameters(personalized_model, config['set_tunable_params'], config)
        train_losses = AverageMeter(name='train/loss')
        for batch_idx, batch in enumerate(train_dataloader):
            signals, targets = batch

            # Move data and labels to the same device as the model
            signals = signals.to(device)
            targets = torch.cat((batch[1][0].to(device), batch[1][1].to(device)), dim=-1)
                        
            # Perform forward pass
            if config['return_embedding']:
                outputs, _ = personalized_model(signals)
            else:
                outputs = personalized_model(signals)
            
            # Loss 
            if config['criterion'] == 'MSELoss':
                loss = F.mse_loss(outputs, targets)
            elif config['criterion'] == 'SmoothL1Loss':
                loss = F.smooth_l1_loss(outputs, targets)
            else:
                raise ValueError("Invalid criterion ...")
            
            loss = config['lambda_supervised'] * loss
            
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # N.B. schedulers are usually called at the end of the epoch, but here we apply it at the end of each batch
            if config['lr_scheduler_enable']:
                scheduler.step()
            
            train_losses.update(loss.item(), signals.size(0))
            
            # Log training loss to TensorBoard
            writer.add_scalar('train/loss', loss.item(), epoch * len(train_dataloader) + batch_idx)
            
        # Log epoch training loss to TensorBoard
        writer.add_scalar('train/loss_epoch', train_losses.avg, epoch)
        
        # Validation loop    
        set_trainable_parameters(personalized_model, 'none', config)
        metrics = ['loss', 'sbp_mae', 'dbp_mae', 'sbp_me', 'dbp_me', 'sbp_mae_std',  'dbp_mae_std', 'sbp_me_std', 'dbp_me_std']
        val_metrics = setup_meter('val', metrics)
        for batch_idx, batch in enumerate(val_dataloader):
            signals, targets = batch

            # Move data and labels to the same device as the model
            signals = signals.to(device)
            targets = torch.cat((batch[1][0].to(device), batch[1][1].to(device)), dim=-1)
            
            # Perform forward pass
            if config['return_embedding']:
                outputs, _ = personalized_model(signals)
            else:
                outputs = personalized_model(signals)
            
            # Loss 
            if config['criterion'] == 'MSELoss':
                val_loss = F.mse_loss(outputs, targets)
            elif config['criterion'] == 'SmoothL1Loss':
                val_loss = F.smooth_l1_loss(outputs, targets)
            else:
                raise ValueError("Invalid criterion ...")
            
            metric_values = get_metric_values(val_loss, outputs, targets)
            update_meter(val_metrics, metric_values, signals.size(0))
        
        # Log epoch validation metrics to TensorBoard
        log_meter_to_tensorboard(writer, val_metrics, epoch, name='val')
        
        # Log LR
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar(f'{config["optimizer_type"]}', current_lr, epoch)
        
        # Print losses
        print(f"Epoch {epoch + 1}/{config['max_training_epochs']} - Train Loss {train_losses.avg}, Val Loss {val_metrics['loss'].avg}")
        
        # Update best model 
        if val_metrics['loss'].avg < best_val:
            save_status(subject_id, epoch, model_name, save_name, personalized_model, optimizer, scheduler, val_metrics, checkpoint_path, config)
            best_val = val_metrics['loss'].avg
        
        #  Early stopping
        if config['es_enable']:
            early_stopping(val_metrics['loss'].avg)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            
    # Load the best model
    personalized_model = get_model_architecture(config)
    load_status(subject_id, model_name, save_name, personalized_model, optimizer, scheduler, checkpoint_path, config)
    personalized_model = personalized_model.to(device)
    set_trainable_parameters(personalized_model, 'none', config)
    
    # Test loop
    metrics = ['loss', 'sbp_mae', 'dbp_mae', 'sbp_me', 'dbp_me', 'sbp_mae_std',  'dbp_mae_std', 'sbp_me_std', 'dbp_me_std']
    test_metrics = setup_meter('test', metrics)

    all_test_outputs = np.empty((0, 2), dtype=float)
    all_test_targets = np.empty((0, 2), dtype=float)
    for batch_idx, batch in enumerate(test_dataloader):
        signals, targets = batch
        
        # Move data and labels to the same device as the model
        signals = signals.to(device)
        targets = torch.cat((batch[1][0].to(device), batch[1][1].to(device)), dim=-1)
        
        # Perform forward pass
        if config['return_embedding']:
            outputs, _ = personalized_model(signals)
        else:
            outputs = personalized_model(signals)
        
        # Loss 
        if config['criterion'] == 'MSELoss':
            test_loss = F.mse_loss(outputs, targets)
        elif config['criterion'] == 'SmoothL1Loss':
            test_loss = F.smooth_l1_loss(outputs, targets)
        else:
            raise ValueError("Invalid criterion ...")
    
        all_test_outputs = np.concatenate((all_test_outputs, outputs.detach().cpu().numpy()), axis=0)
        all_test_targets = np.concatenate((all_test_targets, targets.detach().cpu().numpy()), axis=0)
        
        metric_values = get_metric_values(test_loss, outputs, targets)
        update_meter(test_metrics, metric_values, signals.size(0))
    
    # Log personalization test metrics to TensorBoard
    log_meter_to_tensorboard(writer, test_metrics, epoch, name='test')
    
    writer.close()
    
    # Remove the model from disk
    if not save_models:
        shutil.rmtree(os.path.join(checkpoint_path, save_name, str(subject_id)))
        shutil.rmtree(os.path.join(tensorboard_path, str(subject_id)))
    
    return all_test_targets, all_test_outputs
    