import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Iterable, Optional
import re

# Third-party imports
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
# import torch.distributed as dist

from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut

from timm.data import Mixup
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler, get_state_dict, ModelEma, accuracy

# Local application/library specific imports
import LOO_utils as utils
import vision_transformer as vits
from LOO_dataloader_patientwise import LOO_ECOG_Dataset, DataAugmentation_finetune


import pdb 


def get_args_parser():
    
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)

    parser.add_argument('--batch_size', default=48, type=int)
    
    parser.add_argument('--in_chans', default=3, type=int)
    parser.add_argument('--negative_class_mode', default=0, type=int)

    # Model parameters
    parser.add_argument('--model', default='vit_small', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input-size', default=224,
                        type=int, help='images input size')
    parser.add_argument('--ensemble', default=10, type=int,
                        help='aggregate over the last n epochs')

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    parser.add_argument('--model-ema', action='store_true')
    parser.add_argument(
        '--no-model-ema', action='store_false', dest='model_ema')
    parser.set_defaults(model_ema=True)
    parser.add_argument('--model-ema-decay', type=float,
                        default=0.99996, help='')
    parser.add_argument('--model-ema-force-cpu',
                        action='store_true', default=False, help='')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Finetuning params
    parser.add_argument('--finetune', 
        default=r'/media/enver/easystore/Amal/DeliriumProject/AmalScripts/PatientwiseMainScript/best_ckpt_ep0310.pth', 
        help='finetune from checkpoint')

    # Dataset parameters
    parser.add_argument('--data_location', 
                        default=r'/media/enver/easystore/Amal/DeliriumProject/ModelRun/Input/Round2/60min/ByPatient/LOO_CV',
                        type=str, help='semantic granularity')

    parser.add_argument('--output_dir', 
                        default=r'/media/enver/easystore/Amal/DeliriumProject/ModelRun/Output/ByPatientOutput2',
                        help='path where to save, empty for no saving')
    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dist-eval', action='store_true',
                        default=False, help='Enabling distributed evaluation')
    parser.add_argument('--num_workers', default=10, type=int)

    parser.add_argument('--evaluate_every', default=1, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)

    # # distributed training parameters
    # parser.add_argument('--world_size', default=1, type=int,
    #                     help='number of distributed processes')
    # parser.add_argument('--dist_url', default='env://',
    #                     help='url used to set up distributed training')
    # parser.add_argument("--local_rank", default=0, type=int,
    #                     help="Please ignore and do not set this argument.")
    
    parser.add_argument('--no-plots', action='store_true', help='Disable saving all plots (ROC, PR curves, etc.)')
    
    
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    
    parser.add_argument('--start_epoch_inner', default=0, type=int, metavar='N',
                        help='start inner epoch')
    parser.add_argument('--epochs', default=4, type=int)
    parser.add_argument('--epochs_inner', default= 4, type=int, 
                    help='Number of epochs for inner loop training.')
    
    parser.add_argument('--fold', type=int, default=None)
    parser.add_argument('--local_rank', default=0, type=int)
   
    parser.add_argument('--folds', type=int, nargs='+', default=None,
                    help='A list of specific fold indices to run.')

    # ADD a new argument for the GPU ID
    parser.add_argument('--gpu-id', type=int, default=0,
                    help='The specific ID of the GPU to use.')
    
    parser.add_argument('--patient_classification_threshold', type=float, default=0.5,
                    help='Ratio of positive images required to classify a patient as positive (e.g., 0.5 for >50%).')
    
    parser.add_argument('--dry-run', action='store_true', 
                        help='Enable dry run for quick testing (overrides epoch counts).')
    return parser

def main_train(args):
    
    device = torch.device(f'cuda:{args.gpu_id}')
    print(f" Worker process started. Running on device: {device}")

    utils.fix_random_seeds(args.seed)
    # device = torch.device(args.device)
    seed = args.seed # + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    cudnn.benchmark = True
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    logs_dir = Path(args.output_dir) / "logs"
    preds_dir = Path(args.output_dir) / "predictions"
    
    logs_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    
    
    args.nb_classes = 2
    
    mixup_fn2 = None 
    # num_tasks = utils.get_world_size()
    # global_rank = utils.get_rank()    

    # linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    linear_scaled_lr = args.lr * args.batch_size / 512.0
    args.lr = linear_scaled_lr


    master_df = pd.read_csv(os.path.join(args.data_location, 'alldata.csv'), header=None, names=['filename', 'label', 'patient_id'])
    X, Y, groups = master_df.values, master_df['label'].values, master_df['patient_id'].values
    outer_cv  = LeaveOneGroupOut() # LeaveOnePairOut(labels=master_df['label'].values, patient_ids=master_df['patient_id'].values, random_state=args.seed)
    
    print("\n--- STARTING PHASE 1: Nested Cross-Validation for Evaluation ---")
    
    all_folds = list(outer_cv.split(X, Y, groups))
    folds_to_run_by_this_worker = args.folds
    
    if folds_to_run_by_this_worker is None:
    # If --folds isn't specified, default to running all folds
        folds_to_run_by_this_worker = range(len(all_folds))

    try:
        tw_name = extract_time_window(args.output_dir)
        print(f"Time window extracted: {tw_name}")
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # =========================================================================
    # OUTER LOOP: Iterates through each patient to use as the final test set.
    # =========================================================================
    # for outer_fold_idx, (train_outer_indices, test_outer_indices) in enumerate(outer_cv.split(X, y, groups)):
    for outer_fold_idx in folds_to_run_by_this_worker:      


        # if args.fold is not None and outer_fold_idx != args.fold:
        #     continue

        train_outer_indices, test_outer_indices = all_folds[outer_fold_idx]

        train_outer_df = master_df.iloc[train_outer_indices]
        test_outer_df = master_df.iloc[test_outer_indices]
        held_out_patient_id = test_outer_df['patient_id'].unique().tolist() # test_outer_df['patient_id'].unique()[0]
        held_out_patient_str = ", ".join(map(str, held_out_patient_id))
        
        print(f"\n===== OUTER FOLD {outer_fold_idx}: Holding out patients {held_out_patient_id} =====")
        
        inner_loop_preds, inner_loop_labels = [], []
        
        inner_X = train_outer_df.values
        inner_y = train_outer_df['label'].values                           

        inner_cv = LeaveOneGroupOut()
        inner_groups = train_outer_df['patient_id'].values

        # This loop will now run 9 times on the training data for the outer fold
        for inner_fold_idx, (train_inner_indices, val_inner_indices) in enumerate(inner_cv.split(inner_X, inner_y, groups=inner_groups)):             

            # Split data for this inner fold (e.g., 8 patients for train, 1 for val)
            train_inner_df = train_outer_df.iloc[train_inner_indices]
            val_inner_df = train_outer_df.iloc[val_inner_indices]
            
            # Get all unique patient IDs present in the inner validation set
            held_out_inner_patients = val_inner_df['patient_id'].unique().tolist()
            
            # Convert the list to a string for printing
            held_out_inner_patients_str = ", ".join(map(str, held_out_inner_patients))
            
            print(f"    - Inner Fold {inner_fold_idx+1}: Training on {train_inner_df['patient_id'].nunique()} patients, validating on patients: {held_out_inner_patients_str}")
            
            # Prepare DataLoaders for this specific inner fold
            print("  --- Preparing DataLoaders for inner fold...")
            loader_train_inner = create_dataloader(args, args.data_location, train_inner_df, is_train=True)
            loader_val_inner   = create_dataloader(args, args.data_location, val_inner_df, is_train=False)

            print(f"Creating fresh, temporary model: {args.model}")
            
            model_inner, _, criterion_inner, optimizer_inner, sched_inner, scaler_inner, _ = setup_model_for_fold(
                                        args, device, train_inner_df, is_hero_model=False)
            
            for epoch_inner in range(args.start_epoch_inner, args.epochs_inner+1):
                train_one_epoch(
                    model=model_inner, 
                    criterion=criterion_inner, 
                    data_loader=loader_train_inner, 
                    optimizer=optimizer_inner, 
                    device=device, 
                    epoch=epoch_inner, 
                    loss_scaler=scaler_inner, 
                    max_norm=args.clip_grad, 
                    set_training_mode=(args.finetune == ''),
                    args=args)
                               
                sched_inner.step(epoch_inner)

            _, preds_list, labels_list = evaluate(loader_val_inner, model_inner, device)
            # labels_list, preds_list = torch.cat(labels_list).numpy(), torch.cat(preds_list).numpy()
            
            inner_loop_preds.extend(preds_list)
            inner_loop_labels.extend(labels_list)
            
            if args.dry_run:
                if inner_fold_idx >= 1:
                    print("Stopping inner loop early after 2 folds (for quick testing)")
                    break

        # --- Find the optimal threshold for this OUTER fold ---
        all_inner_preds = torch.cat(inner_loop_preds)
        all_inner_labels = torch.cat(inner_loop_labels)
        optimal_threshold, top_thresholds = validate_and_find_optimal_threshold(all_inner_preds, all_inner_labels)
        # optimal_threshold = 0.41
        
        print(f"--- Optimal threshold for this outer fold found: {optimal_threshold:.4f} ---")
        
        # =========================================================================
        # FINAL TRAINING & PER-EPOCH EVALUATION LOOP
        # =========================================================================
        print("\n--- PHASE 2: Training final model and evaluating at each epoch ---")
        loader_train_outer = create_dataloader(args, args.data_location, train_outer_df, is_train=True)
        
        electrode_groups = {
            "BalanceTestSet": ['C3', 'O1', 'T5', 'T4', 'T3', 'Pz', 'P4', 'P3', 'O2', 'Fz'],
            "C1_HighPerformers": ['T4', 'F8', 'Fp2', 'T3', 'O2', 'T5', 'T6', 'O1', 'F3', 'F7'],
            "C1_PoorPerformers": ['Cz', 'Pz', 'P4', 'Fz', 'P3', 'C3', 'C4', 'F4', 'Fp1', 'F7'],
            "C0_High_C1_Low": ['Cz', 'P4', 'Pz', 'C3', 'P3'],  # High for Class 0 but Low for Class 1
            "C1_High_C0_Low": ['Fp2', 'T4', 'F8', 'T3', 'Fp1'],  # High for Class 1 but Low for Class                   
            "all": list(test_outer_df['filename'].apply(lambda x: x.split('_')[3]).unique())
        }

        test_loaders_by_electrode = create_test_loaders_by_electrode_groups(
            args=args,
            test_df=test_outer_df,
            data_dir=args.data_location,
            electrode_groups=electrode_groups,
            section_idx=3  # Assuming electrode is in 4th field in filename
        )

        
        model_outer, model_without_ddp, crit_outer, opt_outer, sched_outer, scaler_outer, model_ema_outer = setup_model_for_fold(
            args, device, train_outer_df, is_hero_model=True
        )
        
        n_parameters = sum(p.numel() for p in model_outer.parameters() if p.requires_grad)
        
        

        for epoch in range(args.start_epoch, args.epochs+1):
                        
            # 1. Train for one epoch (This part of your code is correct and stays)
            train_stats = train_one_epoch(
                model=model_outer, 
                criterion=crit_outer, 
                data_loader=loader_train_outer, 
                optimizer=opt_outer, 
                device=device, 
                epoch=epoch, 
                loss_scaler=scaler_outer, 
                max_norm=args.clip_grad, 
                model_ema=model_ema_outer,
                set_training_mode=(args.finetune == ''),
                args=args
            )

            
            sched_outer.step(epoch)
            
            # 2. Save the checkpoint for this epoch
            if args.output_dir:
                
                fold_dir = Path(args.output_dir) / f"fold_{outer_fold_idx}_{args.patient_classification_threshold}_patient_{held_out_patient_str}"
                checkpoint_path = fold_dir / f'checkpoint_epoch_{epoch:02d}.pth'
                fold_dir.mkdir(parents=True, exist_ok=True)

                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': opt_outer.state_dict(),
                    'lr_scheduler': sched_outer.state_dict(),
                    'epoch': epoch,
                    'model_ema': get_state_dict(model_ema_outer),
                    'scaler': scaler_outer.state_dict(),
                    'args': args,
                    }, checkpoint_path)   
            
            
            
            epoch_log = {
                    "epoch": epoch,
                    "image_n_parameters": n_parameters,
                    "train_stats": convert_to_json_serializable(train_stats),
                    "test_results_by_group": {}}

            # 2. Evaluate the model for imagelevel classification
            if epoch % args.evaluate_every == 0:
                
                # Create empty lists to hold the results from ALL electrode groups
                all_patient_results_this_epoch = []
                # This loop will run 3 times (for 'top10', 'top5', 'all')
                for group_name, loader_test_outer in test_loaders_by_electrode.items():
                    print(f"\n>>> Evaluating on electrode group: {group_name}")
                    
                    individual_test_stats, individual_preds_raw, individual_targets_raw = evaluate(loader_test_outer, model_outer, device)
                    individual_targets = torch.cat(individual_targets_raw).numpy()
                    individual_preds = torch.cat(individual_preds_raw).numpy()
                    
                    patient_predictions, epoch_stats, image_probs = analyze_epoch_results(
                                                        image_targets=individual_targets,
                                                        final_pred=individual_preds,
                                                        threshold=optimal_threshold,
                                                        top_10_thresholds=top_thresholds,
                                                        num_parameter=n_parameters,
                                                        dataset=loader_test_outer,
                                                        epoch=epoch,
                                                        train_stats=train_stats,
                                                        test_stats=individual_test_stats,
                                                        args=args
                                                    )
                    all_patient_results_this_epoch.extend(patient_predictions)
                    
                    # Your CSV logging for the current group is correct
                    filenames = loader_test_outer.dataset.dataframe['filename'].tolist()
                    patient_ids = loader_test_outer.dataset.dataframe['patient_id'].tolist()
                    true_labels = individual_targets.tolist()
                    
                    image_level_data = []
                    for fn, pid, y_true, prob in zip(filenames, patient_ids, true_labels, image_probs):
                        pred_label = int(prob >= optimal_threshold)
                        image_level_data.append({
                            'filename': fn,
                            'true_label': y_true,
                            'pred_label': pred_label,
                            'probability': prob,
                            'patient_id': pid
                        })
                    
                    csv_path = preds_dir / f"patient_{held_out_patient_str}__fold_{outer_fold_idx}_{args.patient_classification_threshold}_{group_name}_image_predictions.csv"
                    
                    log_predictions_per_epoch(
                                                csv_path=csv_path,
                                                predictions=image_level_data,
                                                epoch=epoch,
                                                fold=outer_fold_idx,
                                                args=args,
                                                threshold=optimal_threshold,
                                                time_window=tw_name,
                                                held_out_patient_id = held_out_patient_id 
                                            )
                    
                    epoch_log["test_results_by_group"][group_name] = {
                                                            "test_stats": convert_to_json_serializable(epoch_stats),
                                                            "patient_predictions": convert_to_json_serializable(patient_predictions)}

                # STEP 2: Print the results from the master list that contains ALL groups
                print(f"  Results for Epoch {epoch}:")
                for patient_data in all_patient_results_this_epoch:
                    print(f"  -> Patient '{patient_data['patient_id']}': "
                        f"True={patient_data['true_label']}, "
                        f"Pred={patient_data['pred_label']} "
                        f"({patient_data['positive_images']}/{patient_data['total_images']} pos. images)")

                # STEP 3: Write all the collected JSON logs to the file at once
                if args.output_dir:
                    with (logs_dir / f"log_test_{held_out_patient_str}.txt").open("a") as f:
                        f.write(json.dumps(epoch_log) + "\n")


    # completion_file = Path(args.output_dir) / f"{tw_name}_gpu{args.gpu_id}_done.txt"
    completion_file = Path(args.output_dir) / f"{tw_name}_gpu{args.gpu_id}_done_threshold_{args.patient_classification_threshold:.2f}.txt"

    with open(completion_file, "w") as f:
        f.write(f"Completed processing for {tw_name} on GPU {args.gpu_id} for threshold of {args.patient_classification_threshold:.2f} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Completion signal created at: {completion_file}")

def create_dataloader(args, data_location, dataframe, is_train):
    """
    Creates a PyTorch DataLoader for a given dataframe and mode.

    Args:
        args: The script's command-line arguments.
        data_location: The base path to the image directory.
        dataframe: The pandas DataFrame for the current data split.
        is_train (bool): If True, enables data augmentation and shuffling for training.
                         If False, disables them for validation/testing.

    Returns:
        A configured torch.utils.data.DataLoader object.
    """
    # 1. Set parameters based on whether this is for training or not
    if is_train:
        shuffle = True
        drop_last = True
        # Use the DataAugmentation class with augmentation enabled
        transform = DataAugmentation_finetune(istrain=True, input_size=args.input_size, in_chans=args.in_chans)
    else:
        shuffle = False
        drop_last = False
        # Use the DataAugmentation class with augmentation disabled
        transform = DataAugmentation_finetune(istrain=False, input_size=args.input_size, in_chans=args.in_chans)

    # 2. Create the Dataset instance
    dataset = LOO_ECOG_Dataset(
        data_location=data_location,
        dataframe=dataframe,
        transform=transform
    )

    # 3. Create the DataLoader instance
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers= args.num_workers,
        pin_memory=args.pin_mem,
        shuffle=shuffle,
        drop_last=drop_last
    )
    
    return data_loader

def create_test_loaders_by_electrode_groups(args, test_df, data_dir, electrode_groups, section_idx=3):
    """
    Creates test DataLoaders for various electrode subsets.

    Args:
        args: Argparse args used for DataLoader creation.
        test_df (pd.DataFrame): Test DataFrame with filenames.
        data_dir (str): Path to the image directory.
        electrode_groups (dict): Dictionary of {group_name: [electrodes]}.
        section_idx (int): Index in filename split by '_' where electrode name appears.

    Returns:
        dict: Dictionary of {group_name: DataLoader}.
    """
    loaders = {}
    for name, electrodes in electrode_groups.items():
        filtered_df = test_df[test_df['filename'].apply(lambda x: x.split('_')[section_idx] in electrodes)]
        loader = create_dataloader(
            args=args,
            data_location=data_dir,
            dataframe=filtered_df,
            is_train=False
        )
        loaders[name] = loader
    return loaders

def create_weighted_loss(dataframe, device):
    
    # ---  Weighted Loss to handle Class Imbalance ---
    try:
         # Count the number of samples for each label (0 and 1)
        class_counts = dataframe['label'].value_counts().sort_index()
                
        # Calculate the weight for each class
        total_samples = class_counts.sum()
        num_classes = len(class_counts)
        class_weights = total_samples / (num_classes * class_counts)
                
        # Convert weights to a PyTorch tensor and send to the active device (e.g., GPU)
        class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float).to(device)
                
        # Create the loss function with the calculated weights
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)
                
        print("Using weighted loss to address class imbalance.")
        print(f"Class weights: {class_weights.to_dict()}")

    except Exception as e:
        print(f"Warning: Could not calculate class weights. Using standard loss. Error: {e}")
        criterion = torch.nn.CrossEntropyLoss()
    
    return criterion

def pos_embed(model, args):

    """
    Loads checkpoint, cleans state_dict, resizes pos_embed, and loads into the model.
    """
    
    if args.finetune and Path(args.finetune).is_file():

        print(f"Loading finetune weights from {args.finetune}...")
        
        checkpoint = torch.load(args.finetune, map_location='cpu')

        # --- 1. Get and Clean the State Dictionary ---
        state_dic = checkpoint.get('SiT_model') or checkpoint.get('model')
        state_dic = {k.replace("module.", ""): v for k, v in state_dic.items()}
        
        state_dic = {k.replace("backbone.", ""): v for k, v in state_dic.items()}

        
        # --- 2. Interpolate Position Embedding ---
        pos_embed_checkpoint = state_dic['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        new_size = int(num_patches ** 0.5)
        
        
        print(f"Resizing position embedding from {orig_size}x{orig_size} to {new_size}x{new_size}")
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = F.interpolate(pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        state_dic['pos_embed'] = new_pos_embed

        # --- 3. Load the weights into the model ---
        msg = model.load_state_dict(state_dic, strict=False)
        print(msg)

def setup_model_for_fold(args, device, df, is_hero_model: bool):
    """
    Creates, prepares, and returns all components needed for a training fold.

    This now includes:
    1. Creating the ViT model.
    2. Loading pre-trained weights and handling positional embeddings.
    3. Handling GPU distribution (DDP).
    4. Creating the weighted loss criterion based on the provided training data.
    5. Creating the optimizer, learning rate scheduler, and loss scaler.
    """
    print("--- Setting up new model, training tools, and distribution ---")

    # Step 1: Create the base model
    model = vits.__dict__[args.model](
        num_classes=args.nb_classes,
        img_size=[args.input_size],
        drop_path_rate=args.drop_path, 
        in_chans=args.in_chans
    )
    model.to(device)
    
    # Step 2: Load pre-trained weights into the base model
    pos_embed(model, args)

    # Step 3: Handle GPU Distribution (DDP).
    # model_without_ddp = model
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     model_without_ddp = model.module
    #     print("Model wrapped in DistributedDataParallel.")
    
    model_without_ddp = model
    
    # Step 4: Conditionally create Model EMA
    model_ema = None
    if is_hero_model and args.model_ema:
        print("--> Enabling Model EMA for this hero model.")
        model_ema = ModelEma(
            model,
            decay=args.model_ema_decay,
            device='cpu' if args.model_ema_force_cpu else '',
            resume=''
        )

    # Step 5: Create the optimizer and other training tools
    optimizer = create_optimizer(args, model_without_ddp) 
    
    
    linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
    args.lr = linear_scaled_lr

    lr_scheduler, _ = create_scheduler(args, optimizer)
    loss_scaler = NativeScaler()
    criterion = create_weighted_loss(df, device)
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'--> Model setup complete. Number of trainable params: {n_parameters / 1e6:.2f}M')
    
    # Return the model that will be used for training
    return model, model_without_ddp, criterion, optimizer, lr_scheduler, loss_scaler, model_ema

def train_one_epoch(model: torch.nn.Module, criterion,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None,
                    mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args=None):
    
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100

    for samples, targets, _ in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(samples, classify=True)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)
        
        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def validate_and_find_optimal_threshold(all_preds_tensor, all_labels_tensor):
    """
    Calculates the threshold by taking the median of the top 10
    thresholds based on Youden's J statistic.
    """
    print("Finding optimal threshold via median of top 10 Youden's J scores...")
    all_probs = torch.nn.functional.softmax(all_preds_tensor.float(), dim=1)[:, 1].cpu().numpy()
    all_labels = all_labels_tensor.cpu().numpy()

    fpr, tpr, thresholds = metrics.roc_curve(all_labels, all_probs)
    j_scores = tpr - fpr

    # --- Your Method: Get indices of the top 10 J-scores ---
    # Use argsort to get the indices that would sort the array in ascending order and take the last 10 for the top 10 scores.
    top_n = 10
    top_j_indices = np.argsort(j_scores)[-top_n:]

    # Get the thresholds corresponding to these top scores
    top_thresholds = thresholds[top_j_indices]

    # Calculate the median of these thresholds
    optimal_threshold = np.median(top_thresholds)
    
    print(f"Top {top_n} thresholds were: {np.round(top_thresholds, 2)}")
    print(f"Median (final) threshold is: {optimal_threshold:.4f}")
    
    return optimal_threshold, top_thresholds

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    targets, preds = [], []
    for images, target, _ in metric_logger.log_every(data_loader, 100, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images, classify=True)
            loss = criterion(output, target)

        preds.append(output.cpu().detach())
        targets.append(target.cpu().detach())

        acc1, acc5 = accuracy(output, target, topk=(1, 2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, preds, targets

@torch.no_grad()
def analyze_epoch_results(image_targets, final_pred, threshold, num_parameter, dataset, epoch, top_10_thresholds, train_stats, test_stats, args):
    """
    Aggregates image predictions to a final patient-level prediction and
    returns detailed results for each patient.
    """
    
    # --- 1. Convert logits to probabilities (Corrected) ---
    prob_tensor = torch.nn.functional.softmax(torch.from_numpy(final_pred).float(), dim=1)
    probs_for_df = prob_tensor[:, 1].cpu().numpy()
       
    patient_ids = list(dataset.dataset.dataframe['patient_id'])

    # --- 2. Create a DataFrame to link predictions to patients ---
    df_preds = pd.DataFrame({
        'patient_id': patient_ids,
        'prob': probs_for_df, # Use the 1D array of probabilities
        'true': image_targets
    })

    epoch_stats = {       
        **{f'train_{k}': v for k, v in train_stats.items()},
        **{f'test_{k}': v for k, v in test_stats.items()},        
        'epoch': int(epoch),
        'image_n_parameters': float(num_parameter)
    }

    # --- 3. Group by patient and get detailed results ---
    patient_results = []
    for pid, group in df_preds.groupby('patient_id'):
        # ... (rest of this section is correct) ...
        image_preds_bin = (group['prob'] >= threshold)
        positive_images_count = int(image_preds_bin.sum())
        total_images_count = len(group)
        pred_label = int((positive_images_count / total_images_count) > args.patient_classification_threshold)
        true_label = int(group['true'].iloc[0])
        patient_results.append({
            'patient_id': pid,
            'true_label': true_label,
            'pred_label': pred_label,
            'positive_images': positive_images_count,
            'total_images': total_images_count,
            'optimal_threshold': threshold, 
            'top_10_thresholds': top_10_thresholds.tolist(),
            'patient_classification_threshold': args.patient_classification_threshold
        })

    # Return the full (N, 2) probability array for reuse in the main loop
    return patient_results, epoch_stats, probs_for_df

def calculate_patient_level_metrics(results):
    """
    Calculates final patient-level metrics from a collected list of all
    patient predictions from all cross-validation folds.
    
    Args:
        results (list of dicts): Each dict must contain 'true_label' and 'pred_label'.

    Returns:
        dict: A dictionary containing the final patient-level performance scores.
    """
    true_labels = [entry['true_label'] for entry in results]
    predicted_labels = [entry['pred_label'] for entry in results]

    print("\n" + "="*50)
    print("FINAL OVERALL PATIENT-LEVEL RESULTS")
    print("="*50)

    # Safely calculate the confusion matrix
    try:
        tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels, labels=[0, 1]).ravel()
    except ValueError:
        print("⚠️ Could not compute full confusion matrix. Reporting available stats.")
        tn, fp, fn, tp = 0, 0, 0, 0

    # Calculate metrics with safety against divide-by-zero
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity / Recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # Specificity
    acc = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)
    balanced_acc = balanced_accuracy_score(true_labels, predicted_labels)

    # Print a nicely formatted summary
    print(f"Accuracy:           {acc:.3f}")
    print(f"Balanced Accuracy:  {balanced_acc:.3f}")
    print(f"Sensitivity (TPR):  {tpr:.3f}")
    print(f"Specificity (TNR):  {tnr:.3f}")
    print(f"F1 Score:           {f1:.3f}")

    # Return results in a dictionary for logging
    log_stats = {
        'Accuracy': acc,
        'Balanced_Accuracy': balanced_acc,
        'TPR': tpr,
        'TNR': tnr,
        'F1': f1,
    }

    return log_stats

def create_completion_signal(output_dir: str, tw_name: str, timestamp: str):
    # Create a file to signal the completion of this time window's processing
    signal_path = Path(output_dir) / f"{timestamp}_{tw_name}_done.txt"
    with open(signal_path, "w") as f:
        f.write(f"Completed processing for {tw_name} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"✅ Completion signal created at: {signal_path}")

def extract_time_window(output_dir):
    """
    Extracts the time window from the output directory path.

    Assumes that the time window is embedded in the directory name (e.g., '60min').
    """
    # Use regex to find a pattern like "60min", "120min", etc. in the path
    match = re.search(r'(\d{1,3}min)', output_dir)
    
    if match:
        return match.group(1)
    else:
        raise ValueError("Time window not found in the output directory path.")

def log_predictions_per_epoch(
    csv_path: Path,
    predictions: list,
    epoch: int,
    fold: int,   
    threshold: float,
    time_window: str,
    held_out_patient_id: list,
    args
):
    """
    Appends per-image predictions for one epoch to a single CSV file.
    Metadata is written only once if the file does not exist.

    Parameters:
    - csv_path: where to save the CSV
    - predictions: list of dicts with keys:
        ['filename', 'true_label', 'pred_label', 'probability', 'patient_id']
    - epoch: current epoch number
    - fold: outer fold index
    - args: main argparse args (for model name, gpu_id, etc.)
    - threshold: optimal threshold used
    - time_window: extracted from output path (e.g., '60min')
    """
    is_new_file = not csv_path.exists()
    
    with open(csv_path, 'a') as f:
        if is_new_file:
            # Write metadata block
            
            f.write(f"# time_window: {time_window}\n")
            f.write(f"# threshold: {threshold:.4f}\n")
            f.write(f"# fold: {fold}\n")

            held_out_str = ", ".join(map(str, held_out_patient_id))
            f.write(f"# held_out_patient_id: {held_out_str}\n")
            
            f.write(f"# gpu_id: {args.gpu_id}\n")
            f.write(f"# model: {args.model}\n")
            f.write(f"# ensemble: {args.ensemble}\n")
            f.write(f"# patient_classification_threshold: {args.patient_classification_threshold}\n\n")
            f.write("epoch,filename,true_label,pred_label,probability\n")
        
        # Append each prediction
        for p in predictions:
            f.write(f"{epoch},{p['filename']},{p['true_label']},{p['pred_label']},{p['probability']:.5f}\n")

def convert_to_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Finetuning on Downstream script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main_train(args)
