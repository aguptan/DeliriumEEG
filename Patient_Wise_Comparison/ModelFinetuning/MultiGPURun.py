import subprocess
from pathlib import Path
import os
import sys
import time
import pandas as pd
import logging
import signal
import torch  # Added for GPU detection

# =============================================================================
# 1. CONFIGURATION FUNCTION
# =============================================================================
def get_config():
    custom_script_dir = "/media/enver/easystore/Amal/DeliriumProject/AmalScripts/PatientwiseMainScript"
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    detected_gpus = torch.cuda.device_count()

    # base_thresholds = [round(x, 2) for x in frange(0.4, 0.60 + 0.001, 0.05)]
    threshold_values = [0.5] # + [x for x in base_thresholds if x != 0.5]

    config = {
        "base_input_dir": Path("/media/enver/easystore/Amal/DeliriumProject/ModelRun/Input/Round2"),
        "base_output_dir": Path("/media/enver/easystore/Amal/DeliriumProject/ModelRun/Output/VaryingTestSets_8.2.25"),
        "finetune_checkpoint_path": Path("/media/enver/easystore/Amal/DeliriumProject/GMML-ECoG-alldata/patient-wise/best_ckpt_ep0310.pth"),
        "python_script_path": Path(f"{custom_script_dir}/LOO_eval_finetune_patientwise2.py"),
        "terminal_command": "gnome-terminal",
        "num_gpus": detected_gpus,
        "run_timestamp": run_timestamp,
        "worker_launch_delay": 1,
        "threshold_values": threshold_values,
        "static_args": {
            "epochs": "75",
            "epochs_inner": "35"
        }
    }
    return config

def frange(start, stop, step):
    while start <= stop:
        yield start
        start += step


# =============================================================================
# 2. LOGGING SETUP
# =============================================================================
def setup_logging(cfg: dict):
    log_dir = cfg['base_output_dir'] / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"mission_control_{cfg['run_timestamp']}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(message)s",
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Logging configured.")
    return logging.getLogger(__name__)

# =============================================================================
# 3. UTILITY FUNCTIONS
# =============================================================================
def get_total_folds(data_dir: Path) -> int:
    logging.info(f"Analyzing {data_dir} to determine total number of folds...")
    try:
        csv_path = data_dir / "alldata.csv"
        df = pd.read_csv(csv_path, header=None, names=['filename', 'label', 'patient_id'])
        n_splits = df['patient_id'].nunique()
        logging.info(f"Found {n_splits} total folds.")
        return n_splits
    except Exception as e:
        logging.error(f"Could not calculate folds. Details: {e}", exc_info=True)
        return 0

def create_fold_assignments(total_folds: int, num_gpus: int) -> dict:
    logging.info("Distributing folds across GPUs...")
    assignments = {i: [] for i in range(num_gpus)}
    for fold_idx in range(total_folds):
        gpu_id = fold_idx % num_gpus
        assignments[gpu_id].append(fold_idx)
    for gpu, folds in assignments.items():
        logging.info(f"  - GPU {gpu} will run {len(folds)} folds: {folds}")
    return assignments

# =============================================================================
# 4. WORKER & SHUTDOWN FUNCTIONS
# =============================================================================
active_workers = []

def launch_worker(cfg: dict, gpu_id: int, folds: list, input_path: Path, output_path: Path, tw_name: str, threshold: float, is_dry_run: bool, static_args: dict) -> subprocess.Popen:
    log_file = output_path / f"{cfg['run_timestamp']}_{tw_name}_gpu_{gpu_id}_threshold_{threshold:.2f}.log"
    logging.info(f"Preparing Worker for GPU {gpu_id}, Threshold {threshold}... Log file: {log_file}")

    dry_run_arg = "--dry-run" if is_dry_run else ""
    args_str = " ".join([f"--{k} {v}" for k, v in static_args.items()])

    # --- Conditionally build the command ---
    if is_dry_run:
        # For a dry run, add a 30-second delay before closing. DO NOT use 'exec'.
        base_command = f"python -u {cfg['python_script_path']}"
        post_command = f"; echo; echo '---'; echo 'DRY RUN FINISHED. This window will close in 30 seconds.'; sleep 30"
    else:
        # For a normal run, use 'exec' to close the terminal automatically on completion.
        base_command = f"exec python -u {cfg['python_script_path']}"
        post_command = ""
    # ------------------------------------

    inner_cmd = (
        f"{base_command} "
        f"--data_location {input_path} --output_dir {output_path} "
        f"--folds {' '.join(map(str, folds))} --gpu-id {gpu_id} "
        f"--patient_classification_threshold {threshold:.2f} "
        f"--finetune {cfg['finetune_checkpoint_path']} "
        f"{args_str} {dry_run_arg} 2>&1 | tee {log_file}{post_command}"
    )

    launch_cmd = [cfg['terminal_command'], '--', 'bash', '-c', inner_cmd]

    try:
        process = subprocess.Popen(launch_cmd)
        active_workers.append(process)
        return process
    except FileNotFoundError:
        logging.error(f"The command '{cfg['terminal_command']}' was not found.")
        return None

def graceful_shutdown(signum, frame):
    logging.warning(f"Shutdown signal ({signum}) received. Terminating all active workers...")
    for worker_process in active_workers:
        if worker_process.poll() is None:
            logging.info(f"  - Terminating worker with PID: {worker_process.pid}")
            worker_process.terminate()
    logging.info("All workers terminated. Exiting.")
    sys.exit(0)

def print_all_summary_files(summary_paths: list):
    logger = logging.getLogger(__name__)
    logger.info("\n Summary of All Completed Runs:\n")

    for tw_name, threshold, path in summary_paths:
        logger.info(f" Time Window: {tw_name} | Threshold: {threshold:.2f}")
        try:
            with open(path, 'r') as f:
                summary = f.read().strip()
                logger.info(summary)
        except Exception as e:
            logger.warning(f"  Could not read summary at {path}: {e}")
        logger.info("-" * 60)


# =============================================================================
# 5. MAIN ORCHESTRATOR
# =============================================================================
def wait_for_all_gpu_signals(tw_name: str, output_dir: Path, num_gpus: int, threshold: float, timeout: int = 7200):
    logger = logging.getLogger(__name__)
    logger.info(f"Waiting for {num_gpus} done files for time window '{tw_name}' and threshold {threshold:.2f}...")
    expected_files = [
        output_dir / f"{tw_name}_gpu{i}_done_threshold_{threshold:.2f}.txt" for i in range(num_gpus)
    ]
    start_time = time.time()

    while True:
        if all(f.exists() for f in expected_files):
            logger.info(f"✅ All completion files for '{tw_name}', threshold={threshold:.2f} detected.")
            return
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Timeout: Not all completion signals received for {tw_name}, threshold={threshold:.2f}")
        time.sleep(10)

def cleanup_workers():
    logger = logging.getLogger(__name__)
    if not active_workers:
        return
    
    logger.warning("Cleaning up active worker processes...")
    for worker_process in active_workers:
        # Check if the process is still running
        if worker_process.poll() is None:
            logger.info(f"  - Terminating worker with PID: {worker_process.pid}")
            worker_process.terminate()
    
    # Clear the global list for the next batch of workers
    active_workers.clear()

def main():
    
    DRY_RUN = '--dry-run' in sys.argv
    if DRY_RUN:
        print("="*40)
        print("DRY MODE")
        print("="*40)
    
    cfg = get_config()
    logger = setup_logging(cfg)

    signal.signal(signal.SIGINT, graceful_shutdown)
    signal.signal(signal.SIGTERM, graceful_shutdown)

    time_windows_to_process = ["120min", "60min", "30min"] # ["120min", "60min", "30min"]
    thresholds_to_process = cfg['threshold_values']
    static_args_for_run = cfg['static_args'].copy()
    # --- ADD THIS BLOCK to limit the work ---
    
    first_tw_path = cfg['base_input_dir'] / time_windows_to_process[0]
    total_folds = get_total_folds(first_tw_path / "ByPatient" / "LOO_CV")
    if total_folds == 0:
        return
    
    
    if DRY_RUN:
            time_windows_to_process = ["120min", "60min"] # Use only the first time window
            thresholds_to_process = [thresholds_to_process[0]]   # Use only the first threshold
            logger.info(f"DRY RUN: Limiting to 1 time window and 1 threshold.")
            logger.info("DRY RUN: Overriding epoch counts for workers.")
            static_args_for_run['epochs'] = '1'
            static_args_for_run['epochs_inner'] = '1'

            # --- New logic to assign up to 2 folds per GPU ---
            logger.info("DRY RUN: Assigning up to 2 folds per GPU...")
            dry_run_assignments = {i: [] for i in range(cfg['num_gpus'])}
            
            for gpu_id in range(cfg['num_gpus']):
                folds_for_this_gpu = []
                
                # Calculate the two potential folds for this GPU
                fold1 = 2 * gpu_id
                fold2 = 2 * gpu_id + 1
                
                # Add the first fold if it exists
                if fold1 < total_folds:
                    folds_for_this_gpu.append(fold1)
                
                # Add the second fold if it exists
                if fold2 < total_folds:
                    folds_for_this_gpu.append(fold2)
                
                # Assign the calculated folds to the GPU
                if folds_for_this_gpu:
                    dry_run_assignments[gpu_id] = folds_for_this_gpu

            fold_assignments = dry_run_assignments
            logger.info(f"DRY RUN: Overriding fold assignments: {fold_assignments}")


    logger.info(f"Script configured to process time windows: {time_windows_to_process}")

    
    
    if not first_tw_path.is_dir():
        logger.error(f"Missing directory: {first_tw_path}")
        return



    fold_assignments = create_fold_assignments(total_folds, cfg['num_gpus'])
    
    if DRY_RUN:
        # Override assignments to give ONE fold to EACH GPU to test all of them
        dry_run_assignments = {}
        for i in range(cfg['num_gpus']):
            # Assign fold 'i' to GPU 'i', if that fold exists
            if i < total_folds:
                dry_run_assignments[i] = [i]
            else:
                dry_run_assignments[i] = [] # This GPU gets no work
        fold_assignments = dry_run_assignments
        logger.info(f"DRY RUN: Overriding fold assignments to test all GPUs: {fold_assignments}")
    
    all_summary_paths = []

# In your main() function

    for threshold in cfg['threshold_values']:
        logger.info(f"\n🔍 Starting sweep for threshold: {threshold:.2f}")

        for tw_name in time_windows_to_process:
            try:
                # --- The logic to launch and wait remains inside 'try' ---
                input_path = cfg['base_input_dir'] / tw_name / "ByPatient" / "LOO_CV"
                output_path = cfg['base_output_dir'] / tw_name
                output_path.mkdir(parents=True, exist_ok=True)

                logger.info("=" * 80)
                logger.info(f" LAUNCHING TIME WINDOW: {tw_name} @ threshold {threshold:.2f}")
                
                # Note: current_workers is no longer needed here
                for gpu_id, folds in fold_assignments.items():
                    if not folds:
                        continue
                    # The launch_worker function adds processes to the global active_workers list
                    launch_worker(cfg, gpu_id, folds, input_path, output_path, tw_name, threshold, DRY_RUN, static_args_for_run)
                    time.sleep(cfg['worker_launch_delay'])
                
                wait_for_all_gpu_signals(tw_name, output_path, cfg['num_gpus'], threshold)
                logger.info(f" Completed time window: {tw_name} @ threshold {threshold:.2f}")

            except TimeoutError as e:
                logger.error(f" FAILED time window: {tw_name} @ threshold {threshold:.2f}. Reason: {e}")
                logger.warning("Proceeding to cleanup before the next time window...")
            
            except Exception as e:
                logger.error(f" An unexpected error occurred for {tw_name} @ threshold {threshold:.2f}: {e}", exc_info=True)
                logger.warning("Proceeding to cleanup before the next time window...")

            finally:
                # --- THIS BLOCK ALWAYS RUNS ---
                # It cleans up all workers from the completed or failed run
                cleanup_workers()

    logger.info(" All time windows and thresholds processed.")


# =============================================================================
# 6. SCRIPT EXECUTION
# =============================================================================
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.critical("FATAL ERROR IN LAUNCHER", exc_info=True)
