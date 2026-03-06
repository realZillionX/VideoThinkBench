import torch, os, argparse, accelerate, warnings, random
import numpy as np
from tqdm import tqdm
from diffsynth.core import UnifiedDataset
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.diffusion import *
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None,
        tokenizer_path=None, audio_processor_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="", lora_rank=32, lora_checkpoint=None,
        preset_lora_path=None, preset_lora_model=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        fp8_models=None,
        offload_models=None,
        device="cpu",
        task="sft",
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
    ):
        super().__init__()
        # Warning
        if not use_gradient_checkpointing:
            warnings.warn("Gradient checkpointing is detected as disabled. To prevent out-of-memory errors, the training framework will forcibly enable gradient checkpointing.")
            use_gradient_checkpointing = True
        
        # Load models
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, fp8_models=fp8_models, offload_models=offload_models, device=device)
        tokenizer_config = ModelConfig(model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/umt5-xxl/") if tokenizer_path is None else ModelConfig(tokenizer_path)
        audio_processor_config = None if audio_processor_path is None else ModelConfig(audio_processor_path)
        self.pipe = WanVideoPipeline.from_pretrained(torch_dtype=torch.bfloat16, device=device, model_configs=model_configs, tokenizer_config=tokenizer_config, audio_processor_config=audio_processor_config)
        self.pipe = self.split_pipeline_units(task, self.pipe, trainable_models, lora_base_model)
        
        # Training mode
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint,
            preset_lora_path, preset_lora_model,
            task=task,
        )
        
        # Store other configs
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.fp8_models = fp8_models
        self.task = task
        self.task_to_loss = {
            "sft:data_process": lambda pipe, *args: args,
            "direct_distill:data_process": lambda pipe, *args: args,
            "sft": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "sft:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: FlowMatchSFTLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
            "direct_distill:train": lambda pipe, inputs_shared, inputs_posi, inputs_nega: DirectDistillLoss(pipe, **inputs_shared, **inputs_posi),
        }
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        
    def parse_extra_inputs(self, data, extra_inputs, inputs_shared):
        for extra_input in extra_inputs:
            if extra_input == "input_image":
                inputs_shared["input_image"] = data["video"][0]
            elif extra_input == "end_image":
                inputs_shared["end_image"] = data["video"][-1]
            elif extra_input == "reference_image" or extra_input == "vace_reference_image":
                inputs_shared[extra_input] = data[extra_input][0]
            else:
                inputs_shared[extra_input] = data[extra_input]
        return inputs_shared
    
    def get_pipeline_inputs(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        inputs_shared = {
            # Assume you are using this pipeline for inference,
            # please fill in the input parameters.
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            # Please do not modify the following parameters
            # unless you clearly know what this will cause.
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        inputs_shared = self.parse_extra_inputs(data, self.extra_inputs, inputs_shared)
        return inputs_shared, inputs_posi, inputs_nega
    
    def forward(self, data, inputs=None):
        if inputs is None: inputs = self.get_pipeline_inputs(data)
        inputs = self.transfer_data_to_device(inputs, self.pipe.device, self.pipe.torch_dtype)
        for unit in self.pipe.units:
            inputs = self.pipe.unit_runner(unit, self.pipe, *inputs)
        loss = self.task_to_loss[self.task](self.pipe, *inputs)
        return loss


def _save_training_state(accelerator, optimizer, scheduler, model_logger, output_path, tag):
    """Save optimizer, scheduler, and step counter alongside LoRA checkpoint.

    Uses accelerator.save_state() when DeepSpeed is active (ZeRO shards the
    optimizer across ranks, so a plain state_dict() on one rank is incomplete).
    Falls back to manual torch.save() for single-GPU / DDP runs.
    """
    accelerator.wait_for_everyone()
    state_dir = os.path.join(output_path, f"training_state_{tag}")
    try:
        # accelerator.save_state persists model, optimizer, scheduler, RNG,
        # and GradScaler in a directory, with all ranks participating.
        accelerator.save_state(output_dir=state_dir)
    except Exception as e:
        if accelerator.is_main_process:
            print(f"[Resume] WARNING: accelerator.save_state failed ({e}), falling back to manual save.")
        if accelerator.is_main_process:
            os.makedirs(state_dir, exist_ok=True)
            torch.save({
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, os.path.join(state_dir, "manual_state.pt"))
    # Always save step counter as a simple file (all strategies need this).
    if accelerator.is_main_process:
        os.makedirs(state_dir, exist_ok=True)
        torch.save({"num_steps": model_logger.num_steps}, os.path.join(state_dir, "step_counter.pt"))
        print(f"[Resume] Saved training state → {state_dir} (step={model_logger.num_steps})")


def _load_training_state(accelerator, optimizer, scheduler, model_logger, state_dir):
    """Restore optimizer, scheduler, and step counter from saved state.

    Returns True only if ALL components were successfully restored.
    Returns False if any critical component failed — caller should treat
    this as a fresh-start scenario.
    """
    if not os.path.isdir(state_dir):
        if accelerator.is_main_process:
            print(f"[Resume] Training state dir not found: {state_dir} — optimizer starts fresh.")
        return False

    success = True

    # Restore step counter first (always available).
    step_file = os.path.join(state_dir, "step_counter.pt")
    if os.path.exists(step_file):
        step_data = torch.load(step_file, map_location="cpu", weights_only=False)
        model_logger.num_steps = step_data.get("num_steps", 0)
        if accelerator.is_main_process:
            print(f"[Resume] Step counter restored: num_steps={model_logger.num_steps}")
    else:
        if accelerator.is_main_process:
            print(f"[Resume] WARNING: step_counter.pt not found in {state_dir}")
        success = False

    # Try accelerator.load_state (handles DeepSpeed ZeRO correctly).
    try:
        accelerator.load_state(input_dir=state_dir)
        if accelerator.is_main_process:
            print(f"[Resume] Optimizer + scheduler restored via accelerator.load_state from {state_dir}")
    except Exception as e:
        if accelerator.is_main_process:
            print(f"[Resume] WARNING: accelerator.load_state failed ({e}), trying manual fallback.")
        # Fallback to manual restore.
        manual_path = os.path.join(state_dir, "manual_state.pt")
        if os.path.exists(manual_path):
            state = torch.load(manual_path, map_location="cpu", weights_only=False)
            try:
                optimizer.load_state_dict(state["optimizer"])
                scheduler.load_state_dict(state["scheduler"])
                if accelerator.is_main_process:
                    print(f"[Resume] Optimizer + scheduler restored via manual fallback.")
            except Exception as e2:
                if accelerator.is_main_process:
                    print(f"[Resume] CRITICAL: Manual optimizer restore also failed: {e2}")
                success = False
        else:
            if accelerator.is_main_process:
                print(f"[Resume] CRITICAL: No manual fallback found, optimizer starts fresh.")
            success = False

    return success


def _seed_dataloader_worker(_worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def launch_training_task_resume(
    accelerator: accelerate.Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
):
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs

    # --- Resume info from args ---
    start_epoch = getattr(args, "start_epoch", 0)
    resume_step = getattr(args, "resume_step", None)

    optimizer = torch.optim.AdamW(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    data_seed = getattr(args, "seed", 42)
    if data_seed is None:
        data_seed = 42

    # Fix: Use a seeded generator so shuffle order is reproducible per epoch.
    # Combined with set_epoch-style seeding in the loop, this makes
    # data ordering deterministic for the same epoch even after restart.
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(data_seed)
    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=True, collate_fn=lambda x: x[0],
        num_workers=num_workers, generator=shuffle_generator,
        worker_init_fn=_seed_dataloader_worker,
    )

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    steps_per_epoch = len(dataloader)
    if steps_per_epoch == 0:
        raise ValueError("[Resume] DataLoader is empty (0 steps per epoch). Check dataset path and metadata.")

    # --- Restore optimizer/scheduler state if checkpoint exists ---
    if args is not None and getattr(args, "lora_checkpoint", None) is not None:
        ckpt_path = args.lora_checkpoint
        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt_basename = os.path.basename(ckpt_path)
        # Derive training state dir from the LoRA checkpoint name
        # e.g., "step-500.safetensors" → "training_state_step-500/"
        #        "epoch-2.safetensors"  → "training_state_epoch-2/"
        tag = ckpt_basename.replace(".safetensors", "")
        state_dir = os.path.join(ckpt_dir, f"training_state_{tag}")
        restored = _load_training_state(accelerator, optimizer, scheduler, model_logger, state_dir)
        if not restored and accelerator.is_main_process:
            print(f"[Resume] WARNING: Training state restore failed. LoRA weights loaded but optimizer is fresh.")
            print(f"[Resume]          Training will continue but convergence may be affected.")
        if not restored:
            start_epoch = 0
            resume_step = None
            model_logger.num_steps = 0
            if accelerator.is_main_process:
                print("[Resume] Resetting progress to epoch 0 / step 0 because training state restore was incomplete.")

    # --- Determine start position ---
    # The dataloader uses a deterministic epoch seed, so we can resume at
    # batch granularity by consuming the already-finished batches in the
    # first resumed epoch and continuing with the restored optimizer state.
    if resume_step is not None and model_logger.num_steps == 0:
        # Only set if _load_training_state didn't already restore it
        model_logger.num_steps = resume_step
    resume_step_in_epoch = 0
    if start_epoch == 0 and resume_step is not None:
        start_epoch = resume_step // steps_per_epoch
        resume_step_in_epoch = resume_step % steps_per_epoch
        if accelerator.is_main_process:
            print(f"[Resume] Estimated start_epoch={start_epoch} from resume_step={resume_step}")
    elif start_epoch == 0 and model_logger.num_steps > 0:
        # Step counter was restored from training state, derive epoch
        start_epoch = model_logger.num_steps // steps_per_epoch
        resume_step_in_epoch = model_logger.num_steps % steps_per_epoch
        if accelerator.is_main_process:
            print(f"[Resume] Derived start_epoch={start_epoch} from restored num_steps={model_logger.num_steps}")
    elif start_epoch > 0 and model_logger.num_steps == 0:
        model_logger.num_steps = start_epoch * steps_per_epoch
        if accelerator.is_main_process:
            print(f"[Resume] start_epoch explicitly set to {start_epoch}; initializing step counter to {model_logger.num_steps}.")

    if accelerator.is_main_process and start_epoch > 0:
        print(f"[Resume] Starting from Epoch {start_epoch}, step counter = {model_logger.num_steps}")
    if accelerator.is_main_process and resume_step_in_epoch > 0:
        print(f"[Resume] Will skip {resume_step_in_epoch} already-finished batches in epoch {start_epoch}.")

    # --- Training loop ---
    for epoch_id in range(start_epoch, num_epochs):
        # Re-seed the shuffle generator for this epoch so data order
        # is epoch-dependent but reproducible across restarts.
        shuffle_generator.manual_seed(data_seed + epoch_id)
        # If accelerate wrapped the dataloader with a DistributedSampler,
        # set_epoch ensures proper shard rotation across processes.
        if hasattr(dataloader, "set_epoch"):
            dataloader.set_epoch(epoch_id)

        skip_batches = resume_step_in_epoch if epoch_id == start_epoch else 0
        progress = tqdm(enumerate(dataloader), total=steps_per_epoch,
                        desc=f"Epoch {epoch_id}/{num_epochs-1}")
        for batch_id, data in progress:
            if batch_id < skip_batches:
                progress.set_postfix(step=model_logger.num_steps, skip=f"{batch_id + 1}/{skip_batches}")
                continue
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if getattr(dataset, "load_from_cache", False):
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss)
                accelerator.log({"train_loss": loss.item()}, step=model_logger.num_steps)
                progress.set_postfix(loss=f"{loss.item():.4f}", step=model_logger.num_steps)
                scheduler.step()

                # Save training state alongside every step-based checkpoint.
                if save_steps is not None and model_logger.num_steps % save_steps == 0:
                    _save_training_state(
                        accelerator, optimizer, scheduler, model_logger,
                        model_logger.output_path, f"step-{model_logger.num_steps}",
                    )
        resume_step_in_epoch = 0

        # Save training state at end of each epoch.
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
        _save_training_state(
            accelerator, optimizer, scheduler, model_logger,
            model_logger.output_path, f"epoch-{epoch_id}",
        )

    model_logger.on_training_end(accelerator, model, save_steps)
    # If on_training_end saved a final step checkpoint (when num_steps
    # is not aligned with save_steps), also save the corresponding
    # training state so it can be used for future resume.
    if save_steps is not None and model_logger.num_steps % save_steps != 0:
        _save_training_state(
            accelerator, optimizer, scheduler, model_logger,
            model_logger.output_path, f"step-{model_logger.num_steps}",
        )


def wan_parser():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser = add_general_config(parser)
    parser = add_video_size_config(parser)
    parser.add_argument("--tokenizer_path", type=str, default=None, help="Path to tokenizer.")
    parser.add_argument("--audio_processor_path", type=str, default=None, help="Path to the audio processor. If provided, the processor will be used for Wan2.2-S2V model.")
    parser.add_argument("--max_timestep_boundary", type=float, default=1.0, help="Max timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--min_timestep_boundary", type=float, default=0.0, help="Min timestep boundary (for mixed models, e.g., Wan-AI/Wan2.2-I2V-A14B).")
    parser.add_argument("--initialize_model_on_cpu", default=False, action="store_true", help="Whether to initialize models on CPU.")
    parser.add_argument("--start_epoch", type=int, default=0, help="Resume training from this epoch index (0-based).")
    parser.add_argument("--resume_step", type=int, default=None, help="Resume training from this global step index.")
    return parser


if __name__ == "__main__":
    parser = wan_parser()
    args = parser.parse_args()

    # ================= Auto Resume Logic =================
    def _parse_ckpt_name(file_name):
        if not file_name.endswith(".safetensors"):
            return None
        if file_name.startswith("epoch-"):
            try:
                idx = int(file_name.split("-")[1].split(".")[0])
                return ("epoch", idx)
            except:
                return None
        if file_name.startswith("step-"):
            try:
                idx = int(file_name.split("-")[1].split(".")[0])
                return ("step", idx)
            except:
                return None
        return None

    def _find_latest_checkpoint(output_path):
        candidates = []
        for file in os.listdir(output_path):
            parsed = _parse_ckpt_name(file)
            if parsed is None:
                continue
            full_path = os.path.join(output_path, file)
            try:
                mtime = os.path.getmtime(full_path)
            except OSError:
                continue
            kind, idx = parsed
            # Check if a matching training_state directory exists.
            tag = file.replace(".safetensors", "")
            state_dir = os.path.join(output_path, f"training_state_{tag}")
            has_state = os.path.isdir(state_dir)
            progress = None
            if has_state:
                step_file = os.path.join(state_dir, "step_counter.pt")
                if os.path.exists(step_file):
                    try:
                        step_data = torch.load(step_file, map_location="cpu", weights_only=False)
                        progress = int(step_data.get("num_steps", 0))
                    except Exception:
                        progress = None
            candidates.append((has_state, progress if progress is not None else -1, mtime, idx, kind, full_path))
        if not candidates:
            return None
        # Prefer checkpoints WITH matching training state; among those, pick
        # the highest recorded global step, then fall back to filesystem time.
        candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
        has_state, progress, mtime, idx, kind, full_path = candidates[-1]
        if not has_state:
            print(f"[Auto Resume] WARNING: No checkpoint has a matching training_state directory. "
                  f"Optimizer will start fresh.")
        return {"path": full_path, "kind": kind, "index": idx, "mtime": mtime, "progress": progress}

    def _apply_resume_from_ckpt(args, ckpt_info, reason):
        if ckpt_info is None:
            return
        kind = ckpt_info["kind"]
        idx = ckpt_info["index"]
        if kind == "step":
            if getattr(args, "resume_step", None) is None and getattr(args, "start_epoch", 0) == 0:
                args.resume_step = idx
                print(f"[Auto Resume] {reason}: resume_step = {idx}")
        elif kind == "epoch":
            if getattr(args, "resume_step", None) is None and getattr(args, "start_epoch", 0) == 0:
                args.start_epoch = idx + 1
                print(f"[Auto Resume] {reason}: start_epoch = {args.start_epoch}")

    if args.lora_checkpoint is None and os.path.exists(args.output_path) and getattr(args, "resume_step", None) is None and getattr(args, "start_epoch", 0) == 0:
        latest_ckpt = _find_latest_checkpoint(args.output_path)
        if latest_ckpt:
            print(f"[Auto Resume] Found latest checkpoint: {latest_ckpt['path']}")
            print(f"[Auto Resume] Setting lora_checkpoint = {latest_ckpt['path']}")
            args.lora_checkpoint = latest_ckpt["path"]
            _apply_resume_from_ckpt(args, latest_ckpt, "Auto resume")
    elif args.lora_checkpoint is not None:
        parsed = _parse_ckpt_name(os.path.basename(args.lora_checkpoint))
        if parsed is not None:
            kind, idx = parsed
            _apply_resume_from_ckpt(args, {"kind": kind, "index": idx}, "Manual checkpoint hint")
        else:
            print(f"[Auto Resume] Checkpoint name not recognized: {args.lora_checkpoint}. Progress will start from step 0.")
    # =====================================================

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
        log_with="tensorboard",
        project_dir=args.output_path,
    )
    accelerator.init_trackers(project_name="wan_train_logs", config=vars(args))
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
        }
    )
    model = WanTrainingModule(
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        task=args.task,
        device="cpu" if args.initialize_model_on_cpu else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,
        "direct_distill:data_process": launch_data_process_task,
        "sft": launch_training_task_resume,
        "sft:train": launch_training_task_resume,
        "direct_distill": launch_training_task,
        "direct_distill:train": launch_training_task,
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)
