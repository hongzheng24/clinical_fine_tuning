from src.agents.coding_agent import _SYSTEM_PROMPT, _build_prompt
from src.utils.state import PipelineState
from src.preprocess.load_medcoder import ExampleClass
from src.utils.pipeline import run_pipeline
import json

from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from dataset import load_dataset

MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"

DATA_DIR = "/users/hzheng29/data/hzheng29/my_clinical_fine_tuning/data/processed_data"          # from prepare_data.prepare()
OUTPUT_DIR = "/users/hzheng29/data/hzheng29/my_clinical_fine_tuning/src/models/checkpoints/sft_lora"


def fine_tune(train_examples: list[ExampleClass], test_examples: list[ExampleClass]):

    # Load dataset
    dataset = load_dataset("json", data_files={
        "train": f"{DATA_DIR}/sft_train.jsonl",
        "validation": f"{DATA_DIR}/sft_val.jsonl",
    })


    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype="auto", device_map="auto")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,      # effective batch size 16
        learning_rate=2e-4,                 # LoRA tolerates a higher LR than full fine-tuning
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,                          # set False and fp16=True if bf16 unsupported
        max_length=2048,
        packing=False,                      # keep examples un-packed: each is a complete, independent task
        assistant_only_loss=True,           # mask loss to the assistant turn -- see module docstring
        report_to="none",                   # set to "wandb"/"tensorboard" if you want run tracking
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    model = trainer.train()
    trainer.save_model(OUTPUT_DIR)  # saves the LoRA adapter, not a full model copy
    print(f"adapter saved to {OUTPUT_DIR}")
    return {
        'model': model,
        'loss': model.training_loss()
    }


def main():
    # fine_tune(..., ..)
    pass


if __name__ == "__main__":
    main()