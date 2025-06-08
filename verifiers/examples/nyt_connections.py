import verifiers as vf
from verifiers.envs.nyt_connections_env import NYTConnectionsEnv

"""
Inference:
CUDA_VISIBLE_DEVICES=0,1,2,3 vf-vllm --model willcb/Qwen2.5-7B-NYTConnections-SFT --tensor-parallel-size 4

Training:
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config-file configs/zero3.yaml --num-processes 4 verifiers/examples/nyt_connections.py
"""

size = '7B'
# model_name = f'willcb/Qwen2.5-{size}-NYTConnections-SFT'
model_name = 'Qwen/Qwen2.5-7B-Instruct'
model, tokenizer = vf.get_model_and_tokenizer(model_name)

vf_env = NYTConnectionsEnv(
    num_eval_samples=20,
    max_concurrent=32,
    max_turns=10  # Maximum number of guesses allowed
)

run_name = f"nyt-connections-grpo-{size}"
training_args = vf.grpo_defaults(run_name=run_name)
training_args.num_iterations = 3
training_args.per_device_train_batch_size = 8
training_args.num_generations = 12
training_args.gradient_accumulation_steps = 4
training_args.max_prompt_length = 500
training_args.max_completion_length = 4096
training_args.max_steps = 10
training_args.mask_env_responses = True

trainer = vf.GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    env=vf_env,
    args=training_args,
)
trainer.train()
