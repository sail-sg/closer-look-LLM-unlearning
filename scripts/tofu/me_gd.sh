MASTER_PORT=$((RANDOM % 50001 + 10000))

forget_losses=(
    ME+GD
)

# You can specify any forget task from 1 to 10
# the standard TOFU benchmark is task 1
task_list=(1)

# pass to python script
export TASK_LIST=$(IFS=,; echo "${task_list[*]}")

learning_rates=(
    1e-5
)



use_LoRA=false
save_root=results/tofu
mask=false

forget_coeff=0.1
regularization_coeff=1.0

save_checkpoint=false


num_epochs=5

### evaluate only at the last epoch
save_steps=last
eval_steps=(last)

### evaluate at each unlearning epoch
# save_steps=steps_per_epoch
# eval_steps=(0 1 2 3 4 5 6 7 8 9 10 11 12 18 24 30 31 36 42 48 54 60 62 72 84 96 108 120 125)

split=forget01 # forget01/forget05/forget10
for forget_loss in ${forget_losses[@]}; do
    for lr in ${learning_rates[@]}; do
        for task_id in ${task_list[@]}; do
            COMMON="use_LoRA=$use_LoRA forget_coeff=$forget_coeff regularization_coeff=$regularization_coeff lr=$lr split=$split forget_loss=$forget_loss num_epochs=$num_epochs \
                mask=$mask fix_ref_model=$fix_ref_model save_root=$save_root save_checkpoint=$save_checkpoint"
            CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$MASTER_PORT \
                    forget.py \
                    --config-name=tofu.yaml \
                    task_id=$task_id \
                    save_steps=$save_steps \
                    $COMMON
            for step in ${eval_steps[@]}; do
                CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
                        eval.py \
                        --config-name=tofu.yaml \
                        task_id=$task_id \
                        eval_unlearn_step=$step \
                        $COMMON
            done
        done
    done
done

split=forget05 # forget01/forget05/forget10
for forget_loss in ${forget_losses[@]}; do
    for lr in ${learning_rates[@]}; do
        for task_id in ${task_list[@]}; do
            COMMON="use_LoRA=$use_LoRA forget_coeff=$forget_coeff regularization_coeff=$regularization_coeff lr=$lr split=$split forget_loss=$forget_loss num_epochs=$num_epochs \
                mask=$mask fix_ref_model=$fix_ref_model save_root=$save_root save_checkpoint=$save_checkpoint"
            CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$MASTER_PORT \
                    forget.py \
                    --config-name=tofu.yaml \
                    task_id=$task_id \
                    save_steps=$save_steps \
                    $COMMON
            for step in ${eval_steps[@]}; do
                CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
                        eval.py \
                        --config-name=tofu.yaml \
                        task_id=$task_id \
                        eval_unlearn_step=$step \
                        $COMMON
            done
        done
    done
done

split=forget10 # forget01/forget05/forget10
for forget_loss in ${forget_losses[@]}; do
    for lr in ${learning_rates[@]}; do
        for task_id in ${task_list[@]}; do
            COMMON="use_LoRA=$use_LoRA forget_coeff=$forget_coeff regularization_coeff=$regularization_coeff lr=$lr split=$split forget_loss=$forget_loss num_epochs=$num_epochs \
                mask=$mask fix_ref_model=$fix_ref_model save_root=$save_root save_checkpoint=$save_checkpoint"
            CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$MASTER_PORT \
                    forget.py \
                    --config-name=tofu.yaml \
                    task_id=$task_id \
                    save_steps=$save_steps \
                    $COMMON
            for step in ${eval_steps[@]}; do
                CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$MASTER_PORT \
                        eval.py \
                        --config-name=tofu.yaml \
                        task_id=$task_id \
                        eval_unlearn_step=$step \
                        $COMMON
            done
        done
    done
done