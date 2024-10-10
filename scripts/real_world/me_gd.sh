MASTER_PORT=$((RANDOM % 50001 + 10000))


forget_losses=(
    ME+GD
)

learning_rates=(
    # 1e-5
    # 2e-6
    5e-6
)

mask=false
mask_template=true

fix_ref_model=false

split=forget
retain=neighbor

use_LoRA=false
forget_coeff=0.5
regularization_coeff=1.0

# evluate at each unlearning epoch
save_steps=last
eval_steps=(last)

save_checkpoint=false

save_root=results/real_world

num_epochs=5
for forget_loss in ${forget_losses[@]}; do
    for lr in ${learning_rates[@]}; do
        COMMON="use_LoRA=$use_LoRA forget_coeff=$forget_coeff regularization_coeff=$regularization_coeff lr=$lr split=$split forget_loss=$forget_loss num_epochs=$num_epochs \
            mask=$mask fix_ref_model=$fix_ref_model save_root=$save_root save_checkpoint=$save_checkpoint"
        CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$MASTER_PORT \
                real_world_unlearn.py \
                --config-name=real_world.yaml \
                save_steps=$save_steps \
                $COMMON
        for step in ${eval_steps[@]}; do
            CUDA_VISIBLE_DEVICES=0,1 python \
                    real_world_eval.py \
                    --config-name=real_world.yaml \
                    eval_unlearn_step=$step \
                    $COMMON
        done
    done
done
