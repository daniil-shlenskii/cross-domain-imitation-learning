export CUDA_VISIBLE_DEVICES=3

EXP_DIR=configs/experiments/orthogonal_grads/main
PROJECT_NAME=OrthogonalGrads

for N_DOMAINS in one_domain two_domains
do
    for PROJ_DIRECTION in state_to_policy policy_to_state both
    do
        for CONFIG_NAME in half_cheetah hopper
        do
            CONFIG_PATH=${EXP_DIR}/${N_DOMAINS}/${PROJ_DIRECTION}/${CONFIG_NAME}.yaml
            python -m train_scripts.train_agent --config $CONFIG_PATH -w --from_scratch --wandb_project ${PROJECT_NAME}
        done
    done
done

