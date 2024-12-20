from sample_factory.launcher.run_description import Experiment, ParamGrid, RunDescription
from swarm_rl.runs.obstacles.quad_obstacle_baseline import QUAD_BASELINE_CLI_8
#PBT（Population-Based Training）：一種自適應訓練方法，讓多個模型共同訓練並相互分享超參數
_params = ParamGrid(
    [
        ("with_pbt", ["True"]),
    ]
)

OBSTACLE_MODEL_CLI = QUAD_BASELINE_CLI_8 + (
    # PBT 設置政策數量（--num_policies=8）、交換頻率、獎勵差距閾值等。
    ' --num_policies=8 --pbt_mix_policies_in_one_env=True --pbt_period_env_steps=10000000 '
    '--pbt_start_mutation=50000000 --pbt_replace_reward_gap=0.2 --pbt_replace_reward_gap_absolute=3.0 '
    '--pbt_optimize_gamma=True --pbt_perturb_max=1.2 '
    # Pre-set hyperparameters 探索損失係數、熵係數等 調整探索讓他一直去探索 0.0005改成0.001增強了代理的探索行為，減少因早期過度專注於局部最優策略的風險。
    '--exploration_loss_coeff=0.001 '#--max_entropy_coeff=0.0005 如果熵懲罰係數過低，可能導致模型過早收斂到次優策略；若完全移除，需依賴任務本身的隨機性促進探索。
    '--anneal_collision_steps=0 --train_for_env_steps=10000000000 '
    # Num workers 設定工作進程（num_workers）、每個工作進程的環境數量 減少了計算資源需求，特別是在 CPU 負載高的情況下
    '--num_workers=34 --num_envs_per_worker=2 --quads_num_agents=8 '
    # Neighbor & General Encoder for obst & neighbor設置鄰居的可見數量和編碼器類型
    '--quads_neighbor_visible_num=6 --quads_neighbor_obs_type=pos_vel --quads_encoder_type=attention '
    # WandB
    '--with_wandb=False --wandb_project=Quad-Swarm-RL --wandb_user=multi-drones '
    '--wandb_group=pbt_obstacle_multi_attn_v2'
)

_experiment = Experiment(
    "pbt_obstacle_multi_attn_v2",
    OBSTACLE_MODEL_CLI,
    _params.generate_params(randomize=False),
)

RUN_DESCRIPTION = RunDescription("obstacles_multi", experiments=[_experiment])