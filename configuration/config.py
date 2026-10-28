import argparse

from datasets import DATASETS
from methods import METHODS


def base_parser():
    parser = argparse.ArgumentParser(description="Class Incremental Learning Research")

    # ========== Experiment configuration ==========
    parser.add_argument("--seeds", type=int, nargs="+", default=[1])
    parser.add_argument("--note", type=str, default="", help="Short description of the exp")
    parser.add_argument("--log_path", type=str, default="results", help="The path logs are saved.")

    # ============ Model configuration =============
    parser.add_argument("--method", type=str, default="l2p", help="Method name", choices=METHODS.keys())
    parser.add_argument("--backbone", type=str, default="vit_base_patch16_224", help="Backbone name")

    # =========== Dataset configuration ============
    parser.add_argument("--dataset", type=str, default="cifar10", help="dataset name", choices=DATASETS.keys())
    parser.add_argument("--data_dir", type=str, default="./data", help="Dataset root directory (see README for expected layouts per dataset).")
    parser.add_argument("--n_tasks", type=int, default=5, help="The number of tasks")
    parser.add_argument("--step_num", type=int, default=-1,
                        help="Number of internal steps for DualPrompt/MVP/FlyPrompt. Only values > 1 enable sample-based internal step scheduling; otherwise these methods follow task boundaries.")

    parser.add_argument("--n", type=int, default=50, help="The percentage of disjoint split. Disjoint=100, Blurry=0")
    parser.add_argument("--m", type=int, default=10, help="The percentage of blurry samples in blurry split. Uniform split=100, Disjoint=0")
    parser.add_argument("--rnd_NM", action='store_true', default=False, help="if True, N and M are randomly mixed over tasks.")

    # =========== Training configuration ===========
    parser.add_argument("--opt_name", type=str, default="sgd", help="Optimizer name")
    parser.add_argument("--sched_name", type=str, default="default", help="Scheduler name")
    parser.add_argument("--use_amp", action="store_true", default=False, help="Use automatic mixed precision.")
    parser.add_argument("--n_worker", type=int, default=0, help="The number of workers")
    parser.add_argument("--batchsize", type=int, default=16, help="batch size")
    parser.add_argument("--lr", type=float, default=0.05, help="learning rate")
    parser.add_argument("--num_epochs", type=int, default=1, help="number of epoch.")
    parser.add_argument("--online_iter", type=float, default=1, help="number of model updates per samples seen.")

    parser.add_argument("--transforms", nargs="*", default=['cutmix', 'autoaug'], help="Additional train transforms [cutmix, cutout, autoaug]")
    parser.add_argument("--no_batchmask", action="store_true", default=False, help="Disable batch mask, use seen mask")

    # ========== Evaluation configuration ==========
    parser.add_argument("--topk", type=int, default=1, help="set k when we want to set topk accuracy")
    parser.add_argument("--eval_period", type=int, default=100, help="evaluation period for true online setup")

    # ============= ViT configurations =============
    parser.add_argument('--profile', action='store_true', default=False, help='enable profiling for ViT_Prompt')

    # ============= EWC / LwF configurations ============
    parser.add_argument("--ewc_lambda", type=float, default=1000.0, help="EWC regularization strength.")
    parser.add_argument("--ewc_gamma", type=float, default=1.0, help="Decay for consolidated EWC Fisher before adding the current task Fisher.")
    parser.add_argument("--ewc_fisher_on_gpu", action=argparse.BooleanOptionalAction, default=True, help="Store EWC Fisher and parameter snapshot on GPU.")
    parser.add_argument("--ewc_empirical_labels", action=argparse.BooleanOptionalAction, default=True, help="Use ground-truth labels for empirical Fisher; otherwise use model predictions.")
    parser.add_argument("--lwf_lambda", type=float, default=1.0, help="LwF distillation loss weight.")
    parser.add_argument("--lwf_temperature", type=float, default=2.0, help="Temperature for LwF knowledge distillation.")
    parser.add_argument("--lwf_teacher_on_gpu", action=argparse.BooleanOptionalAction, default=True, help="Keep the frozen LwF teacher model on GPU.")

    # ============= MISA configurations ============
    parser.add_argument('--load_pt', action='store_true', default=False, help='load pretrained prompts (MISA)')
    parser.add_argument('--flyprompt_pt_path', type=str, default='./checkpoints/flyprompt_misa_prompt.pt',
                        help='Path to MISA-style pretrained FlyPrompt prompt checkpoint.')

    # ============= MePo configurations ============
    parser.add_argument('--mepo_backbone_path', type=str, default=None,
                        help='Path to pretrained backbone checkpoint for MEPO backbone override.')
    parser.add_argument('--cov_path', type=str, default=None,
                        help='Path to covariance matrix .npy for MEPO CLS calibration.')
    parser.add_argument('--cov_coef', type=float, default=0.7,
                        help='Interpolation coeff between original and MEPO-calibrated CLS (0-1).')

    # ======== HiDe / NoRGa configurations =========
    parser.add_argument("--lam_orth", type=float, default=1, help="Orthogonal loss weight for HiDe/NoRGa.")
    parser.add_argument("--ca_num_per_class", type=int, default=200, help="Number of CA samples per class for HiDe/NoRGa.")
    parser.add_argument("--ca_steps", type=int, default=200, help="Number of CA optimization steps for HiDe/NoRGa.")

    # ========== SD-LoRA configurations ==========
    parser.add_argument("--sdlora_rank", type=int, default=10, help="LoRA rank for SD-LoRA (default from original SD-LoRA).")
    parser.add_argument("--sdlora_alpha", type=float, default=0.8, help="Scaling factor alpha for SD-LoRA (default from original SD-LoRA).")
    parser.add_argument("--sdlora_layers", type=str, default="all", help="Which ViT blocks to apply LoRA to (e.g., 'all', 'last4').")
    parser.add_argument("--sdlora_ortho_weight", type=float, default=0.0, help="Orthogonal loss weight for SD-LoRA (0 means disabled).")

    # ========== FlyPrompt configurations ==========
    parser.add_argument("--len_prompt", type=int, default=20, help="The length of the prompt for each expert")
    parser.add_argument("--pos_prompt", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="The position of the prompt")
    parser.add_argument("--rp_dim", type=int, default=10000, help="The dimension of the random projection head")
    parser.add_argument("--rp_ridge", type=float, default=1e4, help="The ridge parameter for the random projection head")
    parser.add_argument("--ema_ratio", type=float, nargs="+", default=[0.9, 0.99], help="The EMA ratio for the expert FCs")
    parser.add_argument("--ensemble_method", type=str, default="softmax_max_prob", choices=["mean", "max_prob", "min_entropy", "softmax_mean", "softmax_max_prob", "softmax_min_entropy"],
                        help="Ensemble method for combining expert outputs: mean (average), max (maximum), min_entropy (minimum entropy), and softmax variants of these.")
    parser.add_argument("--router_type", type=str, default="rpfc", choices=["rpfc", "ws"],
                        help="Routing head type: rpfc (random projection FC) or ws (whitened subspace).")
    parser.add_argument("--ws_k", type=int, default=32,
                        help="Subspace rank for whitened subspace router.")
    parser.add_argument("--no_ema_ensemble", action="store_true", default=False,
                        help="Disable EMA head bank for FlyPrompt (only): skip EMA updates during training and use only the online fc head at evaluation.")
    parser.add_argument("--use_analytic_head", action="store_true", default=False,
                        help="Enable FlyPrompt frozen-feature analytic class head and extra per-eval diagnostics.")
    parser.add_argument("--use_analytic_gain", action="store_true", default=False,
                        help="Use stage-dependent analytic DAN gain as the FlyPrompt evaluation output.")
    parser.add_argument("--analytic_gain_max_lambda", type=float, default=0.5,
                        help="Maximum strength for the stage-dependent analytic DAN gain.")
    parser.add_argument("--analytic_gain_schedule", type=str, default="quadratic", choices=["quadratic", "linear", "constant"],
                        help="Stage schedule for analytic DAN gain strength.")

    # ========== FlyAdapter / FlyLoRA configurations ==========
    parser.add_argument("--fly_lora_rank", type=int, default=5,
                        help="LoRA rank for FlyLoRA expert modules.")
    parser.add_argument("--fly_lora_alpha", type=float, default=1.0,
                        help="Scaling factor alpha for FlyLoRA expert modules.")
    parser.add_argument("--fly_lora_layers", type=int, default=5,
                        help="Number of early ViT blocks using FlyLoRA K/V experts.")
    parser.add_argument("--fly_adapter_down_dim", type=int, default=10,
                        help="Adapter bottleneck dimension for FlyAdapter expert modules.")
    parser.add_argument("--fly_adapter_layers", type=int, default=5,
                        help="Number of early ViT blocks using FlyAdapter experts.")

    # ========== RPFC gating configurations ==========
    parser.add_argument("--use_rp_gate", action="store_true", default=False,
                        help="Use FlyPrompt-style RPFC head for task gating in compatible methods (e.g., SPrompt, HiDe/NoRGa, DualPrompt, MVP).")

    # ========== EMA head bank configurations ==========
    parser.add_argument("--use_ema_head", action="store_true", default=False,
                        help="Use EMA-based classifier head bank and ensemble in compatible methods (e.g., SPrompt, HiDe/NoRGa, DualPrompt, MVP).")

    # ======== Expert similarity analysis ==========
    parser.add_argument("--analysis_expert_similarity", action="store_true", default=False,
                        help="If set, run expert feature similarity / CKA (including residual vs common) analysis after training.")


    # CF-LoRA (closed-form head + slow LoRA)
    parser.add_argument('--cf_head', type=str, default='slda', choices=['slda', 'rp', 'ncm', 'randumb', 'linear'], help='CF-LoRA classifier head')
    parser.add_argument('--cf_rank', type=int, default=8, help='CF-LoRA rank (0 = no LoRA)')
    parser.add_argument('--cf_adapt', type=str, default='lora', choices=['lora', 'adapter', 'prefix', 'softprompt'], help='representation adaptation: LoRA on q/v (all blocks), RanPAC-style bottleneck adapters, K/V prefixes (E45) or soft prompt tokens (E45)')
    parser.add_argument('--cf_prompt_len', type=int, default=20, help='E45: prefix / soft prompt length')
    parser.add_argument('--cf_prompt_blocks', type=int, default=5, help='E45: blocks carrying prefixes / deep soft prompts')
    parser.add_argument('--cf_prompt_gate', action='store_true', help='E45: zero-init gated prefix attention (LLaMA-Adapter style)')
    parser.add_argument('--cf_gate_fixed', type=float, default=None, help='E63: prefix attention with its own softmax scaled by this constant instead of a learned zero-init gate (overrides --cf_prompt_gate)')
    parser.add_argument('--cf_prompt_v_zero', action='store_true', help='E63: initialise the prefix values P_V at zero (the prefix term starts at zero without a gate)')
    parser.add_argument('--cf_prompt_shallow', action='store_true', help='E45: soft prompt only at the input, carried through all blocks (VPT-shallow)')
    parser.add_argument('--cf_adapter_blocks', type=int, default=5)
    parser.add_argument('--cf_adapter_dim', type=int, default=64)
    parser.add_argument('--cf_alpha', type=float, default=16.0, help='CF-LoRA alpha')
    parser.add_argument('--cf_shrink', type=float, default=1e-2, help='SLDA covariance shrinkage')
    parser.add_argument('--cf_bias_corr', type=float, default=0.0, help='E34/E34b: SLDA small-n correction of the quadratic term, b_c += alpha * tr(Sig^-1 Sig_within) / (2 n_c); alpha in [0, 1], 0 = off')
    parser.add_argument('--cf_mean_shrink', type=float, default=0.0, help='E34: SLDA class means shrunk to the global mean with weight kappa/(n_c+kappa) (0 = off)')
    parser.add_argument('--cf_rp_dim', type=int, default=10000, help='RP head projection size')
    parser.add_argument('--cf_ridge', type=float, default=1e5, help='RP head ridge')
    parser.add_argument('--cf_refresh', type=int, default=10, help='RP head: refresh the solve every k batches')
    parser.add_argument('--cf_wd', type=float, default=0.05, help='AdamW weight decay on LoRA')
    parser.add_argument('--cf_warm', type=int, default=0, help='warm phase length in samples (0 = none)')
    parser.add_argument('--cf_warm_lr', type=float, default=5e-3, help='Adam lr for LoRA + linear head during the warm phase')
    parser.add_argument('--cf_after', type=str, default='slow', choices=['slow', 'freeze'], help='after the warm phase: slow LoRA at --lr, or frozen')
    parser.add_argument('--cf_warm_auto', action='store_true', default=False, help='end the warm phase when the closed-form head beats the linear head (EMA) for cf_auto_patience batches; cf_warm then acts as a cap (0 = no cap)')
    parser.add_argument('--cf_auto_patience', type=int, default=20)
    parser.add_argument('--cf_auto_scale', type=str, default='exposed', choices=['exposed', 'total'], help='class-scaled patience: multiply by exposed classes or by the total label-space size')
    parser.add_argument('--cf_auto_tol', type=float, default=1.0, help='fraction of the last P batches in which the closed-form head must be >= the plastic head (1.0 = consecutive)')
    parser.add_argument('--cf_auto_patience_per_class', type=float, default=0.0, help='if > 0, patience (in batches) = this value x number of exposed classes, instead of the fixed cf_auto_patience')
    parser.add_argument('--cf_auto_min', type=int, default=1000, help='minimum samples before the auto switch may fire')
    parser.add_argument('--cf_auto_margin', type=float, default=0.0, help='tolerance: the closed-form head counts as "not worse" when its EMA prequential accuracy is >= plastic - margin')
    parser.add_argument('--cf_auto_clean', action='store_true', default=False, help='E33b: the critic measures both heads on the clean (test-transform) view used for the statistics instead of the augmented training view')
    parser.add_argument('--cf_ref_stride', type=int, default=0, help='E35: after the freeze keep the reference forward (adapter off), the reference head and the paired gain signal alive every k-th batch (0 = only while plastic)')
    parser.add_argument('--cf_diag_ref_eval', action='store_true', default=False, help='E35 diagnostic: at every evaluation also score the reference head on the test set (extra pass, adapter off); logged in diag.json ref_eval')
    parser.add_argument('--cf_stop', type=str, default='critic', choices=['critic', 'gain'], help='E36: critic = freeze at the first admissible point; gain = the critic only sets the floor, then freeze at the plateau of the paired gain after a novelty event (growth rule)')
    parser.add_argument('--cf_gain_h', type=int, default=6400, help='E36: window H (samples) of the growth rule')
    parser.add_argument('--cf_gain_eps', type=float, default=0.005, help='E36: growth rule: novelty when the EMA gain grew by 2*eps since the floor; freeze when OLS slope over H times H <= eps; fallback freeze at floor + 2H without novelty')
    parser.add_argument('--cf_resume', type=str, default='none', choices=['none', 'gain'], help='E36: reopen the plastic phase when the paired gain drops below its value at the freeze by cf_resume_drop for cf_resume_patience sampled batches (needs --cf_ref_stride > 0)')
    parser.add_argument('--cf_resume_drop', type=float, default=0.02)
    parser.add_argument('--cf_resume_patience', type=int, default=25, help='E36: consecutive sampled batches (stride cf_ref_stride) below the drop before resuming')
    parser.add_argument('--cf_resid_guard', type=float, default=1.5, help='E36: in a resumed phase, freeze when the translator residual is >= guard x its value at the last freeze for 2 consecutive compensations (0 = off)')
    parser.add_argument('--cf_auto_horizon', type=int, default=50, help='hysteresis: number of batches over which the linear-head EMA improvement is measured')
    parser.add_argument('--cf_auto_plateau', type=float, default=0.0, help='hysteresis: also require the linear head EMA accuracy to have improved less than this over the last 50 batches (0 = off)')
    parser.add_argument('--cf_probe_alive', action='store_true', default=False, help='after the freeze keep training the linear head on detached features (no extra forward) and log both heads\' prequential EMA for the whole stream')
    parser.add_argument('--cf_rearm', action='store_true', default=False, help='re-open the plastic phase when the probe EMA >= closed EMA + cf_rearm_margin for cf_rearm_patience batches (implies --cf_probe_alive)')
    parser.add_argument('--cf_rearm_margin', type=float, default=0.02)
    parser.add_argument('--cf_rearm_patience', type=int, default=0, help='0 = same P as the freeze rule')
    parser.add_argument('--cf_rearm_min_count', type=int, default=50, help='classes with fewer closed-head samples are ignored in the re-arm gap')
    parser.add_argument('--cf_comp', type=str, default='none', choices=['none', 'ref'], help='E31: at the freeze replace the class means by reference-space (adapter off) means pushed through an online affine translator (one extra forward per batch in the plastic phase)')
    parser.add_argument('--cf_comp_decay', type=float, default=0.99, help='per-batch forgetting of the translator normal equations')
    parser.add_argument('--cf_comp_ridge', type=float, default=1.0)
    parser.add_argument('--cf_comp_every', type=int, default=0, help='E32: while the adapter is plastic, re-apply the reference-mean compensation every N samples and let the closed-form head predict (0 = only at the freeze)')
    parser.add_argument('--cf_clean_stats', action='store_true', default=False, help='E31a: accumulate the closed-form statistics on the clean (test-transform) view; after the freeze the single forward runs on the clean view')
    parser.add_argument('--cf_rda_p', type=int, default=0, help='E39b: anchored RDA head after the freeze, class-specific covariance in a p-dim subspace (0 = off; needs --cf_comp ref)')
    parser.add_argument('--cf_rda_nu', type=float, default=1000.0, help='E39b: strength of the shared-covariance prior (nu -> inf = SLDA)')
    parser.add_argument('--cf_rda_basis', type=int, default=2000, help='E39b: samples after which the frozen-space PCA basis is fixed')
    parser.add_argument('--cf_dump_feats', action='store_true', default=False, help='DIAGNOSTIC (E39a): at stream end save clean train (seen) and test features under the adapted and the frozen backbone')
    parser.add_argument('--cf_dump_trajectory', action='store_true', default=False, help='DIAGNOSTIC (E67): save arrival-time clean train features and current test features at each evaluation; never used by the learner')
    parser.add_argument('--cf_drift_log', action='store_true', default=False, help='DIAGNOSTIC (E56): at every evaluation log the stored SLDA means against the test-set centroids under the current model (rel, Mahalanobis) and the SLDA accuracy with the stored vs 2-fold oracle means; no extra forward')
    parser.add_argument('--cf_oracle', action='store_true', default=False, help='DIAGNOSTIC: at each task end recompute statistics over all seen images with the current adapter (image replay) and fit an affine drift map')
    parser.add_argument('--stream', type=str, default='siblurry', choices=['siblurry', 'cir', 'rendition', 'fewshot'], help='stream generator: Si-Blurry (n, m), class-incremental with repetition, or recurring rendition blocks (ImageNet-R)')
    parser.add_argument('--rend_seq', type=str, nargs='+', default=['sketch', 'cartoon', 'painting', 'sketch', 'tattoo', 'cartoon', 'misc', 'sketch'], help='rendition stream: sequence of experiences (n_tasks = its length), or "random" for a seeded Markov sequence')
    parser.add_argument('--rend_len', type=int, default=8, help='rendition stream: length of the random sequence')
    parser.add_argument('--rend_p_return', type=float, default=0.5, help='rendition stream: probability of returning to an already seen domain')
    parser.add_argument('--fs_shots', type=int, default=10, help='few-shot stream: samples per novel class')
    parser.add_argument('--fs_base_frac', type=float, default=0.5, help='few-shot stream: fraction of classes in the base experience')
    parser.add_argument('--fs_trickle', type=float, default=0.2, help='few-shot stream: fraction of base-class samples spread over the later experiences')
    parser.add_argument('--cir_p_rep', type=float, default=0.3, help='CIR: per-class probability of re-appearing in each later experience')
    parser.add_argument('--cir_first', type=str, default='uniform', choices=['uniform', 'front'], help='CIR: first-occurrence distribution')
    parser.add_argument('--ranpac_first_tasks', type=int, default=1, help='RanPAC: number of initial tasks forming the first (adaptation) session')
    parser.add_argument('--ranpac_first_samples', type=int, default=0, help='RanPAC: end the first (adaptation) session after this many stream samples instead of at a task boundary (0 = task mode)')
    parser.add_argument('--stream_stride', type=int, default=1, help='compute-matched streaming: learn only on every k-th batch (others pass by)')
    parser.add_argument('--sp_blocks', type=int, default=5, help='SinglePrompt: prefix prompts on the first k blocks')
    parser.add_argument('--sp_len', type=int, default=20, help='SinglePrompt: prompt length (key and value prefixes)')
    parser.add_argument('--sp_tau', type=float, default=0.1, help='SinglePrompt: cosine-classifier temperature')
    parser.add_argument('--cf_telemetry', action='store_true', default=False, help='DIAGNOSTIC: per-batch feature drift (extra forwards)')

    args = parser.parse_args()
    return args
