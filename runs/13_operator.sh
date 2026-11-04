#!/usr/bin/env bash
# Operator-level and structural variants on the cosine-convolution trunk.
#
# Paper: Table 4, all fourteen routes plus the C0 control.
# Protocol: SGD 0.01 / wd 1e-4 / 224 px / 150 epochs / cuDNN off, ResNet-18 unless stated.
# Seeds: 42 123 456
# Cost: 44 runs x ~25 min; three routes are repeated on six seeds.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
{
  # C0: the 8-layer cosine-convolution trunk, the reference every route is paired to
  for s in 42 123 456 789 1024 2024; do
    echo "results_mild/C0_v1_k0_l0.0_nodno_s${s}|${s}|--config cpc --op v1|"
  done

  # dual prototype, stage 4 only, second direction mirrored or drawn fresh
  for s in 42 123 456; do
    echo "results_dual/M1_s${s}|${s}|--config cpc --cpc_stages 0,0,1,1 --cpc_pos both --dual_stages 0,0,0,1 --dual_agg soft --beta 10.0 --dual_delta 0.10 --op v1|"
    echo "results_dual/M1b_s${s}|${s}|--config cpc --cpc_stages 0,0,1,1 --cpc_pos both --dual_stages 0,0,0,1 --dual_agg soft --beta 10.0 --dual_delta 0.30 --op v1|"
    echo "results_dual/M2_s${s}|${s}|--config cpc --cpc_stages 0,0,1,1 --cpc_pos both --dual_stages 0,0,0,1 --dual_agg max --beta 10.0 --dual_delta 0.10 --op v1|"
    echo "results_dual/M3c_s${s}|${s}|--config cpc --cpc_stages 0,0,1,1 --cpc_pos both --dual_stages 0,0,0,1 --dual_agg mean --beta 10.0 --dual_delta 0.10 --op v1|"
    echo "results_dual2/M4m_s${s}|${s}|--config cpc --cpc_stages 0,0,1,1 --cpc_pos both --dual_stages 0,0,0,1 --dual_agg max --beta 10.0 --dual_init independent --dual_delta 0.10 --op v1|"
    echo "results_dual2/M4s_s${s}|${s}|--config cpc --cpc_stages 0,0,1,1 --cpc_pos both --dual_stages 0,0,0,1 --dual_agg soft --beta 10.0 --dual_init independent --dual_delta 0.10 --op v1|"
    echo "results_cpcpos/A_conv2_s${s}|${s}|--config cpc --cpc_pos conv2 --op v1|"
    echo "results_cpcpos/B_conv1_s${s}|${s}|--config cpc --cpc_pos conv1 --op v1|"
    echo "results_cpcstages/C16_s${s}|${s}|--config cpc --cpc_stages 1,1,1,1 --cpc_pos both --op v1|"
    echo "results_mild/C1_soft_k0.01_l0.0_nodno_s${s}|${s}|--config cpc --op soft --kappa 0.01 --lambda_dir 0|"
    echo "results_mild/C1_soft_k0.05_l0.0_nodno_s${s}|${s}|--config cpc --op soft --kappa 0.05 --lambda_dir 0|"
    echo "results_mild/C2_mild_k0_l0.1_nodno_s${s}|${s}|--config cpc --op mild --kappa 0 --lambda_dir 0.1|"
    echo "results_mild/C2_mild_k0_l0.3_nodno_s${s}|${s}|--config cpc --op mild --kappa 0 --lambda_dir 0.3|"
    echo "results_mild/C3_v1_k0_l0.0_nodyes_s${s}|${s}|--config cpc --op v1 --proto_no_wd|"
  done

  # three routes were carried to six seeds, so they are paired against the six-seed C0
  # rather than the three-seed one; the extra seeds come first
  for s in 789 1024 2024; do
    echo "results_dual/M3c_s${s}|${s}|--config cpc --cpc_stages 0,0,1,1 --cpc_pos both --dual_stages 0,0,0,1 --dual_agg mean --beta 10.0 --dual_delta 0.10 --op v1|"
    echo "results_dual2/M4m_s${s}|${s}|--config cpc --cpc_stages 0,0,1,1 --cpc_pos both --dual_stages 0,0,0,1 --dual_agg max --beta 10.0 --dual_init independent --dual_delta 0.10 --op v1|"
    echo "results_cpcstages/C12_s${s}|${s}|--config cpc --cpc_stages 0,1,1,1 --cpc_pos both --op v1|"
  done
  for s in 42 123 456; do
    echo "results_cpcstages/C12_s${s}|${s}|--config cpc --cpc_stages 0,1,1,1 --cpc_pos both --op v1|"
  done
} | bash "$HERE/_launch.sh"
