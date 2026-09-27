"""Comprehensive evaluation metrics for business entity resolution.

Calculates:
1. Classification: Accuracy, Precision, Recall, Specificity, F1, F0.5, F2, Balanced Accuracy,
   MCC, Cohen's Kappa, ROC-AUC, PR-AUC/AP, Log Loss, Brier Score.
2. Confusion Matrix: TP, TN, FP, FN.
3. Retrieval / Ranking: Candidate Recall, Recall@K, Precision@K, Hit@K, MRR, MAP@K,
   Avg Candidates/S1, candidate reduction %.
4. Competition: S1 Macro F0.5, Precision, Recall, Singleton performance, Multi-match performance,
   per-country and match-count bucket breakdowns.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .evaluate import f05_for_entity
from .normalize import NormalizedEntity


def compute_all_metrics(
    val_s1: list[NormalizedEntity],
    val_truth: dict[str, set[str]],
    val_cand_predictions: dict[str, list[tuple[str, float, list[float]]]],
    val_final_predictions: dict[str, set[str]],
    total_target_pool_size: int = 10_000_000,
) -> dict[str, Any]:
    """Compute comprehensive classification, ranking, and competition metrics."""
    # ------------------------------------------------------------------
    # 1. Pair-Level Classification Metrics & Confusion Matrix
    # ------------------------------------------------------------------
    all_y_true: list[int] = []
    all_y_pred: list[int] = []
    all_y_prob: list[float] = []

    for s1 in val_s1:
        sid = s1.entity_id
        true_set = val_truth.get(sid, set())
        pred_set = val_final_predictions.get(sid, set())
        cands = val_cand_predictions.get(sid, [])

        for cid, prob, _ in cands:
            label = 1 if cid in true_set else 0
            is_pred = 1 if cid in pred_set else 0
            all_y_true.append(label)
            all_y_pred.append(is_pred)
            all_y_prob.append(float(np.clip(prob, 1e-7, 1.0 - 1e-7)))

    y_t = np.array(all_y_true, dtype=np.int32)
    y_p = np.array(all_y_pred, dtype=np.int32)
    y_prob = np.array(all_y_prob, dtype=np.float64)

    if len(y_t) > 0 and (y_t == 1).any() and (y_t == 0).any():
        tn, fp, fn, tp = confusion_matrix(y_t, y_p, labels=[0, 1]).ravel()
        pair_acc = float(accuracy_score(y_t, y_p))
        pair_prec = float(precision_score(y_t, y_p, zero_division=0))
        pair_rec = float(recall_score(y_t, y_p, zero_division=0))
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        pair_f1 = float(f1_score(y_t, y_p, zero_division=0))
        pair_f05 = float(fbeta_score(y_t, y_p, beta=0.5, zero_division=0))
        pair_f2 = float(fbeta_score(y_t, y_p, beta=2.0, zero_division=0))
        balanced_acc = float((pair_rec + specificity) / 2.0)
        mcc = float(matthews_corrcoef(y_t, y_p))
        kappa = float(cohen_kappa_score(y_t, y_p))
        try:
            roc_auc = float(roc_auc_score(y_t, y_prob))
        except Exception:
            roc_auc = 0.0
        try:
            pr_auc = float(average_precision_score(y_t, y_prob))
        except Exception:
            pr_auc = 0.0
        try:
            lloss = float(log_loss(y_t, y_prob))
        except Exception:
            lloss = 0.0
        brier = float(brier_score_loss(y_t, y_prob))
    else:
        tn = fp = fn = tp = 0
        pair_acc = pair_prec = pair_rec = specificity = pair_f1 = pair_f05 = pair_f2 = 0.0
        balanced_acc = mcc = kappa = roc_auc = pr_auc = lloss = brier = 0.0

    # ------------------------------------------------------------------
    # 2. Retrieval & Ranking Metrics
    # ------------------------------------------------------------------
    k_vals = [1, 3, 5, 10]
    hits_at_k = {k: 0 for k in k_vals}
    rec_at_k = {k: [] for k in k_vals}
    prec_at_k = {k: [] for k in k_vals}
    reciprocal_ranks = []
    avg_precisions = []

    total_eval_queries = 0
    s1_with_truth = 0
    cand_recall_hits = 0
    total_true_matches = 0
    retrieved_true_matches = 0
    total_candidates_retrieved = 0

    for s1 in val_s1:
        sid = s1.entity_id
        true_set = val_truth.get(sid, set())
        cands = val_cand_predictions.get(sid, [])
        # sorted by prob descending (ranking)
        ranked_cands = sorted(cands, key=lambda x: x[1], reverse=True)
        ranked_ids = [c[0] for c in ranked_cands]

        total_eval_queries += 1
        total_candidates_retrieved += len(ranked_ids)

        if true_set:
            s1_with_truth += 1
            total_true_matches += len(true_set)
            retrieved_in_cands = len(set(ranked_ids) & true_set)
            retrieved_true_matches += retrieved_in_cands
            if retrieved_in_cands > 0:
                cand_recall_hits += 1

            # Reciprocal rank (MRR)
            first_hit_rank = None
            for rank_idx, cid in enumerate(ranked_ids, 1):
                if cid in true_set:
                    first_hit_rank = rank_idx
                    break
            reciprocal_ranks.append(1.0 / first_hit_rank if first_hit_rank else 0.0)

            # MAP
            cum_hits = 0
            prec_sum = 0.0
            for rank_idx, cid in enumerate(ranked_ids, 1):
                if cid in true_set:
                    cum_hits += 1
                    prec_sum += cum_hits / rank_idx
            avg_precisions.append(prec_sum / len(true_set) if true_set else 0.0)

            # Metrics @ K
            for k in k_vals:
                top_k = set(ranked_ids[:k])
                k_hits = len(top_k & true_set)
                if k_hits > 0:
                    hits_at_k[k] += 1
                rec_at_k[k].append(k_hits / len(true_set))
                prec_at_k[k].append(k_hits / k)

    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    map_k = float(np.mean(avg_precisions)) if avg_precisions else 0.0
    cand_recall = cand_recall_hits / s1_with_truth if s1_with_truth > 0 else 0.0
    all_target_cand_recall = (
        retrieved_true_matches / total_true_matches if total_true_matches > 0 else 0.0
    )
    avg_cands_per_s1 = total_candidates_retrieved / max(total_eval_queries, 1)
    cand_reduction_pct = 100.0 * (1.0 - (avg_cands_per_s1 / max(total_target_pool_size, 1)))

    recall_at_k_res = {f"Recall@{k}": float(np.mean(rec_at_k[k])) if rec_at_k[k] else 0.0 for k in k_vals}
    prec_at_k_res = {f"Precision@{k}": float(np.mean(prec_at_k[k])) if prec_at_k[k] else 0.0 for k in k_vals}
    hit_at_k_res = {f"Hit@{k}": float(hits_at_k[k] / s1_with_truth) if s1_with_truth > 0 else 0.0 for k in k_vals}

    # ------------------------------------------------------------------
    # 3. Competition Entity-Level Metrics & Stratified Subgroups
    # ------------------------------------------------------------------
    all_f05: list[float] = []
    all_prec: list[float] = []
    all_rec: list[float] = []

    exact_matches = 0
    total_singletons = 0
    correct_singletons = 0
    false_merges_on_singletons = 0

    per_country_f05: dict[str, list[float]] = defaultdict(list)
    per_country_prec: dict[str, list[float]] = defaultdict(list)
    per_country_rec: dict[str, list[float]] = defaultdict(list)

    bucket_f05: dict[str, list[float]] = defaultdict(list)
    bucket_prec: dict[str, list[float]] = defaultdict(list)
    bucket_rec: dict[str, list[float]] = defaultdict(list)

    for s1 in val_s1:
        sid = s1.entity_id
        country = s1.country
        truth = val_truth.get(sid, set())
        pred = val_final_predictions.get(sid, set())

        score = f05_for_entity(truth, pred)
        all_f05.append(score)
        per_country_f05[country].append(score)

        n_true = len(truth)
        bucket = "0 (singleton)" if n_true == 0 else ("1" if n_true == 1 else ("2" if n_true == 2 else "3+"))
        bucket_f05[bucket].append(score)

        if not truth:
            total_singletons += 1
            if not pred:
                correct_singletons += 1
            else:
                false_merges_on_singletons += 1
        else:
            tp_ent = len(truth & pred)
            p_ent = tp_ent / len(pred) if pred else 0.0
            r_ent = tp_ent / len(truth)
            all_prec.append(p_ent)
            all_rec.append(r_ent)
            per_country_prec[country].append(p_ent)
            per_country_rec[country].append(r_ent)
            bucket_prec[bucket].append(p_ent)
            bucket_rec[bucket].append(r_ent)

        if truth == pred:
            exact_matches += 1

    macro_f05 = float(np.mean(all_f05)) if all_f05 else 0.0
    mean_precision = float(np.mean(all_prec)) if all_prec else 0.0
    mean_recall = float(np.mean(all_rec)) if all_rec else 0.0
    singleton_acc = float(correct_singletons / total_singletons) if total_singletons > 0 else 1.0
    exact_match_pct = 100.0 * exact_matches / max(len(val_s1), 1)

    country_breakdown = {}
    for c in sorted(per_country_f05.keys()):
        country_breakdown[c] = {
            "n_entities": len(per_country_f05[c]),
            "macro_f05": float(np.mean(per_country_f05[c])),
            "precision": float(np.mean(per_country_prec[c])) if per_country_prec[c] else 0.0,
            "recall": float(np.mean(per_country_rec[c])) if per_country_rec[c] else 0.0,
        }

    bucket_breakdown = {}
    for b in sorted(bucket_f05.keys()):
        bucket_breakdown[b] = {
            "n_entities": len(bucket_f05[b]),
            "macro_f05": float(np.mean(bucket_f05[b])),
            "precision": float(np.mean(bucket_prec[b])) if bucket_prec[b] else 0.0,
            "recall": float(np.mean(bucket_rec[b])) if bucket_rec[b] else 0.0,
        }

    return {
        # Classification
        "accuracy": pair_acc,
        "precision": pair_prec,
        "recall": pair_rec,
        "specificity": specificity,
        "f1": pair_f1,
        "f05": pair_f05,
        "f2": pair_f2,
        "balanced_accuracy": balanced_acc,
        "mcc": mcc,
        "cohen_kappa": kappa,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "log_loss": lloss,
        "brier_score": brier,
        # Confusion matrix
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        # Retrieval / Ranking
        "candidate_recall": cand_recall,
        "all_target_candidate_recall": all_target_cand_recall,
        "recall_at_k": recall_at_k_res,
        "precision_at_k": prec_at_k_res,
        "hit_at_k": hit_at_k_res,
        "mrr": mrr,
        "map_at_k": map_k,
        "avg_candidates_per_s1": avg_cands_per_s1,
        "candidate_reduction_pct": cand_reduction_pct,
        # Competition
        "macro_f05": macro_f05,
        "mean_precision": mean_precision,
        "mean_recall": mean_recall,
        "n_val_entities": len(val_s1),
        "exact_matches": exact_matches,
        "exact_match_pct": exact_match_pct,
        "total_singletons": total_singletons,
        "correct_singletons": correct_singletons,
        "singleton_accuracy": singleton_acc,
        "false_merges_on_singletons": false_merges_on_singletons,
        "country_breakdown": country_breakdown,
        "bucket_breakdown": bucket_breakdown,
    }
