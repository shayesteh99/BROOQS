"""BROOQS: infer a level-1 phylogenetic network from a tree of blobs.

BROOQS takes a tree of blobs together with a collection of gene trees and
resolves every multifurcating blob into a level-1 cycle, using a quartet-based
signal (an F matrix) and a spectral ordering of the blob's components. The
result is a phylogenetic network in eNewick format.

Run ``brooqs --help`` for the full list of options.
"""

import sys
import os
import argparse
import time
import numpy as np
from treeswift import *
import math
import random
import warnings
warnings.filterwarnings("ignore")

def is_number(x):
    try:
        float(x)
        return True
    except (ValueError, TypeError):
        return False

def __label_tree__(tree_obj, index = 0):
  is_labeled = True
  labels = set()
  for node in tree_obj.traverse_preorder():
    # node.edge_length = 1
    if node.is_leaf():
      continue
    if not node.label or node.label in labels or is_number(node.label):
      is_labeled = False
      node.label = 'I' + str(index)
      index += 1
    labels.add(node.label)
  return is_labeled, index

def remove_random_taxa(trees, annotations, groups, rate =  0.1, mode = "keep"):
    avg_missing = 0
    num_genes = {i:len(trees) for i in range(len(groups))}
    for t in trees:
        total = 0
        tree_groups = [[] for i in range(len(groups))]
        for l in t.traverse_leaves():
            tree_groups[annotations[l.label]].append(l.label)
        for l in t.traverse_leaves():
            p = random.random()
            if p < rate:
                a = annotations[l.label]
                if len(tree_groups[a]) == 1 and mode == "keep":
                    continue
                total += 1
                parent = l.parent
                parent.remove_child(l)
                parent.contract()
                # print(tree_groups[a], l.label)
                tree_groups[a].remove(l.label)
                if len(tree_groups[a]) == 0:
                    num_genes[a] -= 1
                if t.num_nodes(leaves=True, internal=False) <= 4:
                    break
        # print(total)
        avg_missing += total
    print(num_genes)
    print("average missing: ", avg_missing/len(trees))
    return trees

def choose_2(n):
    return n * (n-1)/2

def preprocess(tree, annotations):
    weights = {}
    for l in annotations:
        a = annotations[l]
        if a in weights:
            weights[a] += 1
        else:
            weights[a] = 1

    lcm = math.lcm(*[weights[a] for a in weights])

    k = len(weights)

    num_leaves = {n:[0 for i in weights] for n in tree.traverse_postorder()}
    total_leaves = {n:0 for n in tree.traverse_postorder()}
    same_pair = {n:0 for n in tree.traverse_postorder()}
    same_pair_outside = {n:0 for n in tree.traverse_postorder()}
    for n in tree.traverse_postorder():
        if n.is_leaf():
            a = annotations[n.label]
            num_leaves[n][a] += lcm/weights[a]
            total_leaves[n] += lcm/weights[a]
            # num_leaves[n][a] += 1
            # total_leaves[n] += 1
            # print(n.label, num_leaves[n])
            continue
        for c in n.child_nodes():
            total_leaves[n] += total_leaves[c]
            for i in range(k):
                num_leaves[n][i] += num_leaves[c][i]

    root = tree.root
    for n in tree.traverse_postorder():
        for i in range(k):
            same_pair[n] += choose_2(num_leaves[n][i])
            same_pair_outside[n] += choose_2(num_leaves[root][i] - num_leaves[n][i])

        # print(n.label, num_leaves[n])


    return weights, num_leaves, total_leaves, same_pair, same_pair_outside, lcm

def preprocess_new(tree, annotations, gene_weights):
    weights = {}
    for l in tree.traverse_leaves():
        a = annotations[l.label]
        if a in weights:
            weights[a] += 1
        else:
            weights[a] = 1

    # lcm = math.lcm(*[weights[a] for a in weights])
    groups = set([annotations[a] for a in annotations])
    k = len(groups)
    # k = len(weights)

    num_leaves = {n:[0 for i in groups] for n in tree.traverse_postorder()}
    total_leaves = {n:0 for n in tree.traverse_postorder()}
    total_diff = {n:0 for n in tree.traverse_postorder()}
    total_diff_out = {n:0 for n in tree.traverse_postorder()}
    diff_pair = {n:[0 for i in groups] for n in tree.traverse_postorder()}
    diff_pair_out = {n:[0 for i in groups] for n in tree.traverse_postorder()}

    for n in tree.traverse_postorder():
        if n.is_leaf():
            a = annotations[n.label]
            num_leaves[n][a] += 1/(weights[a] * gene_weights[a])
            total_leaves[n] += 1/(weights[a] * gene_weights[a])
            # num_leaves[n][a] += 1/(weights[a])
            # total_leaves[n] += 1/(weights[a])
            continue
        for c in n.child_nodes():
            total_leaves[n] += total_leaves[c]
            for i in range(k):
                num_leaves[n][i] += num_leaves[c][i]


    root = tree.root
    for n in tree.traverse_postorder():
        for i in range(k):
            diff_pair[n][i] += num_leaves[n][i] * (total_leaves[n] - num_leaves[n][i])
            total_diff[n] += diff_pair[n][i] / 2

            diff_pair_out[n][i] += (num_leaves[root][i] - num_leaves[n][i]) * (total_leaves[root] - num_leaves[root][i] - (total_leaves[n] - num_leaves[n][i]))
            total_diff_out[n] += diff_pair_out[n][i] / 2

    return weights, num_leaves, total_leaves, diff_pair, total_diff, diff_pair_out, total_diff_out


def compute_pairwise_F(tree, i, j, num_leaves, total_leaves, same_pair, same_pair_outside):
    total = 0

    root = tree.root
    for n in tree.traverse_postorder():
        if n.is_leaf():
            continue
        out_total = total_leaves[root] - total_leaves[n]
        out_ij = num_leaves[root][i] + num_leaves[root][j] - num_leaves[n][i] - num_leaves[n][j]
        out_ii = choose_2(num_leaves[root][i] - num_leaves[n][i])
        out_jj = choose_2(num_leaves[root][j] - num_leaves[n][j])

        for c1 in n.child_nodes():
            for c2 in n.child_nodes():
                if c1 == c2:
                    continue
                total += num_leaves[c1][i] * num_leaves[c2][j] * (choose_2(out_total - out_ij) - (same_pair_outside[n] - out_ii - out_jj))
                total += num_leaves[c1][i] * (choose_2(total_leaves[c2] - num_leaves[c2][i] - num_leaves[c2][j]) - (same_pair[c2] - choose_2(num_leaves[c2][i]) - choose_2(num_leaves[c2][j]))) * (num_leaves[root][j] - num_leaves[n][j])
                total += num_leaves[c1][j] * (choose_2(total_leaves[c2] - num_leaves[c2][i] - num_leaves[c2][j]) - (same_pair[c2] - choose_2(num_leaves[c2][i]) - choose_2(num_leaves[c2][j]))) * (num_leaves[root][i] - num_leaves[n][i])


    # for n in tree.traverse_preorder():
    #     print(n.label, diff_pairs[n])
    return total


def compute_pairwise_F_new(tree, i, j, num_leaves, total_leaves, diff_pair, total_diff, diff_pair_out, total_diff_out):
    root = tree.root
    ij_pair = {n:0 for n in tree.traverse_postorder()}
    for n in tree.traverse_postorder():
        if n.is_leaf():
            continue
        ij_pair[n] = num_leaves[n][i] * num_leaves[n][j]

    total = 0
    for n in tree.traverse_postorder():
        if n.is_leaf():
            continue
        
        out_i = num_leaves[root][i] - num_leaves[n][i]
        out_j = num_leaves[root][j] - num_leaves[n][j]
        not_ij_pairs_out = total_diff_out[n] - diff_pair_out[n][i] - diff_pair_out[n][j] + (out_i * out_j)

        for c1 in n.child_nodes():
            for c2 in n.child_nodes():
                if c1 == c2:
                    continue
                
                total += num_leaves[c1][i] * num_leaves[c2][j] * not_ij_pairs_out
                total += num_leaves[c1][i] * (total_diff[c2] - diff_pair[c2][i] - diff_pair[c2][j] + ij_pair[c2]) * (num_leaves[root][j] - num_leaves[n][j])
                total += num_leaves[c1][j] * (total_diff[c2] - diff_pair[c2][i] - diff_pair[c2][j] + ij_pair[c2]) * (num_leaves[root][i] - num_leaves[n][i])

    # for n in tree.traverse_preorder():
    #     print(n.label, diff_pairs[n])
    return total

def compute_pairwise_F_polytomy(tree, i, j, num_leaves, total_leaves, diff_pair, total_diff, diff_pair_out, total_diff_out):
    root = tree.root
    ij_pair = {n:0 for n in tree.traverse_postorder()}
    for n in tree.traverse_postorder():
        if n.is_leaf():
            continue
        ij_pair[n] = num_leaves[n][i] * num_leaves[n][j]

    total = 0
    for n in tree.traverse_postorder():
        if n.is_leaf():
            continue

        out_i = num_leaves[root][i] - num_leaves[n][i]
        out_j = num_leaves[root][j] - num_leaves[n][j]
        not_ij_pairs_out = total_diff_out[n] - diff_pair_out[n][i] - diff_pair_out[n][j] + (out_i * out_j)

        for c1 in n.child_nodes():
            total += num_leaves[c1][i] * not_ij_pairs_out * (num_leaves[n][j] - num_leaves[c1][j])
            total += (total_diff[c1] - diff_pair[c1][i] - diff_pair[c1][j] + ij_pair[c1]) * out_i * (num_leaves[n][j] - num_leaves[c1][j])
            for c2 in n.child_nodes():
                if c1 == c2:
                    continue
                total += num_leaves[c1][i] * (total_diff[c2] - diff_pair[c2][i] - diff_pair[c2][j] + ij_pair[c2]) * (num_leaves[root][j] - num_leaves[c1][j] - num_leaves[c2][j])
    return total


def compute_F_matrix(tree, annotations, gene_weights):
    # weights, num_leaves, total_leaves, same_pair, same_pair_outside, lcm = preprocess(tree, annotations)
    weights, num_leaves, total_leaves, diff_pair, total_diff, diff_pair_out, total_diff_out = preprocess_new(tree, annotations, gene_weights)

    groups = set([annotations[a] for a in annotations])
    k = len(groups)
    present = len(weights)
    missing = k - present

    F = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            # print(i,j)
            # f = compute_pairwise_F(tree, i, j, num_leaves, total_leaves, same_pair, same_pair_outside)/lcm**4
            f = compute_pairwise_F_polytomy(tree, i, j, num_leaves, total_leaves, diff_pair, total_diff, diff_pair_out, total_diff_out)
            F[i, j] = f #* gene_weights[i] * gene_weights[j]
            F[j, i] = f #* gene_weights[i] * gene_weights[j]

    return F

def spectral_cycle_order(F, similarity='direct', eps=1e-12, alpha=0.2):
    """Return a cyclic ordering of the m = F.shape[0] components.
    Parameters
    ----------
    F : (m, m) ndarray
        Pairwise F matrix. Symmetric, zero diagonal. Larger F should mean
        "more adjacent on the cycle" (this matches your Facts 1, 3, 4).
    similarity : {'direct', 'exp', 'inv'}
        How to turn F into a similarity matrix S used for the Laplacian:
          - 'direct': S = F                          (simplest)
          - 'exp'   : S_ij = exp(F_ij / median(F))   (sharpens contrast)
          - 'inv'   : S_ij = 1 / (1 + max(F) - F_ij) (bounded)
        Diagonal of S is always set to 0.
    eps : float
        Tolerance for degenerate angles.
    angle_threshold : float
        Two consecutive components are considered "too close" in angle if
        the cyclic gap between their angles is below this value. In that
        case we use Fact 1 in-betweenness on F to decide their order.
    Returns
    -------
    order : list of int
        A permutation of 0..m-1 giving the inferred cyclic ordering.
        Rotation and reflection are not fixed — compare cycles up to those.
    """
    m = F.shape[0]
    F = np.asarray(F, dtype=float)
    # Build similarity matrix S (off-diagonal only)
    if similarity == 'direct':
        S = F.copy()
    elif similarity == 'exp':
        med = np.median(F[np.triu_indices(m, k=1)])
        med = max(med, 1e-9)
        S = np.exp(F / med)
    elif similarity == 'inv':
        mx = F.max()
        S = 1.0 / (1.0 + mx - F)
    else:
        raise ValueError(f"unknown similarity '{similarity}'")
    np.fill_diagonal(S, 0.0)
    # Symmetrize to guard against tiny asymmetries
    S = 0.5 * (S + S.T)
    # Graph Laplacian L = D - S
    deg = S.sum(axis=1)
    L = np.diag(deg) - S
    # Eigendecomposition of L (real symmetric)
    eigvals, eigvecs = np.linalg.eigh(L)
    # eigvals ascending. eigvals[0] ~ 0 (constant eigenvector).
    # For circular structure, v_2 and v_3 (next two) span the cyclic embedding.
    if m < 3:
        return list(range(m))
    v2 = eigvecs[:, 1]
    v3 = eigvecs[:, 2]
    # lengths = []
    # for i in range(m):
    #     lengths.append(np.sqrt((v2[i]-v2[(i+1) % m])**2 + (v3[i]-v3[(i+1) % m])**2))
    # print(lengths)
    # Sort components by angle in the (v2, v3) plane
    angles = np.arctan2(v3, v2)
    print(angles)
    order = list(np.argsort(angles))
    # print(angles)
    print(order)

    # --- F-based swap of consecutive ambiguous pairs ---
    # For each consecutive pair (a, b) in the cyclic `order` whose angular
    # gap is below `angle_threshold`, look at the left neighbor L (before a)
    # and right neighbor R (after b) in the cyclic order. If the current
    # arrangement is L, a, b, R but Fact 1's in-betweenness on F prefers
    # L, b, a, R, swap a and b in `order`.
    #
    # Concretely: in L, a, b, R, Fact 1 says F[L, b] should be smaller than
    # both F[L, a] and F[a, b] (a is between L and b), and F[a, R] should
    # be smaller than both F[a, b] and F[b, R] (b is between a and R).
    # We compute a simple "in-betweenness violation count" for the current
    # order and for the swapped order, and swap if the swapped order has
    # fewer violations.
    def _violations(left, mid1, mid2, right):
        # Returns number of Fact-1 in-betweenness inequalities violated
        # for the linear sub-path (left, mid1, mid2, right).
        v = 0
        # Triple (left, mid1, mid2): F[left, mid2] should be smaller than
        # both F[left, mid1] and F[mid1, mid2].
        if left:
            if not (F[left, mid2] < F[left, mid1]):
                v += 1
            if not (F[left, mid2] < F[mid1, mid2]):
                v += 1
        # Triple (mid1, mid2, right): F[mid1, right] should be smaller than
        # both F[mid1, mid2] and F[mid2, right].
        if right:
            if not (F[mid1, right] < F[mid1, mid2]):
                v += 1
            if not (F[mid1, right] < F[mid2, right]):
                v += 1
        return v

    for i in range(m):
        a_idx = i
        b_idx = (i + 1) % m
        a, b = order[a_idx], order[b_idx]
        gap = abs(angles[a] - angles[b])
        # print(gap, alpha * 2 * np.pi / m)
        gap = min(gap, 2 * np.pi - gap)  # cyclic
        if gap < alpha * 2 * np.pi / m:
        # if False:
            # print(a,b)
            left = order[(a_idx - 1) % m]
            right = order[(b_idx + 1) % m]

            # lgap = abs(angles[left] - angles[a])
            # lgap = min(lgap, 2 * np.pi - lgap)

            # if lgap < alpha * 2 * np.pi / m:
            #     left = None

            # rgap = abs(angles[b] - angles[right])
            # rgap = min(rgap, 2 * np.pi - rgap)

            # if rgap < alpha * 2 * np.pi / m:
            #     right = None
            # Don't swap if any of the four positions coincide (small m edge case)
            if len({left, a, b, right}) == 4:
                # print(left, a, b, right)
                cur_v = _violations(left, a, b, right)
                swp_v = _violations(left, b, a, right)
                # print(cur_v, swp_v)
                if swp_v < cur_v:
                    order[a_idx], order[b_idx] = order[b_idx], order[a_idx]
                    # print(order)
    # --- end F-based swap ---

    min_score = float('Inf')
    hybrid = None
    hubrid_ind = None

    scores = [(F[order[i], order[(i+2)%m]], (i+1)%m) for i in range(len(order))]
    scores.sort(key=lambda t: t[0])
    print(scores)
    hybrid_ind = scores[0][1]

    old_order = list(order)
    order = order[hybrid_ind:] + order[:hybrid_ind]
    print(order)

    # alt_hybrid_idx is the GROUP INDEX (not position) of the 2nd-best hybrid.
    alt_hybrid_idx = old_order[scores[1][1]] if len(scores) > 1 else None
    alt_score = ((scores[1][0] - scores[0][0]) / scores[0][0]
                 if len(scores) > 1 and scores[0][0] != 0 else float('inf'))

    return order, alt_hybrid_idx, alt_score, angles


def compute_gene_weights(trees, groups, annotations, alpha=1):
    gene_weights = {i:len(trees) for i in range(len(groups))}

    # pairwise_weights = np.ones((len(groups), len(groups)))
    # pairwise_weights *= len(trees)

    for t in trees:
        group_set = set()
        # p_weights = np.zeros((len(groups), len(groups)))
        for l in t.traverse_leaves():
            group_set.add(annotations[l.label])
        for i in range(len(groups)):
            if i not in group_set:
                gene_weights[i] -= 1
                # p_weights[i,:] = 1
                # p_weights[:,i] = 1
        # print(group_set)
        # print(p_weights)
        # pairwise_weights -= p_weights
        # return
    gene_weights = {i:gene_weights[i]/len(trees) for i in gene_weights}
    return gene_weights


def filter_missing(trees, groups, annotations, alpha=1):
    # gene_weights = {i:len(trees) for i in range(len(groups))}

    # pairwise_weights = np.ones((len(groups), len(groups)))
    # pairwise_weights *= len(trees)
    filtered_genetrees = []
    for t in trees:
        group_set = set()
        # p_weights = np.zeros((len(groups), len(groups)))
        for l in t.traverse_leaves():
            group_set.add(annotations[l.label])
        if len(group_set) == len(groups):
            filtered_genetrees.append(t)
    return filtered_genetrees


def identify_hybrid_by_dist_var(children_leaf_sets, gene_trees, k=5):
    """Identify the hybrid child of a 3-child polytomy using pairwise distance variance.

    Repeats k times: picks one leaf at random from each child group, computes
    pairwise distances across all gene trees, and evaluates
    np.var(np.log(distances)) for each of the three pairs.  The pair with the
    smallest variance corresponds to the two non-hybrid children; the remaining
    child is the hybrid.  The child identified as hybrid in the most rounds wins.

    Returns the index (0, 1, or 2) into children_leaf_sets of the hybrid child.
    """
    # Draw all k representative triplets before touching gene trees.
    all_reps = [
        [random.choice(sorted(ls)) for ls in children_leaf_sets]
        for _ in range(k)
    ]
    print(f"[dist-var] {k} triplets:")
    for r, reps in enumerate(all_reps):
        print(f"[dist-var]   round {r + 1}: {reps}")

    needed = set(leaf for reps in all_reps for leaf in reps)

    # Accumulate distances keyed by (round, pair).
    dists = {(r, i, j): []
             for r in range(k)
             for i in range(3)
             for j in range(i + 1, 3)}

    for gt in gene_trees:
        gt_leaves = set(l.label for l in gt.traverse_leaves())
        present = needed & gt_leaves
        if len(present) < 2:
            continue

        # Clone and prune to only the needed leaves so the distance matrix is small.
        pruned = gt.extract_tree_with(present)
        # pruned 
        # for leaf in list(pruned.traverse_leaves()):
        #     if leaf.label not in present:
        #         parent = leaf.parent
        #         parent.remove_child(leaf)
        #         parent.contract()

        dm = pruned.distance_matrix(leaf_labels=True)

        # Distribute distances to all k rounds in one pass over the small matrix.
        for r, reps in enumerate(all_reps):
            for i in range(3):
                for j in range(i + 1, 3):
                    t1, t2 = reps[i], reps[j]
                    if t1 in dm and t2 in dm[t1] and dm[t1][t2] > 0:
                        dists[(r, i, j)].append(dm[t1][t2])

    # Score each round and collect votes.
    vote_counts = [0, 0, 0]
    for r, reps in enumerate(all_reps):
        pair_vars = {}
        for i in range(3):
            for j in range(i + 1, 3):
                d_list = dists[(r, i, j)]
                pair_vars[(i, j)] = (
                    np.var(np.log(d_list)) if len(d_list) > 1 else float('inf')
                )
                print(f"[dist-var] round {r + 1}/{k}  pair ({i},{j}) "
                      f"({reps[i]}, {reps[j]}): "
                      f"n={len(d_list)}  var(log)={pair_vars[(i, j)]:.6g}")

        min_pair = min(pair_vars, key=pair_vars.get)
        hybrid_idx = (set(range(3)) - set(min_pair)).pop()
        vote_counts[hybrid_idx] += 1
        print(f"[dist-var] round {r + 1} → hybrid child {hybrid_idx} "
              f"(taxa={sorted(children_leaf_sets[hybrid_idx])})")

    winner = int(np.argmax(vote_counts))
    print(f"[dist-var] votes: {vote_counts}  →  hybrid child {winner} "
          f"(taxa={sorted(children_leaf_sets[winner])})")
    return winner


def build_network_newick(tob, trees, rate = 0, mode = "free", contract = False, rooted = False, filter = False, min_blob_size = 4, dist_var = False, dist_var_k = 5):
    """Convert a tree of blobs to a phylogenetic network in eNewick format.

    Three phases, tob is only modified between phases 1 and 3:
      1. Traverse every polytomy (tob unchanged) and compute F + spectral order.
      2. Apply any needed reroots to tob (outside == hybrid case).
      3. Build the eNewick string using the stored orders.
    """
    all_leaves = set(l.label for l in tob.traverse_leaves())
    hybrid_counter = [0]

    # ── Phase 1: compute orders for every polytomy (tob unmodified) ──────────
    node_info = {}   # id(node) → info dict

    for node in tob.traverse_postorder():
        if node.is_leaf():
            continue
        children = list(node.child_nodes())
        m = len(children)
        if node.is_root() and m < min_blob_size:
            continue
        if m < min_blob_size-1:
            continue

        node_leaves = set(l.label for l in node.traverse_leaves())
        groups      = [[l.label for l in c.traverse_leaves()] for c in children]
        outside     = [l for l in all_leaves if l not in node_leaves]
        is_root_poly = len(outside) == 0

        if not is_root_poly:
            groups      = groups + [outside]
            outside_idx = m
        else:
            outside_idx = None
        k = len(groups)
        print(groups)

        annotations = {t: i for i, g in enumerate(groups) for t in g}

        if rate == 0:
            pruned_trees = trees
        else:
            pruned_trees = remove_random_taxa(
                [read_tree_newick(t.newick()) for t in trees],
                annotations, groups, rate=rate, mode=mode)

        if filter:
            pruned_trees = filter_missing(trees, groups, annotations)
            gene_weights = {i:1 for i in range(len(groups))}
            print("Number of genes after filtering: ", len(pruned_trees))

        else:
            gene_weights = compute_gene_weights(pruned_trees, groups, annotations)
            # print(gene_weights)

        F = np.zeros((k, k))
        for t in pruned_trees:
            F += compute_F_matrix(t, annotations, gene_weights)
        F /= len(pruned_trees)
        # F /= pairwise_weights

        #normalize F
        # target = (k-3)*(k-1)*(k-2)/6
        # row_sums = F.sum(axis=1)
        # scale = np.sqrt(target / row_sums) 
        # F *= scale[:, None] * scale[None, :]
        # for i in range(k):
        #     row_sum = sum(F[i,])
        #     F[i,] *= (k-3)*(k-1)*(k-2)/(6*row_sum)

        # for i in range(k):
        #     for j in range(i + 1, k):
        #         mean = (F[i,j] + F[j,i])/2
        #         F[i,j] = mean
        #         F[j,i] = mean

        print(F)
        print(sum(F))

        order, alt_hybrid_idx, alt_score, angles = spectral_cycle_order(F)
        h_idx = order[0]
        print(h_idx, order)

        # dist-var override: for 3-child non-root blobs, use pairwise distance
        # variance to identify the hybrid instead of spectral scoring alone.
        if dist_var and m == 3 and not is_root_poly:
            children_leaf_sets = [frozenset(groups[i]) for i in range(m)]
            dv_h = identify_hybrid_by_dist_var(children_leaf_sets, trees, k=dist_var_k)
            if dv_h != h_idx:
                print(f"[dist-var] hybrid override: spectral={h_idx} → dist-var={dv_h}")
            pos = order.index(dv_h)
            order = order[pos:] + order[:pos]
            h_idx = dv_h

        node_info[id(node)] = {
            'order':          order,
            'h_idx':          h_idx,
            'outside_idx':    outside_idx,
            'groups':         groups,
            'm':              m,
            'alt_hybrid_idx': alt_hybrid_idx,
            'alt_score':      alt_score,
            'is_root_poly':   is_root_poly,
            'F':              F,
            'angles':         angles,
        }

    # ── Phase 2: resolve outside == hybrid conflicts ──────────────────────────

    def _log_hybrid_swap(info, alt_h, reason):
        orig_h  = info['h_idx']
        groups  = info['groups']
        orig_taxa = sorted(groups[orig_h])
        alt_taxa  = sorted(groups[alt_h])
        print(f"[hybrid swap] reason       : {reason}")
        print(f"[hybrid swap] original hybrid  (group {orig_h}): {orig_taxa}")
        print(f"[hybrid swap] alternative hybrid (group {alt_h}): {alt_taxa}")
        print(f"[hybrid swap] alt_score         : {info['alt_score']:.6g}")

    if rooted:
        # Tob is correctly rooted — never reroot. For any polytomy whose
        # primary hybrid is the outside group, switch to the alternative.
        for nid, info in node_info.items():
            if info['is_root_poly']:
                continue
            if info['h_idx'] == info['outside_idx']:
                alt_h = info.get('alt_hybrid_idx')
                if alt_h is None:
                    raise ValueError(
                        "A polytomy has hybrid == outside but no alternative "
                        "hybrid is available. Cannot resolve without rerooting."
                    )
                _log_hybrid_swap(info, alt_h, "hybrid==outside and --rooted prevents rerooting")
                curr_order = info['order']
                alt_pos = curr_order.index(alt_h)
                info['order'] = curr_order[alt_pos:] + curr_order[:alt_pos]
                info['h_idx'] = alt_h
    else:
        def _hybrid_leaves():
            s = set()
            for info in node_info.values():
                s.update(info['groups'][info['h_idx']])
            return s

        def _find_reroot_leaf(hl):
            for lf in tob.traverse_leaves():
                if lf.label not in hl:
                    return lf
            return None

        # When no safe leaf exists, swap the polytomy with the smallest
        # alt_score to its alt_hybrid_idx and retry.
        swapped = set()
        while True:
            hl = _hybrid_leaves()
            reroot_leaf = _find_reroot_leaf(hl)
            if reroot_leaf is not None:
                break

            best_nid, best_score = None, float('inf')
            for nid, info in node_info.items():
                if nid in swapped or info.get('alt_hybrid_idx') is None:
                    continue
                if info['alt_score'] < best_score:
                    best_score = info['alt_score']
                    best_nid = nid

            if best_nid is None:
                raise ValueError(
                    "Cannot find a safe reroot point even after trying all "
                    "alternative hybrid assignments."
                )

            info  = node_info[best_nid]
            alt_h = info['alt_hybrid_idx']
            _log_hybrid_swap(info, alt_h, "no safe reroot leaf found; swapping alt hybrid")
            curr_order = info['order']
            alt_pos = curr_order.index(alt_h)
            info['order'] = curr_order[alt_pos:] + curr_order[:alt_pos]
            info['h_idx'] = alt_h
            swapped.add(best_nid)

        needs_reroot = any(
            not info['is_root_poly'] and info['h_idx'] == info['outside_idx']
            for info in node_info.values()
        )

        if needs_reroot:
            r_label = reroot_leaf.label
            reroot_leaf.edge_length = 1
            tob.reroot(reroot_leaf, length=0.5)
            for c in tob.root.child_nodes():
                c.edge_length = None

            for info in node_info.values():
                for j, g in enumerate(info['groups']):
                    if r_label in g:
                        info['outside_idx'] = j
                        info['is_root_poly'] = False
                        break

    # ── Phase 3: build newick using stored orders (no F recomputation) ────────
    def subtree_to_newick(node):
        if node.is_leaf():
            return node.label
        children      = list(node.child_nodes())
        child_newicks = [subtree_to_newick(c) for c in children]
        if len(children) <= 2:
            return f"({','.join(child_newicks)}){node.label or ''}"
        return _build_polytomy(node, children, child_newicks)

    def _contract_arm(arm, anchor_left, anchor_right, F, angles):
        """Iteratively merge adjacent arm elements that fail betweenness.

        For each middle element C2 with neighbours C1 (left) and C3 (right),
        if F(C1,C3) >= F(C2,C1) or F(C1,C3) >= F(C2,C3) the betweenness
        criterion fails.  Merge the angularly-closest pair (C1,C2) or (C2,C3),
        preferring the arm-interior pair when the closest neighbour is an anchor.

        Returns a list of lists of original group indices.
        Each inner list is a 'merged position' — length-1 = singleton (unchanged),
        length > 1 = polytomy at that chain position.
        """
        if not arm:
            return []
        eff = [[int(g)] for g in arm]

        def F_mg(G1, G2):
            return max(F[i, j] for i in G1 for j in G2)

        def mean_angle(G):
            return float(np.mean([angles[i] for i in G]))

        def angle_diff(a, b):
            d = abs(a - b)
            return min(d, 2 * np.pi - d)

        changed = True
        while changed:
            changed = False
            n = len(eff)
            for pos in range(n):
                C2 = eff[pos]
                C1 = eff[pos - 1] if pos > 0 else [int(anchor_left)]
                C3 = eff[pos + 1] if pos < n - 1 else [int(anchor_right)]

                f13 = F_mg(C1, C3)
                if f13 < F_mg(C2, C1) and f13 < F_mg(C2, C3):
                    continue  # betweenness satisfied

                # Violation: merge the closest pair by angle.
                d12 = angle_diff(mean_angle(C1), mean_angle(C2))
                d23 = angle_diff(mean_angle(C2), mean_angle(C3))

                if d12 <= d23:
                    if pos > 0:               # merge C1 (arm element) with C2
                        eff[pos - 1] = eff[pos - 1] + C2
                        eff.pop(pos)
                    elif n > 1:               # C1 is anchor; merge C2 with C3
                        eff[pos] = C2 + eff[pos + 1]
                        eff.pop(pos + 1)
                else:
                    if pos < n - 1:           # merge C2 with C3 (arm element)
                        eff[pos] = C2 + eff[pos + 1]
                        eff.pop(pos + 1)
                    elif pos > 0:             # C3 is anchor; merge C1 with C2
                        eff[pos - 1] = eff[pos - 1] + C2
                        eff.pop(pos)

                changed = True
                break

        return eff

    def _build_polytomy(node, children, child_newicks):
        info = node_info.get(id(node))
        if info is None:
            return f"({','.join(child_newicks)})"

        order        = info['order']
        h_idx        = info['h_idx']
        outside_idx  = info['outside_idx']
        groups       = info['groups']
        is_root_poly = info['is_root_poly']
        F_mat        = info['F']
        angles       = info['angles']

        # Match each current child to its stored group index by leaf set.
        child_strs = {}
        for child, nwk in zip(children, child_newicks):
            child_leaves = frozenset(l.label for l in child.traverse_leaves())
            for gi, g in enumerate(groups):
                if frozenset(g) == child_leaves:
                    child_strs[gi] = nwk
                    break

        h_num  = hybrid_counter[0]
        hybrid_counter[0] += 1
        h_full = f"({child_strs[h_idx]})#H{h_num}"
        h_ref  = f"#H{h_num}"

        def build_chain(arm, end_str):
            """arm: list of ints or lists-of-ints (merged positions)."""
            result = end_str
            for elem in reversed(arm):
                if isinstance(elem, list):
                    result = f"({','.join(child_strs[o] for o in elem)},{result})"
                else:
                    result = f"({child_strs[elem]},{result})"
            return result

        if is_root_poly:
            k    = len(groups)
            rest = [order[i] for i in range(1, k)]
            if contract and rest:
                rest = _contract_arm(rest, int(h_idx), int(h_idx), F_mat, angles)
            return f"({h_full},{build_chain(rest, h_ref)})"

        p         = list(order).index(outside_idx)
        left_arm  = [order[i] for i in range(1, p)]
        right_arm = [order[i] for i in range(p + 1, len(order))]

        if contract:
            left_arm  = _contract_arm(left_arm,  int(h_idx),       int(outside_idx), F_mat, angles)
            right_arm = _contract_arm(right_arm, int(outside_idx), int(h_idx),        F_mat, angles)

        left_str  = build_chain(list(reversed(left_arm)), h_full)
        right_str = build_chain(right_arm, h_ref)
        return f"({left_str},{right_str})"

    return subtree_to_newick(tob.root) + ";"


def resolve_tob(tob, trees):
    all_leaves = [l.label for l in tob.traverse_leaves()]
    for n in tob.traverse_preorder():
        if n.is_root() and len(n.child_nodes()) <= 3:
            continue
        if len(n.child_nodes()) > 2:
            node_leaves = [l.label for l in n.traverse_leaves()]
            groups = []
            for c in n.child_nodes():
                groups.append([l.label for l in c.traverse_leaves()])
            groups.append([l for l in all_leaves if l not in node_leaves])
            print(groups)
            k = len(groups)
            
            annotations = {}
            for i in range(len(groups)):
                for t in groups[i]:
                    annotations[t] = i

            F = np.zeros((k, k))
            for t in trees:
                F += compute_F_matrix(t, annotations)
            F /= len(trees)

            print(F)
            spectral_cycle_order(F)








def main():
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('-i', '--input', required=True, help="Input Trees")
  parser.add_argument('-t', '--tob', required=True, help="Tree of Blob")
  parser.add_argument('-r', '--rate', required=False, default = '0', help="Rate of random pruning")
  parser.add_argument('-m', '--mode', required=False, default = "keep", choices=["keep", "free"], help="Filter gene trees with missing components")
  parser.add_argument('-o', '--output', required=False, default=None, help="Output eNewick file (default: <input>_network.enewick)")
  parser.add_argument('-s', '--seed', required=False, default=1142, help="Random Seed")
  parser.add_argument('--filter', action='store_true', default=False,
                      help="Filter gene tree with missing components (default: off)")
  parser.add_argument('--contract', action='store_true', default=False,
                      help="Contract angularly-close cycle components into polytomies when betweenness fails (default: off)")
  parser.add_argument('--rooted', action='store_true', default=False,
                      help="Treat the tob as correctly rooted: never reroot; if hybrid==outside for any blob, use the alternative hybrid instead (default: off)")
  parser.add_argument('--dist-var', action='store_true', default=False,
                      help="For size-4 blobs (3 children), identify the hybrid via pairwise distance variance instead of spectral scoring. Requires --rooted. (default: off)")
  parser.add_argument('--dist-var-k', type=int, default=5, metavar='K',
                      help="Number of random-leaf repetitions for --dist-var majority vote (default: 5)")
  parser.add_argument('--min-blob-size', type=int, default=2, metavar='N',
                      help="Skip resolving blobs with <= N children; leave them as polytomies (default: 2)")


  start = time.time()
  args = parser.parse_args()
  random.seed(a=int(args.seed))
  np.random.seed(int(args.seed))


  with open(args.input, "r") as f:
    lines = f.readlines()
    input_trees = [read_tree_newick(t.split()[0]) for t in lines]
    # input_trees = remove_random_taxa(input_trees)
    for t in input_trees:
        __label_tree__(t)

  with open(args.tob, "r") as f:
    tob = read_tree_newick(f.readlines()[0])

  rate = float(args.rate)
  mode = args.mode
  contract = args.contract
  rooted = args.rooted
  dist_var = args.dist_var
  if dist_var and not rooted:
      parser.error("--dist-var requires --rooted")
  dist_var_k = args.dist_var_k
  min_blob_size = max(4,int(args.min_blob_size))
  network_newick = build_network_newick(tob, input_trees, rate = rate, mode = mode, contract = contract, rooted = rooted, filter = args.filter, min_blob_size = min_blob_size, dist_var = dist_var, dist_var_k = dist_var_k)
  print(network_newick)

  end = time.time()

  out_path = args.output if args.output else os.path.splitext(args.input)[0] + "_network.enewick"
  with open(out_path, "w") as f:
      f.write(network_newick + "\n")
  print(f"Network written to {out_path}")

  print("runtime: ", end-start)




if __name__ == "__main__":
  main()  