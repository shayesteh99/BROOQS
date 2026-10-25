# BROOQS

**BROOQS** infers a **level-1 phylogenetic network** from a **tree of blobs** and a
collection of **gene trees**.

A tree of blobs captures *where* reticulation happened (each multifurcating node,
or "blob", marks a region of the species history that is not tree-like) but not
*how*. BROOQS resolves every blob into a level-1 cycle: for each blob it builds a
quartet-based signal matrix (the *F* matrix) from the gene trees, computes a
spectral cyclic ordering of the blob's components, picks the hybrid, and stitches
the cycles back together. The result is a phylogenetic network in
[eNewick](http://dx.doi.org/10.1186/1471-2105-9-532) format.

## Installation

BROOQS is a small Python tool. Install it locally from a clone of this
repository:

```bash
git clone https://github.com/shayesteh99/BROOQS.git
cd BROOQS
pip install .
```

This puts a `brooqs` command on your `PATH`. Requires Python ≥ 3.9;
`numpy` and `treeswift` are installed automatically.

## Usage

```bash
brooqs -i <gene_trees> -t <tree_of_blobs> [options]
```

### Input

| Flag | Required | Description |
|------|----------|-------------|
| `-i`, `--input` | yes | Gene trees, **one Newick tree per line**. Trees may have missing taxa. Internal node labels are added automatically if absent. |
| `-t`, `--tob` | yes | The tree of blobs, a **single Newick tree on the first line**. Every node with more than two children is treated as a blob to be resolved. |

Leaf labels in the gene trees must match the leaf labels in the tree of blobs.

### Output

| Flag | Default | Description |
|------|---------|-------------|
| `-o`, `--output` | `<input>_network.enewick` | Path for the resulting eNewick network. The network string is also printed to stdout. |

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--rooted` | off | Trust the rooting of the tree of blobs: never reroot it. If a blob's inferred hybrid turns out to be the "outside" group, the next-best hybrid is used instead. Recommended when your tree of blobs has a known outgroup. |
| `--min-blob-size` `N` | 4 | Leave blobs with `N` or fewer children unresolved (kept as polytomies). The effective minimum is 4 — smaller blobs cannot form a cycle. By default, BROOQS resolves all polytomies of size>3. |
| `--dist-var` | off | For 4-blobs, identify the hybrid from the **variance of pairwise gene-tree distances** rather than random assignment. Requires `--rooted`. |
| `--dist-var-k` `K` | 5 | Number of random representative-leaf draws used for the `--dist-var` majority vote. |
| `--filter` | off | Use only gene trees that contain **all** components of a blob (instead of down-weighting incomplete ones). |
| `--order-mode` `{spectral,pq_tree}` | `spectral` | How the cyclic order of each blob is computed: the spectral embedding of the *F* matrix, or PQ-tree circular seriation (`circular_seriation.py`). Default is `spectral` (which is shown to be more accurate than `pq_tree`). |
| `-s`, `--seed` `S` | 1142 | Random seed (affects `--dist-var`). |

## Example

The [`Example/`](Example) directory contains a simulated 15-taxon dataset with one
reticulation:

| File | What it is |
|------|------------|
| `true_net.nwk` | the true phylogenetic network (eNewick) — the answer |
| `true_tob.nwk` | the true tree of blobs (one blob of 7 children) — BROOQS input |
| `true_genetrees.tree` | 1000 true gene trees, one per line — BROOQS input |

Run BROOQS on it:

```bash
cd Example
brooqs -i true_genetrees.tree -t true_tob.nwk -o brooqs_network.enewick
```

The tree of blobs has a single blob with seven children:

```
((t1,t9),(t4,t14),t5,(t6,t8),((t7,t11),t13),t10,t15)
```

BROOQS resolves that blob into a cycle and writes:

```
(OUT,(((((t1,t9),((t4,t14),(t5,((t6,t8))#H0))),(((t7,t11),t13),(t10,(t15,#H0)))),(t2,(t3,t12)))));
```

Here BROOQS places the hybrid on the `(t6,t8)` clade (`#H0`), with one parent edge
next to `t5` and the other next to `t15`. That matches the true network
(`true_net.nwk`), where the clade `(t6,t8)` (labelled `#H31`) is the hybrid, with
parents adjacent to `t5` and to `t15`.

You can visualize/compare the eNewick networks with tools such as
[PhyloNetworks.jl](https://github.com/JuliaPhylo/PhyloNetworks.jl),
[IcyTree](https://icytree.org/), or [Dendroscope](https://uni-tuebingen.de/en/199936).

## Citation

If you use BROOQS in your work, please cite this repository.
