# Archetype letter mapping (old A/B/C -> published A'/B'/C')

The letters `A, B, C, ...` attached to archetypes by the per-subclass PCHA fits
(`archetype_1, archetype_2, ...`) are **arbitrary** — they are vertex order from the fit,
not a biological ordering. Figures publish **primed** letters instead:

> **A' is the most superficial archetype; primed letters run in laminar-depth order
> (A' -> B' -> C' -> D') within each subclass.**

The relabel is **display-only**. Score/marker/loading TSVs keep the old letters, so e.g.
mouse L2/3 `score_A` is the column shown as `C'`.

## Where the depth order comes from

All IT cells are pooled into one embedding, archetype centroids are taken, and the
centroids are sorted by **angle around the arc** (a 1-D gradient renders as a horseshoe in
2-D, so ordering along the curve — not the 2-D spread — is the depth order). Within each
subclass the arc rank `j` becomes `ALPHABET[j] + "'"`.

| species | derivation script | persisted map |
|---|---|---|
| mouse (cheng22, 11 archetypes) | `scripts/it_evo/15.mouse_it_joint_pca_archetype_map.py` (`ARC_ORDER_CURATED`) | `local_data/res/it_evo/15.mouse_IT_joint_archetype_arc_order.tsv` |
| human (jorstad23, 13 archetypes) | `scripts/it_evo/22.human_it_joint_pca_archetype_map.py` (`ARC_ORDER_CURATED`) | `local_data/res/it_evo/22.human_IT_joint_archetype_arc_order.tsv` |

Both orders are **curated**, not purely computed: 15 overrules the angular sort on the L6IT
segment (L6IT B/C are 5.6 degrees apart, inside the noise; L6IT A sits in the arc interior on
249 cells), keeping both ranks on record as `arc_rank` vs `arc_rank_angular`. 22 runs in two
passes — pass 1 writes `curated=False` (no consumer may use it), pass 2 sets
`ARC_ORDER_CURATED` after inspecting the figures.

## The mapping

### Mouse — `15.mouse_IT_joint_archetype_arc_order.tsv`

| subclass | old (05 fit) | new (published) | note |
|---|---|---|---|
| L2/3 | C, B, A | A', B', C' | reversed |
| L4   | C, B, A | A', B', C' | reversed |
| L5IT | B, A    | A', B'     | reversed |
| L6IT | A, B, C | A', B', C' | identity |

As a per-key dict:

```
L23_C -> A'   L23_B -> B'   L23_A -> C'
L4_C  -> A'   L4_B  -> B'   L4_A  -> C'
L5IT_B-> A'   L5IT_A-> B'
L6IT_A-> A'   L6IT_B-> B'   L6IT_C-> C'
```

### Human — `22.human_IT_joint_archetype_arc_order.tsv`

| subclass | old (04 fit) | new (published) | note |
|---|---|---|---|
| L2/3 IT | D, C, B, A | A', B', C', D' | reversed |
| L4 IT   | C, B, A    | A', B', C'     | reversed |
| L5 IT   | D, C, B, A | A', B', C', D' | reversed |
| L6 IT   | A, B       | A', B'         | identity |

## Color convention

Color follows the **displayed** (primed) label, never the internal key:
`A' -> C0, B' -> C1, C' -> C2, D' -> C3`.

## How consumers apply it

**`scripts/it_evo`** — reads the persisted TSVs, does not hard-code:

```python
def relabel_for(arc, token):
    """{old_letter: new_letter} for one subclass from a 15/22-style depth-arc table."""
    sub = arc[arc['token'] == token]
    return dict(zip(sub['old_letter'], sub['new_letter']))
```

Used by `12`, `14`, `18c`, `18d`, `32`.

**`scripts/it`** — only L2/3 is published, so the L2/3 row of the table above is written out
literally:

```python
ARCH_RELABEL = {'A': "C'", 'B': "B'", 'C': "A'"}   # internal (= archetype_1/2/3) -> displayed
ARCH_ORDER   = ['C', 'B', 'A']                     # so columns read A', B', C'
ARCH_COLORS  = {'A': 'C2', 'B': 'C1', 'C': 'C0'}   # color follows the displayed label
```

In `41` (per-layer `arch_relabel`, L2/3 entry only), `42`, `48`, `48.v2`, `49`, `50`;
mirrored in `it_evo/24b`, `it_evo/31`.

**`scripts/l23_evo`** — the human L2/3 4-archetype fit, relabeled as a plain full reversal
described in-script as "mirrors the mouse L2/3 reversal":

```python
ARCH_RELABEL = {'archetype_1': "D'", 'archetype_2': "C'", 'archetype_3': "B'", 'archetype_4': "A'"}
```

In `28`, `30`, `35`, `36`, `37`, `38`; `26.viz.spearman_heatmap.py` does the same by reversing
the human columns. This agrees with 22's derived L2/3 order, but is stated as a convention
there rather than read from the depth record.

**Caveat:** mouse archetypes in `l23_evo` are mostly left **unprimed** (`21.viz`, `53.viz` use
`ARCHETYPE_NAMES = ['A', 'B', 'C']`), so `l23_evo` mouse letters do not match `scripts/it`
mouse letters — `it/48`'s `A'` is `l23_evo`'s `C`.
