# Architecture Decision Records (ADRs)

Each ADR captures one genuinely convention-dependent design choice — the
kind a reviewer would otherwise have to reverse-engineer from the code —
together with the alternatives considered and the consequences of the
choice. The code remains the source of truth; an ADR documents *why* the
code is the way it is.

Format follows the SciTeX-ecosystem ADR style (numbered `NNNN-slug.md`,
sections: Status / Context / Decision / Considered and rejected /
Consequences / Implementation / References).

| ADR | Title |
| --- | ----- |
| [0001](0001-true-negative-for-alarm-based-seizure-warning.md) | True negative for an alarm-based seizure-warning system — SOP-length interictal opportunities |

The conceptual companion to ADR-0001 (with the SPH/SOP framework and a
worked example) is
[`docs/math/alarm_confusion_matrix.md`](../math/alarm_confusion_matrix.md).
