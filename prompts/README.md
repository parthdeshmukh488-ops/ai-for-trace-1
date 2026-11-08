# The build transcript

Eleven files, one per phase. Each is written to be pasted into a terminal
coding agent as-is, at the repository root, in order.

## How to use them

1. Open a session at the repository root.
2. Paste the contents of `phase-00.md`.
3. When the agent reports the gate is green, verify it yourself by running the
   gate command.
4. Commit. Then paste the next phase.

Do not paste two phases at once. The gates exist because each layer is the
oracle for the next one, and an agent working on phases 3 and 4 together will
produce an attribution measurement whose error cannot be separated from a bug
in the thing it is measuring against.

## Before the first paste

The agent should read `AGENT_BRIEF.md` first — every phase file says so in its
first line, but it is worth knowing that is where the hard rules live. The two
that matter most and are easiest to lose track of over a long build:

* no number in any file that was not measured by a run that produced a
  `results/*.json`;
* when a measurement disagrees with the expected answer, it goes in the
  write-up rather than getting fixed.

## If a phase goes sideways

Each file ends with a **Report** section listing what to say on completion. If
the agent cannot reach the gate, the useful response is that report with the
blocker named, not a partial implementation presented as finished. Specifying
this up front is most of what keeps a long build honest.

## Local convenience

The brief is tracked as `AGENT_BRIEF.md` so it stays out of tool-specific
naming. If your agent auto-loads a context file from the repository root, copy
it to whatever name that tool expects and leave the copy untracked:

```bash
cp AGENT_BRIEF.md CONTEXT.md && echo "CONTEXT.md" >> .git/info/exclude
```
