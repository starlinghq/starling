# Protected evidence

Reference fixtures, golden outputs, and tolerance files per model profile (plan section 10.1). Changes need maintainer review through CODEOWNERS.

Implementation jobs read these files and never edit them to make a candidate pass. New edge cases are proposed in a separate PR that names the reference source.

Tolerances are set before optimization starts. Release evaluation holdout sets (P-030) do not live here: they stay where implementation jobs have no read or write access, and are rotated under maintainer control.

Files here are byte-exact (`-text` in `.gitattributes`).
