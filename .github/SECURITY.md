# Security policy

Report vulnerabilities privately through GitHub's private vulnerability reporting on this repository (Security tab, "Report a vulnerability"). Do not open a public issue.

Include the affected package and version, the model bundle identity if relevant, and a minimal reproducer. We aim to acknowledge reports within 5 working days. This is an internal operating target, not a commercial SLA.

Areas of particular interest: native interop and memory ownership, model bundle extraction and loading (path traversal, digest bypass), and anything that causes a network call during normal inference.
