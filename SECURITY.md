# Security

Report a vulnerability through GitHub's private reporting:
[Security Advisories](https://github.com/p4ik/mlx-beam/security/advisories/new).
Please do not open a public issue for it.

What counts: anything that lets a request reach data or code it should
not - the access control (`--api-key`, the Host allowlist, CORS), the
request parsers, the checkpoint loader (a checkpoint runs no code unless
`--trust-remote-code` is given), the vendored parts.

Expect an answer within a week. A fix ships as a patch release and is
named in the changelog; the report is credited if you want it to be.

Supported: the latest release on PyPI. The engine is 0.x; earlier
releases get no backports.
