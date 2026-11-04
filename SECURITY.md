# Security Policy

## Supported versions

Only the latest commit on `main` is supported. There are no maintained release branches yet.

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Report them privately through
[GitHub private vulnerability reporting](https://github.com/leo-aa88/aerodoppler/security/advisories/new).
Include the affected commit, steps to reproduce and the impact you expect. You should get an
acknowledgement within a week.

## What counts as a security issue

AeroDoppler handles private recordings and can produce location estimates, so privacy failures
are treated as security issues. Examples:

- the server reachable from the network in the default configuration, or any authentication
  bypass when it is placed behind a proxy;
- recordings, spectrograms, events or results sent to an external service without the operator
  asking for it, or network access while `AERODOPPLER_OFFLINE=true`;
- path traversal or arbitrary file read/write through the API, uploads or archive extraction;
- purge (`DELETE /api/recordings/{id}`, `aerodoppler audio purge`) leaving the original audio or
  derived data behind;
- the authorization acknowledgement being bypassable;
- unsafe handling of downloaded archives or DEM tiles (for example code execution from crafted
  files).

A wrong location estimate is not a security issue; use a result report for that.
