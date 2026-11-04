# CLAUDE.md

Project instructions for Claude Code. The shared guidance for all coding agents is in AGENTS.md,
imported below; put new shared rules there rather than here.

@AGENTS.md

## Claude-specific notes

- Verify before claiming: run `make lint` and the relevant tests, and quote their real output.
- Long jobs (`make test`, benchmarks, `make demo`) are better run in the background and polled.
