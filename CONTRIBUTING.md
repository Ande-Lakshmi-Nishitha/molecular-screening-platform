# Contributing

Each of us works on one permanent personal branch. We do not make a branch per story.

| Person | Branch | GitHub user |
| --- | --- | --- |
| Ayushi | `dev-ayushi` | see `.github/team.json` |
| Nishitha | `dev-nishitha` | see `.github/team.json` |
| Abheetha | `dev-abheetha` | see `.github/team.json` |

1. Start of every work session: `git checkout dev-<you>`, `git fetch origin`, `git merge origin/main`.
2. Commit small and often, with a plain message: `Add ML worker skeleton`. No Jira key is needed.
3. `git push` to your own branch whenever you want a backup.
4. When a piece of work is ready, open a PR from your branch into `main`. Fill in the template and tick the unit-test evidence box. Naming the requirement IDs (FR-010) or Jira key (MSP-40) in the description is welcome but optional.
5. A different repo owner reviews and merges with **Create a merge commit** once CI is green. Never merge your own PR. Never delete your branch.
6. After it merges, run step 1 again so your branch picks up the merge.
7. Never commit `.env`, secrets, model weights, pickle files or real molecular structures.

Shared files (docker-compose.yml, requirements*.txt, OpenAPI, config, migrations): tell the team in chat first. Only Ayushi creates database migrations.

Full commands, conflict fixes and troubleshooting are in `docs/GIT_WORKFLOW.md`.
