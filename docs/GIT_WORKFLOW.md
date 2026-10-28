# Git workflow: three people, three permanent branches

```
main            protected: changes arrive only through reviewed pull requests
dev-ayushi      Ayushi's branch    (only Ayushi commits here)
dev-nishitha    Nishitha's branch  (only Nishitha commits here)
dev-abheetha    Abheetha's branch  (only Abheetha commits here)
```

The branches are permanent. You keep using yours for the whole project. Nobody deletes them, and nobody makes extra branches.

Commands below are for Windows `cmd` or PowerShell in the repo folder. Replace `dev-nishitha` with your own branch.

## One-time setup on your computer

```
git clone https://github.com/<owner>/molecular-screening-platform.git
cd molecular-screening-platform
git checkout dev-nishitha
git config user.name "Your Name"
git config user.email "you@example.com"
```

`git branch` should show `* dev-nishitha`. If it shows `main`, you are on the wrong branch.

## Every work session

1. **Start from the latest `main`:**
   ```
   git checkout dev-nishitha
   git fetch origin
   git merge origin/main
   ```
   "Already up to date" is fine. If Git reports a conflict, see "Fixing a conflict" below.
2. **Work, then save your changes:**
   ```
   git status
   git add .
   git commit -m "Add ML worker skeleton"
   ```
   Look at `git status` before `git add .` and make sure no `.env`, `__pycache__` or model files are listed.
3. **Back up to GitHub** (any time, and always before you stop for the day):
   ```
   git push
   ```
   The very first push of a branch needs `git push -u origin dev-nishitha`.

## When a piece of work is ready: pull request

1. `git push`, then open the repo on GitHub. Click **Compare & pull request** (or Pull requests, New pull request, base `main`, compare your branch).
2. Title: say what the change does, at least 15 characters. Example: `Add ML worker and Tox21 signals (FR-010)`.
3. Fill in the template and tick **Unit-test evidence attached**.
4. Wait for the two checks, **PR format** and **CI required**, to go green.
5. Ask one of the other two to review. They open **Files changed**, then **Review changes**, then **Approve**.
6. **The reviewer merges**, using **Create a merge commit**. The author never merges their own PR. Do not click Delete branch.
7. **Everyone syncs after a merge:**
   ```
   git checkout dev-nishitha
   git pull origin main
   ```
   For the author this is a plain fast-forward. For the others it brings down the new work.

You can keep committing on your branch while your PR is open. New commits join the same PR until it is merged.

## Rules the checks enforce

- A PR must come from one of the three personal branches and go into `main`.
- The PR author must be the person who owns that branch (`.github/team.json`).
- The title must describe the change (15 characters or more, not a branch name).
- The unit-test evidence box must be ticked.
- At least one other owner must approve. The check "Author did not merge own PR" turns red if the author merges.

## Fixing a conflict

A conflict happens when two branches changed the same lines.
1. `git merge origin/main` stops and lists the conflicted files.
2. Open each file. Between `<<<<<<<` and `>>>>>>>` choose what to keep, and delete the marker lines.
3. `git add <file>` for each file, then `git commit` (keep the default message).
4. Run the tests, then `git push`.
To give up and start again: `git merge --abort`.

## If something goes wrong

- **I committed on `main` by mistake and have not pushed:**
  ```
  git checkout dev-nishitha
  git merge main
  git checkout main
  git reset --hard origin/main
  git checkout dev-nishitha
  ```
- **I committed on someone else's branch:** do not push. Tell the team in chat.
- **Push rejected, "non-fast-forward":** your branch is behind. Run `git fetch origin`, `git merge origin/main`, then `git push`.
- **Push to `main` rejected:** that is expected. Open a PR instead.
- **PR format check is red:** open the check, read the list. Fix the title or description (the check re-runs by itself) or open the PR from the right branch.
