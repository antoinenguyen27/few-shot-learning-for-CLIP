# Git Workflow For The Team

Use one branch per method or task.

## Create A Branch

Start from `main`:

```bash
git checkout main
git pull
git checkout -b method/<your-method-name>
```

Examples:

```bash
git checkout -b method/promptsrc
git checkout -b method/lp-plus-plus
git checkout -b method/dpc
git checkout -b method/promptkd
```

## Work In Your Folder

Edit only your method directory unless you have coordinated a common change.

```text
Promptsrc/
LP++/
DPC/
promptkd/
```

Use `common/` through imports; do not copy shared data-loading code into your folder.

## Check What Changed

```bash
git status
git diff
```

## Commit

```bash
git add <files-you-changed>
git commit -m "Implement <method-name> scaffold"
```

## Push

```bash
git push -u origin <your-branch-name>
```

## Avoid

- Do not commit raw data, caches, checkpoints, or result dumps.
- Do not use notebooks.
- Do not edit another person's method directory without asking.
- Do not change split JSON files for your own method.
- Do not merge large paper repos wholesale into this repo.
