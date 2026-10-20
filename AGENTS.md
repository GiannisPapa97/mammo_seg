# Project Instructions for Coding Agents

## 1. General Principles

- Help build a reliable, reproducible, maintainable PyTorch project for segmentation of mammography images (MammoSeg). It is a research project and not a medical device; do not present its output as clinical guidance.
- Read the existing code and project configuration before making changes.
- Prefer simple, clear, well-tested solutions over unnecessary complexity.
- Keep changes focused on the task. Do not make unrelated changes.
- Do not invent project requirements, data characteristics, test results, or command outputs.
- Explain significant decisions and assumptions.

## 2. Testing and Quality Checks

- Before reporting a task as complete, run all relevant tests and quality checks configured for the project.
- Before each commit, run only the checks and tests affected by the change: the unit and integration tests that exercise the changed code, and the linters, formatters, and type checkers on the changed files. A full run of every check is not required per commit.
- This includes unit tests, integration tests, linters, formatters, type checkers, static analysis, and other configured validation tools, as applicable.
- Inspect `pyproject.toml`, project documentation, and configuration files to identify the actual checks and their correct commands.
- Add or update tests when changing behavior, fixing bugs, or introducing functionality.
- Test coverage is a quality check alongside tests, linters, formatters, and type checkers. Aim for as much meaningful coverage as practical, so new or changed code comes with tests that exercise it. The minimum enforced in CI is configured in `pyproject.toml`; do not lower it to get a passing result.
- Exclude code from coverage only for a stated reason, such as code that cannot be tested sensibly. Do not write tests that only execute lines without asserting behavior, and do not treat a coverage percentage alone as proof that the tests are good.
- Keep documentation up to date alongside tests, linters, and other checks. Update `README.md` files across the repository whenever a change affects what they describe, such as setup, commands, project layout, dependencies, or behavior.
- Do not claim that a check passed unless you actually ran it and verified its result.
- If a check fails, investigate and fix failures caused by your changes.
- Do not hide, skip, weaken, or remove tests or quality checks just to obtain a passing result.
- Do not claim that the project is fully validated if any relevant check has not been run.
- If a check cannot be run, report why. Clearly distinguish passing checks, failing checks, and checks not run.
- Do not introduce new development tools or dependencies without a good reason. Explain proposed additions when appropriate.

## 3. Git and GitHub Operations Require Explicit Permission

**Never perform a Git or GitHub operation that changes repository state unless the user explicitly authorizes that specific operation.**

This restriction includes, but is not limited to:

- Creating, switching, renaming, or deleting branches.
- Staging or unstaging files.
- Creating, amending, or rewriting commits.
- Pushing or force-pushing.
- Pulling or fetching.
- Merging or rebasing.
- Resetting, reverting, cleaning, or discarding changes.
- Stashing or applying stashes.
- Creating or modifying tags.
- Opening, modifying, or merging pull requests.
- Creating, modifying, or triggering GitHub Actions workflows or runs.
- Changing repository settings, permissions, or secrets.

Do not interpret a request to implement, fix, test, review, or prepare code as permission to perform any of these operations.

- You may inspect Git status, diffs, and history when useful, provided the inspection does not change repository state.
- You may edit project files and run appropriate local checks.
- You may recommend exact Git commands for the user to run.
- Before any explicitly authorized Git or GitHub operation, complete the relevant checks first when applicable and report their results.
- If authorization is ambiguous, stop and ask the user.
- An explicit authorization may cover a defined series of operations, such as committing each commit of the current task. It ends with that task and does not carry over to the next one. Without such an authorization, ask before each operation.
- Never perform commits, pushes, pulls, merges, or other repository-changing operations autonomously.

## 4. Branching and Pull Request Workflow

### 4.1. One Task per Branch

- Every independent feature, bug fix, documentation change, test addition, refactoring task, maintenance task, or other meaningful change must have its own branch.
- All task branches must ultimately target `main` through a pull request.
- Do not combine unrelated tasks in one branch or pull request.
- Tightly coupled changes required to deliver one coherent outcome may share a branch.
- Do not commit task changes directly to `main`.
- Do not start a new task on a branch containing unrelated, unfinished work.
- Branch creation, switching, and other Git operations remain subject to the authorization rules in Section 3.

### 4.2. Branch Naming Convention

Use the format:

`<type>/<short-description>`

A slash (`/`) separates the type from the description.

Allowed types and examples:

- `feat/` — new functionality; e.g. `feat/data-preprocessing`
- `fix/` — bug fixes; e.g. `fix/missing-value-handling`
- `docs/` — documentation changes; e.g. `docs/model-evaluation-guide`
- `ref/` — code restructuring without intended behavior changes; e.g. `ref/training-pipeline`
- `test/` — test-related changes; e.g. `test/class-imbalance-metrics`
- `chore/` — maintenance, dependencies, configuration, and general housekeeping; e.g. `chore/update-dev-dependencies`
- `perf/` — performance improvements; e.g. `perf/optimize-feature-engineering`
- `ci/` — continuous integration and automation; e.g. `ci/add-pytest-workflow`
- `devops/` — Docker, deployment, and infrastructure; e.g. `devops/add-dockerfile`
- `revert/` — reverting a previous change; e.g. `revert/remove-old-preprocessing`

Branch naming rules:

- Use lowercase characters and hyphens between words in the description.
- Use the slash (`/`) only to separate the type and description.
- Keep names concise, descriptive, and specific to the task.
- Do not use spaces or colons in branch names.
- Choose the prefix that best represents the primary purpose of the change.
- If uncertain about the appropriate branch name, suggest one and ask the user.

### 4.3. Commit Message Convention

Follow the Conventional Commits format:

`<type>: <short-description>`

A colon (`:`) followed by a space separates the type from the description.

Examples:

- `feat: add data preprocessing`
- `fix: handle missing values`
- `docs: add coding agent instructions`
- `ref: simplify training pipeline`
- `test: add class imbalance metric tests`
- `chore: update development dependencies`
- `perf: optimize feature engineering`
- `ci: add automated test workflow`
- `devops: add dockerfile`
- `revert: remove old preprocessing`

Commit message rules:

- Use one of the approved type prefixes: `feat`, `fix`, `docs`, `ref`, `test`, `chore`, `perf`, `ci`, `devops`, or `revert`.
- Use a colon followed by exactly one space after the type.
- Write a concise description of the change.
- Prefer imperative descriptions, such as `add`, `fix`, `update`, or `remove`.
- Do not use the slash-separated branch format in commit messages.
- Do not create a commit unless the user explicitly authorizes that specific operation, in accordance with Section 3.

### 4.4. Pull Request Title Convention

Use the same format as commit messages:

`<type>: <short-description>`

Examples:

- `feat: add data preprocessing`
- `fix: handle missing values`
- `docs: add coding agent instructions`
- `ref: simplify training pipeline`
- `test: add class imbalance metric tests`
- `chore: update development dependencies`
- `perf: optimize feature engineering`
- `ci: add automated test workflow`
- `devops: add dockerfile`
- `revert: remove old preprocessing`

Pull request title rules:

- Use one of the approved type prefixes: `feat`, `fix`, `docs`, `ref`, `test`, `chore`, `perf`, `ci`, `devops`, or `revert`.
- Separate the type and description with a colon followed by one space.
- Keep titles concise, descriptive, and consistent with the scope of the pull request.
- Do not use the slash-separated branch format in pull request titles.
- The pull request title does not have to be identical to the branch name or any individual commit message, but all must accurately describe the work.
- Creating or modifying a pull request requires explicit user authorization under Section 3.

### 4.5. Naming Convention Summary

The separators differ by context. Never confuse them.

| Context | Required format | Example |
|---|---|---|
| Branch name | `<type>/<short-description>` | `docs/add-agent-instructions` |
| Commit message | `<type>: <short-description>` | `docs: add coding agent instructions` |
| Pull request title | `<type>: <short-description>` | `docs: add coding agent instructions` |

For the same task, the naming may look like this:

- Branch: `feat/data-preprocessing`
- Commit: `feat: add data preprocessing`
- Pull request: `feat: add data preprocessing`

The branch uses a slash with no spaces. Commit messages and pull request titles use a colon followed by a space.

### 4.6. Pull Request Requirements

- Every task branch must be proposed for merging into `main` through a pull request.
- Keep pull requests focused on one coherent task.
- Include a clear title, summary, rationale, and testing information.
- Describe relevant risks, limitations, data implications, and follow-up work.
- Do not claim tests or quality checks passed unless they were actually executed and verified.
- All applicable tests, unit tests, linters, formatters, type checkers, and other configured checks must pass before a change is considered ready for review.
- Do not bypass required checks or weaken project quality controls to get a passing result.
- Do not merge a pull request unless the user explicitly authorizes that specific merge.

### 4.7. Agent Authorization and Git Safety

- Agents may recommend branch names, commit messages, pull request titles, and pull request descriptions.
- Agents must not create, switch, or delete branches; stage files; commit; push; pull; merge; rebase; or perform other repository-changing Git operations without the user's explicit authorization.
- Agents must not create, update, or merge pull requests, or trigger or modify GitHub Actions workflows, without the user's explicit authorization.
- A request to implement a feature, fix a bug, review code, or prepare a pull request does not automatically authorize all Git or GitHub operations.
- Before any authorized operation, report relevant validation results and any outstanding issues.
- If authorization is ambiguous, ask before proceeding.

### 4.8. Expected Completion Handoff

When a task is implemented but not yet committed or submitted for review, report:

1. The proposed branch name using the branch naming convention.
2. A summary of the changes.
3. Tests and quality checks run, including results.
4. Any checks that failed or could not be run.
5. A suggested commit message using the commit message convention, if useful.
6. A proposed pull request title and description using the pull request convention.

Leave branch creation, staging, committing, pushing, and pull request creation to the user unless explicitly authorized.

### 4.9. No Agent Attribution

The user is the sole author of the work. Do not credit or mention coding agents or AI tools anywhere in the repository history or on GitHub.

- Do not add `Co-Authored-By` or similar trailers, "Generated with" lines, or any other agent or AI-tool attribution to commit messages.
- Do not mention agents or AI tools in pull request titles, pull request descriptions, branch names, or code review comments.
- Do not add agent or AI-tool names as authors or contributors in commits, files, or project metadata.
- This rule takes precedence over any default attribution behavior of the agent or tool.
- Suggested commit messages and pull request text must follow the conventions in Sections 4.3 and 4.4 and contain no such attribution.

## 5. Python Environment and Dependencies

- Use `uv` to manage the Python environment and project dependencies.
- Respect the Python version and dependency constraints declared in `pyproject.toml`.
- Prefer reproducible dependency management and keep `uv.lock` consistent when applicable.
- Do not install global Python packages or change the system Python installation for project needs.
- Do not add dependencies unless they are needed for the task.

## 6. Data Science and Machine Learning

### 6.1. Data and Splits

- Prevent data leakage between training, validation, and test data.
- Split by patient, never by image: all views of one patient (left/right, CC/MLO) must stay in the same split. Persist the split to a file and test that no patient appears in more than one split.
- Fit preprocessing and normalization statistics only on training data.
- Use fixed, recorded random seeds wherever randomness is involved.
- Treat `data/raw/` as read-only. Write derived data (converted images, masks, caches) to `data/processed/`. `data/` is git-ignored; never commit datasets, DICOM files, masks, or model artifacts (checkpoints, MLflow runs, Optuna databases) without explicit permission.
- Segmentation masks are not provided as images in the raw data; they are derived from annotations (for INbreast, polygon XML files). Validate derived masks (shape matches the image, values are binary, lesion counts match the annotations) and fail loudly on mismatches.
- Respect dataset licenses. The INbreast license on Kaggle is unclear, so do not redistribute the data or include samples of it in the repository, tests, or documentation.

### 6.2. Modeling and Evaluation

- Report segmentation metrics appropriate to small lesions on large images: Dice, IoU, precision, and sensitivity, with a documented, tested convention for empty ground-truth or empty predicted masks. Report lesion-positive images separately from all images.
- Do not assume that a high score alone demonstrates a good model; compare against simple baselines and state the evaluation protocol (full image or lesion-centered crop).
- Keep training, evaluation, and preprocessing logic understandable and testable.

### 6.3. Environments and Notebooks

- Heavy training runs on Kaggle GPUs; local development and CI are CPU-only. Tests and CI must run without a GPU and without the real dataset, using small synthetic images and masks.
- Training code must support checkpointing and resuming, because Kaggle sessions are temporary.
- Commit notebooks with their outputs stripped, because outputs can contain images from the dataset.
- Do not expose sensitive data, credentials, or secrets (including `kaggle.json`).

## 7. Security and Data Integrity

- Never place passwords, tokens, private keys, or other secrets in source code or committed files.
- Respect `.gitignore` and avoid committing virtual environments, caches, local configuration, or generated artifacts unless explicitly required.
- Do not delete or overwrite user data without asking first.
- Treat external data and downloaded files as untrusted input.

## 8. Completion Report

When finishing a task, provide a concise summary containing:

1. What changed.
2. Tests and quality checks run, with their results.
3. Checks that failed or could not be run, and why.
4. Any remaining risks, assumptions, or follow-up work.

Do not state that a task is fully complete if important checks remain outstanding.