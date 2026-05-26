## Development Workflow
- **Git Commit Routine**: After EVERY successfully completed code task or sub-task, you MUST make a Git commit. Do not wait for the user to remind you.
- **Commit Message Style**: You must follow the **Conventional Commits** format.
  - Format: `<type>(<scope>): <short description in english>`
  - Allowed types: feat, fix, docs, style, refactor, test, chore
  - Example: `feat(api): implement user registration`

## Verification Before Commit
- Before running any commit, you must run the project's test command to ensure nothing is broken.
- If tests fail, DO NOT commit. Fix the issue first.
