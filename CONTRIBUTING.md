# Contributing to AI Agent Memory Router

## Commit Message Guidelines

This project follows [Conventional Commits](https://www.conventionalcommits.org/) for commit messages. This enables automatic semantic versioning and changelog generation.

### Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code (white-space, formatting, missing semi-colons, etc)
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance
- **test**: Adding missing tests or correcting existing tests
- **chore**: Changes to the build process or auxiliary tools and libraries such as documentation generation

### Scopes

- **api**: API changes
- **core**: Core functionality changes
- **mcp**: MCP server changes
- **docker**: Docker configuration changes
- **ci**: CI/CD changes

### Examples

```
feat(api): add memory routing endpoint
fix(core): resolve memory retrieval bug
docs: update API documentation
style: fix code formatting
refactor(mcp): simplify tool registration
perf(core): optimize memory search algorithm
test(api): add integration tests for memory endpoints
chore: update dependencies
```

### How to Commit

1. Stage your changes: `git add .`
2. Commit without message: `git commit`
3. Vim will open with the commit template
4. Fill in the template following the format above
5. Save and exit vim

### Semantic Versioning

- **Major version** (1.0.0): Breaking changes
- **Minor version** (1.1.0): New features (backward compatible)
- **Patch version** (1.1.1): Bug fixes (backward compatible)

The semantic-release tool automatically determines version bumps based on your commit messages.

## Development Workflow

1. Create a feature branch from `main`
2. Make your changes following the commit guidelines
3. Push your branch and create a pull request
4. After review and merge, semantic-release will automatically create a new release

## Questions?

If you have questions about contributing, please open an issue or reach out to the maintainers.
