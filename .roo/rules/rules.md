### 🔄 Project Awareness & Context

- **Always read the memory bank files** at the start of a new conversation to understand the project's architecture, goals, style, and constraints.
- **Maintain consistent naming conventions, file structure, and architecture patterns** as described in the memory bank.

### 🧱 Code Structure & Modularity

- **Never create a file longer than 500 lines of code.** If a file approaches this limit, refactor by splitting it into modules or helper files.
- **Organize code into clearly separated modules**, grouped by feature or responsibility.
- **Use clear, consistent imports** (prefer relative imports within packages).

### 🧪 Testing & Reliability

- **Always create Pytest unit tests for new features** (functions, classes, routes, etc).
- **After updating any logic**, check whether existing unit tests need to be updated. If so, do it.
- **Tests should live in a `/tests` folder** mirroring the main app structure.
  - Include at least:
    - 1 test for expected use
    - 1 edge case
    - 1 failure case

### ✅ Task Completion

- **Document completed tasks in the memory bank's `progress.md`** immediately after finishing them.
- Add new sub-tasks or TODOs discovered during development to the `activeContext.md` file in the memory bank.

### 📎 Style & Conventions

- **Use Python** as the primary language with appropriate type hints.
- **Follow PEP8** and format with `black` and `isort` for consistent imports.
- **Use the project's validation system** in `utils/validation/` for input validation.
- **Use Gradio** for UI components with consistent styling from the `BaseUI` class.
- **Implement proper error handling** using the centralized error system in `utils/error_handling/`.
- Write **docstrings for every function** using the Google style:

  ```python
  def example():
      """
      Brief summary.

      Args:
          param1 (type): Description.

      Returns:
          type: Description.
      """
  ```

### 📚 Documentation & Explainability

- **Update `README.md`** when new features are added, dependencies change, or setup steps are modified.
- **Comment non-obvious code** and ensure everything is understandable to a mid-level developer.
- When writing complex logic, **add an inline `# Reason:` comment** explaining the why, not just the what.

### 🧠 AI Behavior Rules

- **Never assume missing context. Ask questions if uncertain.**
- **Never hallucinate libraries or functions** – only use known, verified Python packages.
- **Always confirm file paths and module names** exist before referencing them in code or tests.
- **Never delete or overwrite existing code** unless explicitly instructed to or if part of a documented task in the memory bank.

### 📚 References & API Documentation

- **BFL API Scalar**: https://api.us1.bfl.ai/scalar - API documentation for BFL services
- **Gradio Interface**: https://www.gradio.app/docs/gradio/interface - Documentation for Gradio Interface components and usage
