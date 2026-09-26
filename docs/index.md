# Documentation Index

Start with the [README](../README.md) for the project overview and the [installation guide](../installation.md) for setup. Usage guides describe current entry points; design documents label proposed features and evaluation targets separately.

## Usage and technical guides

| Document | Contents |
| --- | --- |
| [Installation](../installation.md) | Main environment, optional interfaces, SAM checkpoints, and CellViT worker setup |
| [Platform overview](../platform-overview.md) | Implemented components, algorithms, assumptions, and limitations |
| [SAM guide](../sam-guide.md) | Meta SAM automatic masks, parameters, and filters |
| [SAM testing](../sam-testing.md) | Functional checks and workload-specific measurements |
| [CellViT guide](../cellvit-guide.md) | Subprocess architecture, preprocessing, and postprocessing |
| [CellViT testing](../cellvit-testing.md) | Integration and output-validation protocol |
| [Validation record](validation.md) | Tested environment, completed checks, and known test failures |
| [Contributing](../CONTRIBUTING.md) | Development workflow, validation, and contribution guidelines |
| [Development notes](../CLAUDE.md) | Environment conventions, entry points, and diagnostic scripts |

## Architecture and planning

| Document | Contents | Status |
| --- | --- | --- |
| [Project overview](project-overview.md) | Motivation, intended users, and feature areas | Current scope and development goals |
| [Architecture](architecture.md) | Module layout, execution flow, and possible service architecture | Implemented components and labeled design options |
| [Roadmap](roadmap.md) | Twelve proposed phases, priorities, and illustrative milestones | Planning framework, not a release commitment |
| [Implementation plan](implementation-plan.md) | Data, training, evaluation, deployment, and quality practices | Proposed practices and measurable targets |

The main interface uses Streamlit, with an optional Gradio interface and a minimal Streamlit interface. A packaged CLI, separate FastAPI/React service, Qt desktop application, distributed tiling system, and hosted deployment remain proposed extensions.

Performance and accuracy targets in the planning documents require defined datasets, hardware, protocols, and evidence before they can be reported as results.

## Project contact

Use the [repository](https://github.com/XZhang0726/cell-segment-platform) and [issue tracker](https://github.com/XZhang0726/cell-segment-platform/issues) for source code, questions, and reproducible bug reports. See the README's [attribution and licensing section](../README.md#attribution-and-licensing) for reuse information.
