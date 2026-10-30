# AI CAD Tutor

[Back to MIT Maker Portfolio](../../README.md)

AI CAD Tutor is a local AI learning application for SOLIDWORKS and CSWA education, developed by [0oKevino0](https://github.com/0oKevino0). The application is in use; this project directory contains a separate source snapshot prepared for external technical review.

## What the application does

- Answers course-grounded questions with cited source material.
- Offers Hint 1, Hint 2, Guided Steps, and Full Explanation.
- Separates the current lesson step from the level of help requested: asking for an explanation does not automatically advance the student.
- Preserves the lesson position when switching modes.
- Uses supported lesson records for guided procedures and does not invent a step-by-step procedure for an unsupported topic.

The Shell walkthrough illustrates these behaviors. Lesson progress follows the student's requests; it does not directly verify their actions inside SOLIDWORKS. Included interfaces, adapters, or stubs do not establish that every planned CAD, vision, or grading feature is implemented.

## Start reading here

[Browse source](source/) · [React interface](source/apps/web/src/App.tsx) · [Course API](source/services/api/app/routes/course.py) · [Teaching modes](source/services/api/app/services/teaching_modes.py)

| Area | Files |
| --- | --- |
| Frontend and interaction state | `source/apps/web/src/App.tsx`, `api.ts`, `components/` |
| API entry and configuration | `source/services/api/app/main.py`, `launcher.py`, `config/settings.py` |
| Course request routing | `source/services/api/app/routes/course.py` |
| Teaching modes | `source/services/api/app/services/teaching_modes.py` |
| Course responses and citations | `source/services/api/app/services/course_demo.py`, `source/services/api/app/services/course.py`, `source/services/api/app/course_citations.py` |
| Follow-up handling | `source/services/api/app/services/demo_conversation.py`, `assistant_interaction.py` |
| Retrieval | `source/services/rag/retrieval/`, `source/model_mvp/rag/` |
| Local model integration | `source/services/inference/backends/v1b_demo_backend.py`, `source/services/inference/v1b_demo_sidecar.py` |
| Shared contracts | `source/services/api/app/models/course.py`, `source/packages/contracts/` |

Some implementation names contain `demo`; those names are preserved from the original source, rather than renamed for this review copy.

## Architecture and tools

The reviewed course workflow uses a React/TypeScript frontend, a Python/FastAPI/Pydantic API, custom lexical retrieval, and local inference with Qwen/Qwen3-8B and a V1B LoRA adapter through PyTorch, Hugging Face Transformers, and PEFT. Other included backend interfaces are not a claim that every backend is active in the shown workflow.

## Contents and runtime requirements

This snapshot includes 88 application source and dependency/configuration files. It preserves their relative directory structure under `source/`. It is not a complete deployment bundle.

The following runtime inputs are intentionally excluded:

- Base model weights, tokenizer cache, LoRA adapter files, and adapter configuration.
- Course documents, the concept corpus, Shell lesson records, and associated permission/source records.
- Private training or evaluation data, environment files, credentials, logs, and internal working history.

The original dependency declarations are retained in `source/pyproject.toml` and `source/apps/web/package.json`; the frontend lockfile is also included. PyTorch, Transformers, and PEFT are additional sidecar requirements, not pinned by the root Python project metadata. Available notices and dependency metadata are included separately.

To inspect the frontend independently, use its existing `typecheck` and `build` scripts after installing the declared packages in `source/apps/web/`. For the complete course workflow, separately authorized runtime inputs and matching model/adapter digests must first be supplied. Preserve the `source/` layout and use `source/` as the Python working directory; the original Python packaging configuration does not package every corpus/schema resource. The placeholder-only `CONFIGURATION_TEMPLATE.txt` documents configuration names, not working credentials or a complete launch recipe. No model files are downloaded by this review package.

The original test scripts remain in dependency metadata, but this copy does not include the full test suite or private fixtures. No end-to-end deployment is claimed for this public snapshot. The working application's video is provided separately in the portfolio.

## Authorship and assistance

The project owner leads the design and development. Claude assisted with code review and portions of the code, including frontend code. Codex assisted with version control and review. This work is not represented as entirely handcrafted without AI assistance.

## Snapshot and licensing

Prepared on October 29, 2026, from the previously reviewed local application source. File hashes are listed in the [portfolio review manifest](../../REVIEW_MANIFEST.json). The source files preserve the reviewed candidate bytes; the inference sidecar's introductory usage example had previously been sanitized to remove internal paths. Its runtime implementation was not changed for this copy.

The original project metadata declares `Proprietary`. Public availability for inspection does not add an open-source license. Existing third-party notices are retained under `licenses/`; see `LICENSE_STATUS.txt` and `DEPENDENCIES_AND_LICENSES.json`. No dependency implementation, model weights, or course content is bundled.

Paths in the dependency and license metadata are relative to this project directory. The original root Python project metadata is retained at `source/pyproject.toml`.
