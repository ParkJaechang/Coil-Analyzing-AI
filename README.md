# Coil-Analyzing-AI

Experimental AI sweep and AI-assisted modeling infrastructure for Coil-Analyzing.

This repository is backend-only. It does not drive hardware, does not replace the rule-based WebApp modeling workflow, and does not provide Streamlit or WinApp UI code. The current bootstrap provides dataset and sweep utilities only.

WebApp and WinApp integration must be explicit, adapter-based, and user-reviewed. AI suggestions are advisory only.

## Current Scope

- AI sweep target/schema contracts
- Sweep plan generation
- Long sweep LUT concatenation from prebuilt segment commands
- Manifest IO and validation
- Long measurement segmentation by manifest

## Out of Scope

- Hardware invocation
- Real measurement CSV storage
- Upload caches
- Generated outputs
- Model artifacts
- ML/RL training
- Streamlit or PySide6 UI

## Development

```powershell
python -m pip install -e .[test]
python -m pytest -q
```
