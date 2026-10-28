# Model artifacts
Only ONNX, TorchScript and native XGBoost files (never pickle, SEC-008). Weight files are git-ignored; each file's SHA-256 is registered in the `models` table. Mounted read-only into workers.
