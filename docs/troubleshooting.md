# Troubleshooting

## Windows: sentencepiece build fails during gguf install

When `pip` tries to install `gguf` on Windows it may need to build `sentencepiece` from
source. With the latest `cmake` (3.31+) this fails with an error similar to:

```
Compatibility with CMake < 3.5 has been removed from CMake.
```

This happens because the bundled `sentencepiece` project still advertises
compatibility with very old CMake policy versions. You can resolve the issue in
either of two ways:

1. **Pin CMake to an older release** – before installing the requirements run:
   ```powershell
   pip install "cmake<3.31"
   ```
   and retry the installation.

2. **Install a prebuilt wheel** – install a binary `sentencepiece` wheel that
   matches your Python version (for example `pip install sentencepiece==0.1.99`)
   and then proceed with `pip install -r requirements.txt`. Because the wheel is
   already present `pip` will skip building from source.

Either approach avoids the failing CMake configure step and lets the
installation continue normally.
