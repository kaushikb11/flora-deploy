"""flora-deploy: run NVIDIA Flora labeling models at scale on Modal."""

__version__ = "0.0.1"

# Human label for the framework logic, hashed into the config hash (PLAN.md D11). Bump it to mark
# an intentional output change. It is not the safety net: the hash also covers a digest of the
# source of every output-path module (``config/logic.py``), so a fix there changes the hash even
# if nobody bumps this.
LOGIC_VERSION = "0.0.1"
