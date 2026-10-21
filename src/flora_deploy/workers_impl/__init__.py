"""Modal-agnostic cores of the shard workers (``workers.ShardWorker``).

Each module here holds the undecorated logic of one worker kind; ``modal_app/workers_app.py``
wraps it in a module-scope Modal class that the app factory decorates.
"""
