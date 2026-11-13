"""Cross-process lock for persistent export identity."""
from contextlib import contextmanager
from pathlib import Path
import fcntl

@contextmanager
def artifact_lock(target):
    target=Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    # Never unlink this lock: another process may already be waiting on its inode.
    with target.with_name('.'+target.name+'.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(),fcntl.LOCK_UN)
