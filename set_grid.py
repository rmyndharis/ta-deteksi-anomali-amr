"""Set grid interval eksperimen secara konsisten.

Pemakaian:  python set_grid.py 60   (atau 30)
Mengubah dua nilai sekaligus agar tidak pernah tidak sinkron:
- config.py               : INTERVAL_MINUTES
- build_training_population.py : POPULATION_INTERVAL_MINUTES
"""
import pathlib
import re
import sys

n = sys.argv[1] if len(sys.argv) > 1 else ""
assert n in ("30", "60"), "pemakaian: python set_grid.py 30|60"

here = pathlib.Path(__file__).resolve().parent
targets = [
    ("config.py", "INTERVAL_MINUTES"),
    ("build_training_population.py", "POPULATION_INTERVAL_MINUTES"),
]
for fname, var in targets:
    p = here / fname
    src = p.read_text(encoding="utf-8")
    new_src, cnt = re.subn(rf"(?m)^{var} = \d+", f"{var} = {n}", src)
    assert cnt == 1, f"{fname}: pola {var} ketemu {cnt} kali"
    p.write_text(new_src, encoding="utf-8")
    print(f"{fname:32s} {var} = {n}")
