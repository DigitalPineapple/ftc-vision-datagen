import sys

path = sys.argv[1]
new_mtllib_line = sys.argv[2]

with open(path, "r", encoding="utf-8") as f:
    first = f.readline()
    f.readline()  # discard old mtllib line
    rest = f.read()

with open(path, "w", encoding="utf-8") as f:
    f.write(first)
    f.write(new_mtllib_line + "\n")
    f.write(rest)
