"""Strip the submission-packaging targets from a copied Makefile.

The packaging targets' scrub gates spell out the identifying-string patterns they
search for, so any anonymized artifact that ships the Makefile would de-anonymize
itself through its own scrubber. Used by `make submission-supplement` (and the
4open.science mirror preparation) to remove those targets from the shipped copy;
the reproduction targets reviewers actually need are untouched.

Usage: python3 scripts/strip_packaging_targets.py <path-to-Makefile>
"""
import sys

TARGETS = ("submission-archive:", "submission-supplement:")


def main(path: str) -> None:
    lines = open(path).readlines()
    out: list[str] = []
    skip = False
    for line in lines:
        if line.startswith(TARGETS):
            skip = True
            while out and out[-1].startswith("#"):
                out.pop()
            continue
        if skip:
            if line.startswith("\t"):
                continue
            if line.strip() == "":
                skip = False
                continue
            skip = False
        out.append(line)
    open(path, "w").writelines(out)


if __name__ == "__main__":
    main(sys.argv[1])
