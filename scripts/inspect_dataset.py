"""Inspect LeRobot v2.1 metadata after installing this project."""
import sys

from robosyn_act.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["check-data", *sys.argv[1:]]))
