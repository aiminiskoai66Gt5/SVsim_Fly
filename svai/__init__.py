"""AI opponents and headless tooling for SVsim.

This package must stay importable without tkinter so that headless runs
(fuzzing, arena, training) never touch GUI code.
"""
