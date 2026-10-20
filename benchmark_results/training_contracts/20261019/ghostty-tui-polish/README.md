# Ghostty TUI polish

Added searchable Ctrl+P commands, changed-row rendering, terminal-native semantic styles, responsive Escape respecting ESCDELAY, contextual workspace/setting headers, more visible retained runs, and settings page/Home/End navigation. Original launch/stop reviews and captured-run checks remain in place. Palette digits remain literal on compact screens.

90 unique tests passed (30 focused, 60 studio regressions), followed by 11 polish rechecks after type normalization. Full package type checking passed for 441 files. Ruff, Black, isort and repository hygiene passed. Black was checked outside the known sandbox worker-IPC restriction.

A live PTY opened the real curses interface, searched model commands with Ctrl+P, returned to Configure with Escape, and quit normally without launching work. Dedicated right-side Ghostty preview launches were attempted; its persistent window and screenshots were not independently confirmed, so no unrelated desktop captures are retained. This is terminal UX evidence, not model quality or a live provider test.
