# slurmwatch demo tape (alternative to assets/render_demo.py).
#
# The committed README hero (assets/demo.gif) is produced by:
#     python assets/render_demo.py
#
# This VHS tape records the *live* TUI in mock mode instead. Render with:
#     vhs assets/demo.tape            (requires https://github.com/charmbracelet/vhs)

Output assets/demo.gif
Set Shell bash
Set FontSize 16
Set Width 1320
Set Height 760
Set Padding 16
Set Framerate 24
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms

Type "SLURMWATCH_MOCK=1 slurmwatch"
Sleep 300ms
Enter
Sleep 3s
Type "c"
Sleep 1200ms
Type "m"
Sleep 1200ms
Type "g"
Sleep 1200ms
Type "v"
Sleep 1500ms
Type "q"
