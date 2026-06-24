# slurmwatch demo — render with:  vhs assets/demo.tape   (produces assets/demo.gif)
# Requires https://github.com/charmbracelet/vhs  (vhs ttyd ffmpeg)

Output assets/demo.gif
Set Shell bash
Set FontSize 17
Set Width 1300
Set Height 820
Set Padding 18
Set Framerate 24
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms

Type "SLURMWATCH_MOCK=1 slurmwatch"
Sleep 300ms
Enter
Sleep 3s
Type "m"
Sleep 1s
Type "c"
Sleep 1s
Type "q"
