#define MyAppName "Driver Drowsiness & Safety Monitoring"
#define MyAppVersion "1.0"
#define MyAppPublisher "Vishwa Tha Editor"
#define MyAppExeName "driver_drowsiness_gui.exe"

[Setup]
AppId={{5D4A9C4B-7E1B-4F2B-9C9B-1234567890AB}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={pf}\DrowsinessMonitor
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=no
OutputDir=release
OutputBaseFilename=DrowsinessMonitorSetup
Compression=lzma
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Files]
; EXE built by PyInstaller
Source: "release\driver_drowsiness_gui.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
; Start Menu shortcut
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"

; Desktop shortcut
Name: "{commondesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"

[Run]
; Run app automatically after install (optional)
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent
