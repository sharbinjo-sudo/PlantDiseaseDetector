; ================================
; 🌿 Flutter Windows Installer Script (Updated for PlantDiseaseDL)
; ================================

[Setup]
AppName=Plant Disease Detector
AppVersion=1.0
AppPublisher=Forename Studios
DefaultDirName={pf}\Plant Disease Detector
DefaultGroupName=Plant Disease Detector
OutputDir=C:\Users\sharb\Documents\PlantDiseaseDL\plant_disease\build\installer
OutputBaseFilename=PlantDiseaseInstaller
SetupIconFile=C:\Users\sharb\Documents\PlantDiseaseDL\plant_disease\windows\runner\resources\app_icon.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=admin
UninstallDisplayIcon={app}\plant_disease.exe

[Files]
; ✅ Adjusted to the new build output path
Source: "C:\Users\sharb\Documents\PlantDiseaseDL\plant_disease\build\windows\x64\runner\Release\*"; \
    DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
; ✅ Create shortcuts for the new exe location
Name: "{group}\Plant Disease Detector"; Filename: "{app}\plant_disease.exe"
Name: "{commondesktop}\Plant Disease Detector"; Filename: "{app}\plant_disease.exe"

[Run]
; ✅ Launch app after installation
Filename: "{app}\plant_disease.exe"; \
    Description: "Launch Plant Disease Detector"; \
    Flags: nowait postinstall skipifsilent
