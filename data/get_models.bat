@echo off

cd %~dp0
powershell -Command "Invoke-WebRequest https://fox-gieg.com/patches/github/n1ckfg/Pix2PixP5/data/model.zip -OutFile model.zip"
powershell Expand-Archive model.zip -DestinationPath .
del model.zip

@pause