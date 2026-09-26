@echo off
rem =====================================================================
rem resume_kepler_build.cmd
rem
rem Resumes the Kepler TRAIN detection-views build (preprocessing_pipeline.py
rem build-train). The build is resumable: it skips KOIs that already have a
rem view file or a recorded ephemeris exclusion, so running this after any
rem interruption picks up where it stopped. Nothing is lost by re-running it.
rem
rem Cache moved to C:\lightkurve_cache on 2026-09-20: the original
rem D:\lightkurve_cache is gone (that drive letter now holds unrelated data),
rem so the ~46 GB of cached FITS no longer exists and the remaining stars are
rem re-downloaded. Kept off OneDrive on purpose - it is tens of GB of FITS.
rem
rem Launched detached via WMI Win32_Process.Create so it survives the session
rem that started it. Console output goes to the stdout/stderr files below;
rem the authoritative progress record is detection_views/kepler_train/
rem build_progress.txt and progress.jsonl.
rem =====================================================================

set "EXO_LIGHTKURVE_CACHE=C:\lightkurve_cache"
set "PY=C:\Users\ayush\AppData\Local\Python\pythoncore-3.14-64\python.exe"
set "RESEARCH=C:\Users\ayush\OneDrive\Desktop\VS\Research"
set "OUT=%RESEARCH%\detection_views\kepler_train"

cd /d "%RESEARCH%"
"%PY%" preprocessing_pipeline.py build-train --workers 2 1>> "%OUT%\stdout_resume6.txt" 2>> "%OUT%\stderr_resume6.txt"
