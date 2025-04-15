@echo off
setlocal enabledelayedexpansion

:: Configuration
set WAIT_MINUTES=5
set CSV_FILE=data/video_list.csv
set MAX_BATCHES=87
set CURRENT_BATCH=1

echo Starting sermon processing in small batches
echo Wait time between batches: %WAIT_MINUTES% minutes
echo CSV file: %CSV_FILE%
echo Total batches to process: %MAX_BATCHES%
echo.

:: Process videos in small batches using POC mode (5 at a time)
:process_loop
if %CURRENT_BATCH% GTR %MAX_BATCHES% goto end

echo.
echo Processing Batch %CURRENT_BATCH% of %MAX_BATCHES%
echo Time: %TIME%
echo.

:: Run Python script with POC mode (processes only 5 videos)
python process_batch.py --csv %CSV_FILE%

set /a CURRENT_BATCH=%CURRENT_BATCH%+1

if %CURRENT_BATCH% LEQ %MAX_BATCHES% (
    echo.
    echo Batch completed. Waiting %WAIT_MINUTES% minutes before next batch...
    echo Next batch will start at approximately:
    
    :: Calculate and display resume time
    for /f "tokens=1-3 delims=:." %%a in ("%TIME%") do (
        set /a HOUR=%%a
        set /a MINUTE=%%b+%WAIT_MINUTES%
        set /a HOUR_ADD=!MINUTE!/60
        set /a MINUTE=!MINUTE!%%60
        set /a HOUR=(!HOUR!+!HOUR_ADD!)%%24
        echo !HOUR!:!MINUTE!:%%c
    )
    
    :: Wait for specified minutes
    timeout /t %WAIT_MINUTES%0 /nobreak
    goto process_loop
)

:end
echo.
echo All batches completed!
echo Time: %TIME%