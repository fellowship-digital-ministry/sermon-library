@echo off
REM process_sermons.bat - Download new sermons, transcribe them, and generate embeddings
REM Usage: process_sermons.bat [CHANNEL_ID]
REM If CHANNEL_ID is not provided, the default Fellowship Baptist Church channel is used.

setlocal

set CHANNEL_ID=%1
if "%CHANNEL_ID%"=="" (
    set CHANNEL_ID=UCek_LI7dZopFJEvwxDnovJg
)

REM Step 1: monitor the channel and process new videos
python "%~dp0transcription\monitor_channel.py" --channel-id %CHANNEL_ID% --process --cleanup

REM Step 2: generate embeddings for any new transcripts
python "%~dp0tools\transcript_to_embeddings.py" --video_list_csv "%~dp0transcription\data\video_list.csv" --transcript_dir "%~dp0transcription\data\transcripts" --skip_existing

echo Pipeline completed.
endlocal

