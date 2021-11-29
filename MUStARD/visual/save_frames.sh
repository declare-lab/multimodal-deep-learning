#!/usr/bin/env bash

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"
pushd "${SCRIPT_DIR}/.." > /dev/null

videos_folder_path=data/videos/utterances_final
frames_folder_path=data/frames/utterances_final
ext=mp4

mkdir "${frames_folder_path}"

for video_file_path in "${videos_folder_path}"/*."${ext}"; do
    slash_and_video_file_name="${video_file_path:${#videos_folder_path}}"
    slash_and_video_file_name_without_extension="${slash_and_video_file_name%.${ext}}"
    video_frames_folder_path="${frames_folder_path}${slash_and_video_file_name_without_extension}";
    mkdir "${video_frames_folder_path}"
    ffmpeg -i "${video_file_path}" "${video_frames_folder_path}/%05d.jpg"
done

popd > /dev/null
