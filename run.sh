#!/bin/bash

# Video size setting
OUTPUT_SIZE_W=1280
OUTPUT_SIZE_H=720

# Copy-and-Paste setting
MODE=O

filename='video_folder.txt'
fileNum=0
while read line
do
    path[$fileNum]=$(echo $line | cut -d" " -f 1)
    target[$fileNum]=$(echo $line | cut -d" " -f 2)
    fileNum=`expr $fileNum + 1`
    
done < "$filename"

for ((index=0; index<$fileNum; index++))
do
    echo 'Start to generate boundingbox...'
    cd ./maskRCNN
    python getBoundingBox.py --videoPath ${path[$index]} --target ${target[$index]} --outputSize $OUTPUT_SIZE_W $OUTPUT_SIZE_H
    cd ..
    echo 'generate boundingbox finished.'

    echo 'Start to generate mask video...'
    cd ./segmentation
    python genMaskVideo.py --videoPath ${path[$index]} --outputSize $OUTPUT_SIZE_W $OUTPUT_SIZE_H
    cd ..
    echo 'Generate mask video finished.'

    echo 'Start to run alignment network...'
    cd ./inpainting
    python CPNet.py -g 0 -D ./
    cd ..
    echo 'Alignment work finished.'

    echo 'Start to run Copy-and-Paste...'
    cd ./inpainting
    python myACP.py --mode $MODE
    cd ..
    echo 'Program finished.'
done
