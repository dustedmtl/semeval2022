#!/bin/sh

if [ "X$1" = "X" ]; then
    echo "Give data file as argument"
    exit
fi

if [ "X$2" = "X" ]; then
    echo "Give submission name as an argument"
    exit
fi

datafile=$1

if [ ! -f $datafile ]; then
    echo "Not a file: $datafile"
    exit
fi

subname=$2

date=`date +"%Y%m%d"`

fullsub="${date}_$subname"
echo $fullsub
mkdir -p $fullsub/submission
cd $fullsub/submission && ln -sf ../../$datafile task2_subtaska.csv
zipname="${fullsub}.zip"
echo $zipname
cd .. && zip -rq1 $zipname submission
