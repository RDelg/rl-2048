#!/bin/bash

# build - A script build Dockerfile

##### Constants

mode=
image_name='ml-base'

##### Script to lunch build

build()
{
    case $mode in 
        s | SIMPLE )
        echo "Building from image"
        #### TODO: Maybe refresh
        docker build -t ${image_name}:latest .
        ;;
        c | CLOUD )
        echo "Building Cloud mode"
        #### TODO: Maybe we should add some validation here
        gcloud builds submit --timeout=1200 --tag gcr.io/$(gcloud config get-value project 2> /dev/null)/${image_name}:$SHORT_SHA --tag gcr.io/$(gcloud config get-value project 2> /dev/null)/${image_name}:latest . 
        ;;
    * )
    params
    exit 1
    esac
}

##### Instructions

usage()
{
    echo "usage: build.sh [-m --mode]"
}

params()
{
    echo "usage: build.sh -m [c CLOUD|s SIMPLE]"
}

if [ $# -gt 0 ]; then

    while [ "$1" != "" ]; do
        case $1 in -m | --mode )           
            shift
            mode=$1
            build
            ;;
            * )
            usage
            exit 1
        esac
        shift
    done
else
    usage
    exit 1
fi
