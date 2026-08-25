#!/bin/bash

LOCAL_DIR="./results/log/"
REMOTE_USER="velarii"
REMOTE_IP="192.168.100.124"
REMOTE_PATH="/home/velarii/Proj/SciencePlots/data/side_plate_sim/"

upload() {
    scp "${LOCAL_DIR}$1" "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_PATH}"
}

upload "$1"
