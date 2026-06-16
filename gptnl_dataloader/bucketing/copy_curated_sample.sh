#!/bin/bash

PATH_TO_DATALOADER=${HOME}/dataloader
source ${PATH_TO_DATALOADER}/.env
source config.env
# echo "Copying curated sample from $CURATED_OUTPUT_DIR data to $COPY_CURATED_OUTPUT_DIR"
# rsync -avzP --stats --ignore-existing $CURATED_OUTPUT_DIR $COPY_CURATED_OUTPUT_DIR

# Make a symlink instead of copying
echo "Creating symlink for curated sample from $CURATED_OUTPUT_DIR to $COPY_CURATED_OUTPUT_DIR"
mkdir -p $(dirname "$COPY_CURATED_OUTPUT_DIR")
ln -s "$CURATED_OUTPUT_DIR" "$COPY_CURATED_OUTPUT_DIR"
echo "Creating symlink complete."
