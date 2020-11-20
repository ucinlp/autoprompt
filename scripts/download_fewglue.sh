#!/usr/bin/env bash

## Help text
Help()
{
  echo "Downloads FewGLUE and SuperGLUE datasets."
  echo
  echo "Usage:"
  echo "  scripts/download_fewglue.sh [DATA_DIR]"
  echo
}

if [[ $1 =~ ^(-h|--help) ]]; then
  Help
  exit
fi

## Main script
DATA_DIR=${1:-data}
SUPERGLUE_DIR=$DATA_DIR/SuperGLUE
SUPERGLUE_URL='https://dl.fbaipublicfiles.com/glue/superglue/data/v2/combined.zip'
FEWGLUE_DIR=$DATA_DIR/FewGLUE
FEWGLUE_URL='https://github.com/timoschick/fewglue.git'

# Download and extract SuperGLUE
if [ -d $SUPERGLUE_DIR ]; then
    echo "$SUPERGLUE_DIR already exists. Skipping SuperGLUE download."
else
    mkdir -p $SUPERGLUE_DIR
    wget -P $DATA_DIR $SUPERGLUE_URL 
    unzip $DATA_DIR/combined.zip -d $SUPERGLUE_DIR
    rm $DATA_DIR/combined.zip
fi

# Download FewGLUE
if [ -d $FEWGLUE_DIR ]; then
    echo "$FEWGLUE_DIR already exists. Skipping FewGLUE download."
else
    git clone $FEWGLUE_URL $FEWGLUE_DIR
    rm -rf $FEWGLUE_DIR/.git
    mv $FEWGLUE_DIR/FewGLUE/* $FEWGLUE_DIR
    rm -rf $FEWGLUE_DIR/FewGLUE
fi
