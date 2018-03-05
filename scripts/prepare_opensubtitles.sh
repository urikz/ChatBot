#!/bin/bash
set -e

OPENSUBTITLES="OpenSubtitles2018"
PREPROCESSED_DIR="preprocessed.deduplicate.truecase.max-length-20.min-length-2"
VOCAB_SIZE=50000

PROJECT_DIR="$(dirname $0)/.."

declare -A NUM_XML_FILES_MAP=( [OpenSubtitles]=2317 [OpenSubtitles2016]=323905 [OpenSubtitles2018]=446612 )

function assert_file_exists {
  if ! [ -e $1 ]
  then
    echo "File $1 does not exists"
    exit 1
  fi
}

function assert_files_have_same_num_lines {
  assert_file_exists $1
  assert_file_exists $2
  if [ $(wc -l $1 | cut -f 1 -d' ') -ne $(wc -l $2 | cut -f 1 -d' ') ]
  then
    echo "Files $1 and $2 have different number of lines!"
    exit 1
  fi
}

if [ -d "$OPENSUBTITLES" -a -d "$OPENSUBTITLES/xml/en" ]
then
  echo "XML files already exist. Skipping extraction..."
  NUM_XML_FILES=$(find "$OPENSUBTITLES/xml/en/" -name *.xml.gz -print | wc -l)
  if [ $NUM_XML_FILES -ne ${NUM_XML_FILES_MAP[$OPENSUBTITLES]} ]
  then
    echo "But number of xml files is not what we expected: $NUM_XML_FILES"
    exit 1
  fi
else
  if [ -e "$OPENSUBTITLES.tar.gz" ]
  then
    echo "File $OPENSUBTITLES.tar.gz already exists. Skipping download..."
  else
    wget "http://opus.lingfil.uu.se/download.php?f=$OPENSUBTITLES/en.tar.gz" -O "$OPENSUBTITLES.tar.gz"
  fi
  pigz --decompress --keep --stdout "$OPENSUBTITLES.tar.gz" | tar xf -
fi

mkdir -p "$OPENSUBTITLES/$PREPROCESSED_DIR"

if [ -e "$OPENSUBTITLES/$PREPROCESSED_DIR/train.tsv" ]
then
  echo "$OPENSUBTITLES/$PREPROCESSED_DIR/train.tsv already exists. Skipping data preprocessing..."
  assert_file_exists "$OPENSUBTITLES/$PREPROCESSED_DIR/tune.tsv"
  assert_file_exists "$OPENSUBTITLES/$PREPROCESSED_DIR/test.tsv"
else
  python "$PROJECT_DIR/scripts/prepare_opensubtitles.py" \
    --data "$OPENSUBTITLES/xml/en" \
    --out "$OPENSUBTITLES/$PREPROCESSED_DIR/train.tsv,$OPENSUBTITLES/$PREPROCESSED_DIR/tune.tsv,$OPENSUBTITLES/$PREPROCESSED_DIR/test.tsv" \
    --ratios 0.8,0.1,0.1 \
    --deduplicate \
    --max-length 20 \
    --min-length 2 \
    --num-workers $(nproc --all)
fi

FULL_VOCAB_FILE="$OPENSUBTITLES/$PREPROCESSED_DIR/train.vocab.full.tsv"

if [ -e "$FULL_VOCAB_FILE" ]
then
  echo "Vocabulary already exists. Skipping vocabulary building..."
  if [ $(wc -l $FULL_VOCAB_FILE | cut -f 1 -d' ') -lt 4 ]
  then
    echo "But it contains too few lines"
    exit 1
  fi
else
  echo -e "<PAD>\t-1" > "$FULL_VOCAB_FILE"
  echo -e "<GO>\t-1" >> "$FULL_VOCAB_FILE"
  echo -e "<EOS>\t-1" >> "$FULL_VOCAB_FILE"
  echo -e "<UNK>\t-1" >> "$FULL_VOCAB_FILE"
  cat "$OPENSUBTITLES/$PREPROCESSED_DIR/train.tsv" | tr '\t' '\n' | tr ' ' '\n' | sort -S 20G | uniq -c | sort -nr -S 20G | awk '{ print $2"\t"$1}' >> "$FULL_VOCAB_FILE"
fi

OUTPUT_DIR="$OPENSUBTITLES/$PREPROCESSED_DIR/vocab-$VOCAB_SIZE"

mkdir -p "$OUTPUT_DIR"

if [ -e "$OUTPUT_DIR/vocab.tsv" ]
then
  echo "Vocabulary of size $VOCAB_SIZE already exists. Skipping getting most frequent words..."
  if ! [ $(wc -l $OUTPUT_DIR/vocab.tsv | cut -f 1 -d' ') = $VOCAB_SIZE ]
  then
    echo "But it contains incorrect number of lines: $(wc -l $OUTPUT_DIR/vocab.tsv)"
    exit 1
  fi
else
  head -n $VOCAB_SIZE $FULL_VOCAB_FILE > "$OUTPUT_DIR/vocab.tsv"
fi

if [ -e "$OUTPUT_DIR/train.source.npz" ]
then
  echo "$OUTPUT_DIR/train.source.npz already exists. Skipping numberization and corpus packing..."
  assert_file_exists "$OUTPUT_DIR/train.target.npz"
  assert_file_exists "$OUTPUT_DIR/tune.source.npz"
  assert_file_exists "$OUTPUT_DIR/tune.target.npz"
  assert_file_exists "$OUTPUT_DIR/test.source.npz"
  assert_file_exists "$OUTPUT_DIR/test.target.npz"
else
  python $PROJECT_DIR/ShaLab/data/preprocess.py \
    --data "$OPENSUBTITLES/$PREPROCESSED_DIR/train.tsv" \
    --vocab "$OUTPUT_DIR/vocab.tsv" \
    --out "$OUTPUT_DIR/train" \
    --max-unk-ratio 0.2 \
    --max-length 20 \
    # --target-min-length 5

  python $PROJECT_DIR/ShaLab/data/preprocess.py \
    --data "$OPENSUBTITLES/$PREPROCESSED_DIR/tune.tsv" \
    --vocab "$OUTPUT_DIR/vocab.tsv" \
    --out "$OUTPUT_DIR/tune" \
    --max-unk-ratio 0.2 \
    --max-length 20 \
    # --target-min-length 5

  python $PROJECT_DIR/ShaLab/data/preprocess.py \
    --data "$OPENSUBTITLES/$PREPROCESSED_DIR/test.tsv" \
    --vocab "$OUTPUT_DIR/vocab.tsv" \
    --out "$OUTPUT_DIR/test" \
    --max-unk-ratio 0.2 \
    --max-length 20 \
    # --target-min-length 5
fi
