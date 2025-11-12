#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR=""
OUTPUT_DIR=""
PROGARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        -i|--input)
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
          -h|--help)
            echo "Usage: $0 --input INPUT_DIR --output OUTPUT_DIR [-- other args...]"
            echo
            echo "Options:"
            echo "  -i, --input     Path to input directory containing .png files"
            echo "  -o, --output    Path to output directory (will be created if missing)"
            echo
            echo "Additional options are passed directly to dab.py."
            echo
            echo "--- dab.py help ---"
            python dab.py --help || true
            exit 0
            ;;
        --) # explicit end of script args
            shift
            PROGARGS=("$@")
            break
            ;;
        --*) # any other --flag is for the program
            PROGARGS+=("$1")
            shift
            # if it has a value, grab it too
            if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
                PROGARGS+=("$1")
                shift
            fi
            ;;
        -*)
            echo "Unknown short option: $1"
            exit 1
            ;;
        *)
            PROGARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" ]]; then
    echo "Usage: $0 --input INPUT_DIR --output OUTPUT_DIR [-- other args...]"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

find "$INPUT_DIR" -type f -name '*.png' \
| xargs -I {} basename {} .png \
| xargs -P 2 -I {} python dab.py \
    --input-file "$INPUT_DIR/{}.png" \
    --out-dir "$OUTPUT_DIR" \
    "${PROGARGS[@]}"
