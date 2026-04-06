#!/bin/bash

# Check if a file was provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 filename [column_number]"
    exit 1
fi

FILE=$1
COL=${2:-1} # Defaults to column 1 if not specified

# 1. Print the header (first line)
head -n 1 "$FILE"

# 2. Process the data (from line 2), round it, and sort
tail -n +2 "$FILE" | awk -v col="$COL" '{
    for (i=1; i<=NF; i++) {
        if (i == 1)
            printf "%s", $i
        else
            printf "%.2g", $i

        printf "%s", (i==NF ? "" : "\t")
    }
    printf "\n"
}' | sort -k"$COL","$COL"g | column -t
