#!/bin/bash

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -i|--input)
        input_dir="$2"
        shift
        shift
        ;;
        -o|--output)
        output_dir="$2"
        shift
        shift
        ;;
        -p|--python)
        python_script="$2"
        shift
        shift
        ;;
        --postfix)
        postfix="$2"
        shift
        shift
        ;;
        *)
        python_args+=("$1")
        shift
        ;;
    esac
done

# Check that required arguments are set
if [[ -z "$input_dir" ]]; then
    echo "Input directory not specified"
    exit 1
fi
if [[ -z "$output_dir" ]]; then
    echo "Output directory not specified"
    exit 1
fi
if [[ -z "$python_script" ]]; then
    echo "Python script not specified"
    exit 1
fi
if [[ -z "$postfix" ]]; then
    echo "Postfix pattern not specified"
    exit 1
fi

# Find all files in the input directory with the specified postfix
find "$input_dir" -name "*$postfix" -type f | while read file; do
    # Get the relative path of the file (removing the input directory)
    rel_path="${file#$input_dir/}"

    # Create the output directory if it doesn't exist
    mkdir -p "$output_dir/$(dirname "$rel_path")"

    # replace the file extension with .npz
    rel_path="${rel_path%.*}.npz"

    # Build the command to run the python script with the specified arguments
    cmd=("python" "$python_script" "$file" "$output_dir/$rel_path")
    for arg in "${python_args[@]}"; do
        cmd+=("$arg")
    done

    # Run the command
    "${cmd[@]}"
done