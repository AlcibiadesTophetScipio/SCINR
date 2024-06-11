echo "Input num: $#" 
echo "GPU: $1"
echo "Dataset: $2"
echo "Training strategy: $3"
if [ $# -eq 0 ]; then
    export CUDA_VISIBLE_DEVICES="0"
else
    export CUDA_VISIBLE_DEVICES="$1"
fi
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# Find the path to the root of the repo.
SCRIPT_PATH=$(dirname $(realpath -s "$0"))
REPO_PATH=$(realpath -s "${SCRIPT_PATH}/..")

# Dataset setting
datasets_folder="${REPO_PATH}/datasets"
if [ "$2" == "Cambridge" ]; then
  scenes=("Cambridge/GreatCourt" "Cambridge/KingsCollege" "Cambridge/OldHospital" "Cambridge/ShopFacade" "Cambridge/StMarysChurch")
elif [ "$2" == "7scenes" ]; then
  scenes=("7scenes/chess" "7scenes/fire" "7scenes/heads" "7scenes/office" "7scenes/pumpkin" "7scenes/redkitchen" "7scenes/stairs")
elif [ "$2" == "12scenes" ]; then
  scenes=("12scenes/apt1_kitchen" "12scenes/apt1_living" "12scenes/apt2_bed" "12scenes/apt2_kitchen" "12scenes/apt2_living" "12scenes/apt2_luke" "12scenes/office1_gates362" "12scenes/office1_gates381" "12scenes/office1_lounge" "12scenes/office1_manolis" "12scenes/office2_5a" "12scenes/office2_5b")
else
  echo "Please set the right dataset."
  exit 1
fi

# Print results
function print_results {
  for scene in "${scenes[@]}"; do
    scene_mod=${scene#"$2/"}
    scene_dir=${out_dir}/$scene_mod
    # echo $scene $scene_mod $scene_dir
    if [ $3 == 'sep' ]; then
      echo "${scene_mod}: $(cat "${scene_dir}/test_log.txt" | tail -$1 | head -1)"
    elif [ $3 == 'together' ]; then
      echo "${scene_mod}: $(cat "${out_dir}/test_${scene_mod}_log.txt" | tail -$1 | head -1)"
    else
      echo "Please specify the right combination method."
      exit 1
    fi
  done
}

training_exe="${REPO_PATH}/trainer_base.py"
testing_exe="${REPO_PATH}/test.py"

if [ "$3" == "together" ]; then
  out_dir="${REPO_PATH}/output-ACE/$2-together"
  mkdir -p "$out_dir"
  if [ "$4" == "print" ]; then
    echo "Directly output results."
    print_results $5 $2 $3
    exit 0
  fi

  scenes_array=()
  for scene in "${scenes[@]}"; do
    scenes_array+=("${datasets_folder}/$scene")
  done
  # echo ${scenes_array[@]}

  # Train
  python $training_exe --scenes "${scenes_array[@]}" \
          --output_map_file "${out_dir}/$2.pt" \
          --training_buffer_size 8000000 \
          --epochs 16 \
          --regressor_select "ace" \
          2>&1 | tee "${out_dir}/train_log.txt"

  # Test
  for scene in "${scenes[@]}"; do
    scene_mod=${scene#"$2/"}
    python $testing_exe \
           --scene "${datasets_folder}/$scene" \
           --network "${out_dir}/$2.pt" \
           --regressor_select "ace" \
           2>&1 | tee "${out_dir}/test_${scene_mod}_log.txt"
  done
  
elif [ "$3" == "sep" ]; then
  out_dir="${REPO_PATH}/output-ACE/$2-sep"
  for scene in "${scenes[@]}"; do
    scene_mod=${scene#"$2/"}
    scene_dir=${out_dir}/$scene_mod
    mkdir -p "$scene_dir"
    if [ "$4" == "print" ]; then
      echo "Directly output results."
      print_results $5 $2 $3
      exit 0
    fi
    # echo $scene $scene_mod $scene_dir

    # Check datasets folder
    if [ ! -d "${datasets_folder}/$scene" ];then
      echo "${datasets_folder}/$scene does not exist."
      exit 1
    fi
    # Train
    python $training_exe \
           --scenes "${datasets_folder}/$scene" \
           --output_map_file "$scene_dir/${scene_mod}.pt" \
           --training_buffer_size 8000000 \
           --epochs 16 \
           --regressor_select "ace" \
           2>&1 | tee "$scene_dir/train_log.txt"
    # Test
    python $testing_exe \
           --scene "${datasets_folder}/$scene" \
           --network "${scene_dir}/${scene_mod}.pt" \
           --regressor_select "ace" \
           2>&1 | tee "$scene_dir/test_log.txt"
  done
fi

print_results 2 $2 $3