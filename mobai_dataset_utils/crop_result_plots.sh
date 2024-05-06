#!/bin/zsh

# needs imagemagick
if command -v convert &> /dev/null; then
    echo "imagemagick is installed."
else
    echo "imagemagick is not installed."
    return
fi


resultsPath="$1"

if [[ -v resultsPath ]]; then
  results=($resultsPath/**/*.png)

  for result in "${results[@]}"
  do 
    convert "$result" -crop 1200x450+320+60 "${result:0:-4}"_cropped.png;
  done
else
  echo "No results path given. Example use case: ./crop_result_plots.sh ./"
fi
  

