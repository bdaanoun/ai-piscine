#!/bin/bash

repos=(
  "classification"
  "pipeline"
  "training"
  "model-selection"
  "neural-networks"
  "keras"
  "keras-2"
  "nlp"
  "nlp-spacy"
  "forest-prediction"
)

base_url="https://learn.zone01oujda.ma/git/bdaanoun"

get_branch() {
  git ls-remote --symref "$1" HEAD \
    | awk '/ref:/ {print $2}' \
    | sed 's#refs/heads/##'
}

for repo in "${repos[@]}"
do
  echo "Processing $repo"

  url="$base_url/$repo.git"

  branch=$(get_branch "$url")

  echo "Detected branch: $branch"

  git remote add "$repo" "$url" 2>/dev/null
  git fetch "$repo" "$branch"

  git subtree add --prefix="$repo" "$repo" "$branch"

  echo "Done $repo"
  echo "----------------------"
done

echo "All repositories imported!"