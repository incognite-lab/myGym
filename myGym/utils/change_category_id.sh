#!/bin/bash


help()
{
   # Display Help
   echo "Changes category id in the specified annotation json file."
   echo
   echo "Usage:"
   echo "sh change_category_id.sh old_category_id new_category_id annotation_file"
   echo "Example:"
   echo "sh change_category_id.sh 1 2 annotations.json"
   echo
}

help
sed -i  's|"category_id": '"$1"'|"category_id": '"$2"'|g' "$3"
sed -i  's|"id": '"$1"', "name"|"id": '"$2"', "name"|g' "$3"
