#!/usr/bin/zsh

# TODO: make os agnostic

regex='.*(ref_\d*_\d*|(probe|ref)_\d*)\.txt'

ls $1/**/*.txt | grep -v -P $regex
