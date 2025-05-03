#!/bin/bash
SearchPath="/home/qzhuang/code/llvm-project/build/Release/lib"
SymbolName="TypeIDResolver"

for file in "$SearchPath"/*.so; do
  if nm -D "$file" 2>/dev/null | grep -q "$SymbolName"; then
    echo "Symbol '$SymbolName' found in: $file"
  fi
done