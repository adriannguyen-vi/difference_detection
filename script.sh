#!/bin/bash

OUTPUT_FILE="/home/ubuntu/adrian/drone_AI/progress_monitoring/comparision_result_C5/comparision_result_C5.zip"

aws s3 cp "$OUTPUT_FILE" s3://viact-adrian-storage-655384763347-ap-east-1-an/

# min-num-features:20000, skip-3dmodel:true, dsm:true, dtm:true, dem-euclidean-map:true, optimize-disk-space:true, skip-report:true
# ccdbed8a-7721-493f-a836-cadc28c3276c: new c1 r1  01 21
