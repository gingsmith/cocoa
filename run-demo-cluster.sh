#!/bin/bash

/root/spark/bin/spark-submit \
  --master `cat /root/spark-ec2/cluster-url` \
  --class "distopt.driver" \
  --driver-memory 80423M \
  --driver-java-options "-Dspark.local.dir=/mnt/spark,/mnt2/spark -XX:+UseG1GC" \
  target/scala-2.10/cocoa-assembly-0.1.jar \
  "$@"

