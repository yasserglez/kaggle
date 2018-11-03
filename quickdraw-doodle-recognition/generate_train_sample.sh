export JAVA_HOME="/usr/lib/jvm/java-8-openjdk-amd64"
export SPARK_LOCAL_DIRS="/mnt/data/spark/tmp"
export SPARK_DRIVER_MEMORY="16G"

python generate_train_sample.py 112640 768 7198021
