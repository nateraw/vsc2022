# Need to install aws cli if you don't have it already...
# https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-linux.html

# curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
# unzip awscliv2.zip
# sudo ./aws/install

# Download the data
# reference - 59.4 GB with 40,311 object
# query - 13.8 GB with 8404 objects
TRAIN_DATA_DIR=./data/train/
mkdir -p $TRAIN_DATA_DIR
aws s3 cp s3://drivendata-competition-meta-vsc-data-us/train/ $TRAIN_DATA_DIR --recursive --region us-east-1 --no-sign-request


# TEST_DATA_DIR=./data/test
# mkdir -p $TEST_DATA_DIR
# aws s3 cp s3://drivendata-competition-meta-vsc-data-us/test/ $TEST_DATA_DIR --recursive --region us-east-1 --no-sign-request
